from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pygame

import physics


FOCUS_ORDER = ("cells", "chart", "console")


@dataclass(frozen=True)
class Command:
    """A console command. A usage ending in '...' takes its last argument as the rest of the line."""

    usage: str
    description: str
    handler: Callable[[object, list[str]], None]

    @property
    def name(self) -> str:
        return self.usage.split()[0]

    @property
    def arity(self) -> int:
        return len(self.usage.split()) - 1

    def accepts(self, args: list[str]) -> bool:
        if self.usage.endswith("..."):
            return len(args) >= self.arity
        return len(args) == self.arity


COMMANDS = [
    Command("help", "List console commands", lambda app, a: app.print_commands()),
    Command("ls", "List saved snapshots", lambda app, a: app.list_snapshots()),
    Command("save NAME", "Save the current snapshot", lambda app, a: app.save_snapshot(a[0])),
    Command("load NAME", "Load a snapshot", lambda app, a: app.try_load_snapshot(a[0])),
    Command("name INDEX NAME...", "Name a probe (shown on grid instead of its index)", lambda app, a: app.name_probe(a[0], " ".join(a[1:]))),
    Command("strike LABEL", "Strike the cell under a labelled probe", lambda app, a: app.console_strike(a[0])),
    Command("strike X Y", "Strike a cell", lambda app, a: app.console_strike_xy(a[0], a[1])),
    Command("set VAR VALUE", "Set a parameter", lambda app, a: app.console_set(a[0], a[1])),
    Command("run SECONDS", "Pause, advance by SECONDS immediately, stay paused", lambda app, a: app.console_run(a[0])),
    Command("pause", "Pause the simulation", lambda app, a: app.set_paused(True)),
    Command("resume", "Resume the simulation", lambda app, a: app.set_paused(False)),
    Command("stats", "Print grid-wide activity summary", lambda app, a: app.console_stats()),
    Command("probes", "Print each probe's live energy/flame/illumination", lambda app, a: app.console_probes()),
    Command("inspect X Y", "Print one cell's kind, energy, flame, weight, illumination and teaching signal", lambda app, a: app.console_inspect(a[0], a[1])),
    Command("params", "Print the current parameter values", lambda app, a: app.print_var_list()),
    Command("clear", "Clear the enabled pattern and zero activity", lambda app, a: app.clear_enabled()),
    Command("fill", "Enable every cell", lambda app, a: app.fill_enabled()),
    Command("enable X Y", "Make a normal cell", lambda app, a: app.console_set_kind(a[0], a[1], physics.NORMAL)),
    Command("good X Y", "Make a GOOD teacher cell: gives no light, never lit; striking it rewards what just fired beside it", lambda app, a: app.console_set_kind(a[0], a[1], physics.GOOD)),
    Command("bad X Y", "Make a BAD teacher cell: gives no light, never lit; striking it punishes what just fired beside it", lambda app, a: app.console_set_kind(a[0], a[1], physics.BAD)),
    Command("disable X Y", "Remove a cell", lambda app, a: app.console_set_kind(a[0], a[1], physics.EMPTY)),
    Command("settle", "Flame off, energy full where enabled", lambda app, a: app.zero_activity()),
    Command("unlearn", "Reset every cell's weight to 1.0 and its heat to 0 (forget all learning)", lambda app, a: app.unlearn()),
    Command("probe X Y", "Add a chart probe", lambda app, a: app.console_probe(a[0], a[1])),
    Command("deleteprobe X Y", "Delete the chart probe at a cell", lambda app, a: app.console_deleteprobe(a[0], a[1])),
    Command("twophase SEED", "Run the learning experiment live: adaptation grows routes, then reward teaches a, then b", lambda app, a: app.console_twophase(a[0])),
    Command("twophase stop", "Stop the learning experiment", lambda app, a: app.console_twophase("stop")),
    Command("robot SEED", "Run the robot, with red as food: a sheet in a round body learning which colour is food", lambda app, a: app.console_robot(a)),
    Command("robot SEED FOOD", "Run the robot with FOOD (red or blue) as food", lambda app, a: app.console_robot(a)),
    Command("robot stop", "Stop the robot", lambda app, a: app.console_robot(a)),
    Command("speed N", "Run N simulation steps per frame (1..50)", lambda app, a: app.console_speed(a[0])),
    Command("grid SIZE", "Resize the grid to SIZE x SIZE, keeping the pattern centred", lambda app, a: app.console_grid(a[0])),
    Command("shift DX DY", "Shift the whole pattern and its probes", lambda app, a: app.console_shift(a[0], a[1])),
    Command("undo", "Undo the last change", lambda app, a: app.undo()),
    Command("quit", "Quit the simulator", lambda app, a: app.request_quit()),
]


def find_command(words: list[str]) -> Command | None:
    for command in COMMANDS:
        if command.name == words[0] and command.accepts(words[1:]):
            return command
    return None


@dataclass(frozen=True)
class HelpEntry:
    section: str
    trigger: str
    description: str


@dataclass(frozen=True)
class KeyAction:
    section: str
    scope: str | None
    trigger: str
    description: str
    matches: Callable[[pygame.event.Event], bool]
    handler: Callable[[object], None]


def _plain_key(key: int) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and not (event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_META | pygame.KMOD_SHIFT))


def _shift_key(key: int) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and bool(event.mod & pygame.KMOD_SHIFT) and not (event.mod & pygame.KMOD_CTRL)


def _ctrl_shift_key(key: int) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and bool(event.mod & pygame.KMOD_SHIFT) and bool(event.mod & pygame.KMOD_CTRL)


def _ctrl_key(key: int) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and bool(event.mod & pygame.KMOD_CTRL)


def _shift_char(key: int, char: str) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and bool(event.mod & pygame.KMOD_SHIFT) and getattr(event, "unicode", "") == char


def _letter_key(key: int) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and not (event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_META))


def _unmodified_arrow(key: int) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and not (event.mod & pygame.KMOD_SHIFT)


# Keys that do the same thing as a console command submit that command, so it is echoed and logged.
KEY_ACTIONS = [
    KeyAction("Global", None, "Esc", "Quit", lambda event: event.key == pygame.K_ESCAPE, lambda app: app.submit("quit")),
    KeyAction("Global", None, "Ctrl+C", "Quit", _ctrl_key(ord("c")), lambda app: app.submit("quit")),
    KeyAction("Global", None, "Tab", "Cycle focus", lambda event: event.key == pygame.K_TAB, lambda app: app.cycle_focus()),
    KeyAction("Global", None, "h", "Toggle help overlay", _plain_key(ord("h")), lambda app: app.toggle_help()),
    KeyAction("Global", None, "?", "Toggle help overlay", _shift_char(pygame.K_SLASH, "?"), lambda app: app.toggle_help()),
    KeyAction("Global", None, "v", "Start/stop MPEG recording", _plain_key(ord("v")), lambda app: app.toggle_recording()),
    KeyAction("Global", None, "U", "Undo", _letter_key(ord("u")), lambda app: app.submit("undo")),
    KeyAction("Chart", "chart", "-", "Zoom out in time", _plain_key(ord("-")), lambda app: app.chart_zoom_out()),
    KeyAction("Chart", "chart", "=", "Zoom in in time", _plain_key(ord("=")), lambda app: app.chart_zoom_in()),
    KeyAction("Chart", "chart", "t", "Toggle chart retrigger", _plain_key(ord("t")), lambda app: app.toggle_chart_retrigger()),
    KeyAction("Grid", "cells", "=", "Zoom in grid (grid SIZE/2)", _plain_key(ord("=")), lambda app: app.submit(f"grid {app.state.grid_size[0] // 2}")),
    KeyAction("Grid", "cells", "-", "Zoom out grid (grid SIZE*2)", _plain_key(ord("-")), lambda app: app.submit(f"grid {app.state.grid_size[0] * 2}")),
    KeyAction("Grid", "cells", "Space", "Single-step while paused", lambda event: event.key == pygame.K_SPACE, lambda app: app.single_step()),
    KeyAction("Grid", "cells", "a", "Add probe under mouse", _plain_key(ord("a")), lambda app: app.submit_at_mouse("probe {x} {y}")),
    KeyAction("Grid", "cells", "c", "Clear enabled pattern", _plain_key(ord("c")), lambda app: app.submit("clear")),
    KeyAction("Grid", "cells", "d", "Delete probe under mouse", _plain_key(ord("d")), lambda app: app.submit_at_mouse("deleteprobe {x} {y}")),
    KeyAction("Grid", "cells", "e", "Toggle energy-only view", _plain_key(ord("e")), lambda app: app.toggle_show_only("E")),
    KeyAction("Grid", "cells", "f", "Fill grid with enabled cells", _plain_key(ord("f")), lambda app: app.submit("fill")),
    KeyAction("Grid", "cells", "i", "Toggle illumination-only view", _plain_key(ord("i")), lambda app: app.toggle_show_only("I")),
    KeyAction("Grid", "cells", "w", "Toggle weight-only view", _plain_key(ord("w")), lambda app: app.toggle_show_only("W")),
    KeyAction("Grid", "cells", "n", "Name probe under mouse", _plain_key(ord("n")), lambda app: app.name_probe_under_mouse()),
    KeyAction("Grid", "cells", "p", "Pause/resume", _plain_key(ord("p")), lambda app: app.submit("resume" if app.paused else "pause")),
    KeyAction("Grid", "cells", "r", "Randomly strike cells", _plain_key(ord("r")), lambda app: app.random_strike()),
    KeyAction("Grid", "cells", "z", "Zero flame, reset energy to full where enabled", _plain_key(ord("z")), lambda app: app.submit("settle")),
    KeyAction("Grid", "cells", "Shift+Z", "Forget all learning: every weight back to 1.0", _shift_key(ord("z")), lambda app: app.submit("unlearn")),
    KeyAction("Grid", "cells", "Ctrl+Shift+Up", "Split at cursor and move upper cells up", _ctrl_shift_key(pygame.K_UP), lambda app: app.split_shift_state(0, -1)),
    KeyAction("Grid", "cells", "Ctrl+Shift+Down", "Split at cursor and move lower cells down", _ctrl_shift_key(pygame.K_DOWN), lambda app: app.split_shift_state(0, 1)),
    KeyAction("Grid", "cells", "Ctrl+Shift+Left", "Split at cursor and move left cells left", _ctrl_shift_key(pygame.K_LEFT), lambda app: app.split_shift_state(-1, 0)),
    KeyAction("Grid", "cells", "Ctrl+Shift+Right", "Split at cursor and move right cells right", _ctrl_shift_key(pygame.K_RIGHT), lambda app: app.split_shift_state(1, 0)),
    KeyAction("Grid", "cells", "Shift+Up", "Shift whole pattern up", _shift_key(pygame.K_UP), lambda app: app.submit("shift 0 -1")),
    KeyAction("Grid", "cells", "Shift+Down", "Shift whole pattern down", _shift_key(pygame.K_DOWN), lambda app: app.submit("shift 0 1")),
    KeyAction("Grid", "cells", "Shift+Left", "Shift whole pattern left", _shift_key(pygame.K_LEFT), lambda app: app.submit("shift -1 0")),
    KeyAction("Grid", "cells", "Shift+Right", "Shift whole pattern right", _shift_key(pygame.K_RIGHT), lambda app: app.submit("shift 1 0")),
    KeyAction("Grid", "cells", "Up", "Select previous parameter", _unmodified_arrow(pygame.K_UP), lambda app: app.select_previous_var()),
    KeyAction("Grid", "cells", "Down", "Select next parameter", _unmodified_arrow(pygame.K_DOWN), lambda app: app.select_next_var()),
    KeyAction("Grid", "cells", "Left", "Decrease selected parameter (set VAR VALUE)", _unmodified_arrow(pygame.K_LEFT), lambda app: app.submit_selected_var_scaled(1 / 1.05)),
    KeyAction("Grid", "cells", "Right", "Increase selected parameter (set VAR VALUE)", _unmodified_arrow(pygame.K_RIGHT), lambda app: app.submit_selected_var_scaled(1.05)),
]


STATIC_HELP = [
    HelpEntry("Mouse", "Left click on grid", "Toggle the enabled state of a cell"),
    HelpEntry("Mouse", "Left drag on grid", "Paint more cells with the same state"),
    HelpEntry("Mouse", "Ctrl+click on grid", "Toggle a GOOD teacher cell (good X Y / disable X Y)"),
    HelpEntry("Mouse", "Right click on grid", "Strike a cell and retrigger the chart"),
    HelpEntry("Mouse", "Click a panel", "Move focus to grid, chart, or console"),
    HelpEntry("Console", "Enter", "Execute the current console command"),
    HelpEntry("Console", "Backspace", "Delete one character from the console input"),
]


def dispatch_keydown(app, event: pygame.event.Event) -> bool:
    for action in KEY_ACTIONS:
        if action.scope is not None and action.scope != app.focus_name:
            continue
        if action.matches(event):
            action.handler(app)
            return True
    return False


def build_help_lines() -> list[str]:
    lines = ["Press h to close help.", ""]
    sections = ["Global", "Mouse", "Grid", "Chart", "Console"]
    entries_by_section: dict[str, list[tuple[str, str]]] = {section: [] for section in sections}

    for action in KEY_ACTIONS:
        entries_by_section[action.section].append((action.trigger, action.description))
    for entry in STATIC_HELP:
        entries_by_section[entry.section].append((entry.trigger, entry.description))
    for command in COMMANDS:
        entries_by_section["Console"].append((command.usage, command.description))

    for section in sections:
        lines.append(f"{section}:")
        for trigger, description in entries_by_section[section]:
            lines.append(f"{trigger:16} {description}")
        lines.append("")

    return lines
