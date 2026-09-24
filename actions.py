from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pygame


FOCUS_ORDER = ("cells", "chart", "console")


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
    return lambda event: event.key == key and bool(event.mod & pygame.KMOD_SHIFT)


def _ctrl_shift_key(key: int) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and bool(event.mod & pygame.KMOD_SHIFT) and bool(event.mod & pygame.KMOD_CTRL)


def _ctrl_key(key: int) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and bool(event.mod & pygame.KMOD_CTRL)


def _shift_char(key: int, char: str) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and bool(event.mod & pygame.KMOD_SHIFT) and getattr(event, "unicode", "") == char


def _letter_key(key: int) -> Callable[[pygame.event.Event], bool]:
    return lambda event: event.key == key and not (event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_META))


KEY_ACTIONS = [
    KeyAction("Global", None, "Esc", "Quit", lambda event: event.key == pygame.K_ESCAPE, lambda app: app.request_quit()),
    KeyAction("Global", None, "Ctrl+C", "Quit", _ctrl_key(ord("c")), lambda app: app.request_quit()),
    KeyAction("Global", None, "Tab", "Cycle focus", lambda event: event.key == pygame.K_TAB, lambda app: app.cycle_focus()),
    KeyAction("Global", None, "h", "Toggle help overlay", _plain_key(ord("h")), lambda app: app.toggle_help()),
    KeyAction("Global", None, "?", "Toggle help overlay", _shift_char(pygame.K_SLASH, "?"), lambda app: app.toggle_help()),
    KeyAction("Global", None, "v", "Start/stop MPEG recording", _plain_key(ord("v")), lambda app: app.toggle_recording()),
    KeyAction("Global", None, "U", "Undo", _letter_key(ord("u")), lambda app: app.undo()),
    KeyAction("Chart", "chart", "-", "Zoom out in time", _plain_key(ord("-")), lambda app: app.chart_zoom_out()),
    KeyAction("Chart", "chart", "=", "Zoom in in time", _plain_key(ord("=")), lambda app: app.chart_zoom_in()),
    KeyAction("Chart", "chart", "t", "Toggle chart retrigger", _plain_key(ord("t")), lambda app: app.toggle_chart_retrigger()),
    KeyAction("Grid", "cells", "=", "Zoom in grid", _plain_key(ord("=")), lambda app: app.decrease_grid_size()),
    KeyAction("Grid", "cells", "-", "Zoom out grid", _plain_key(ord("-")), lambda app: app.increase_grid_size()),
    KeyAction("Grid", "cells", "Space", "Single-step while paused", lambda event: event.key == pygame.K_SPACE, lambda app: app.single_step()),
    KeyAction("Grid", "cells", "a", "Add probe under mouse", _plain_key(ord("a")), lambda app: app.add_probe_under_mouse()),
    KeyAction("Grid", "cells", "c", "Clear enabled pattern", _plain_key(ord("c")), lambda app: app.clear_enabled()),
    KeyAction("Grid", "cells", "d", "Delete probe under mouse", _plain_key(ord("d")), lambda app: app.delete_probe_under_mouse()),
    KeyAction("Grid", "cells", "e", "Toggle energy-only view", _plain_key(ord("e")), lambda app: app.toggle_show_only("E")),
    KeyAction("Grid", "cells", "f", "Fill grid with enabled cells", _plain_key(ord("f")), lambda app: app.fill_enabled()),
    KeyAction("Grid", "cells", "i", "Toggle illumination-only view", _plain_key(ord("i")), lambda app: app.toggle_show_only("I")),
    KeyAction("Grid", "cells", "n", "Name probe under mouse", _plain_key(ord("n")), lambda app: app.name_probe_under_mouse()),
    KeyAction("Grid", "cells", "p", "Pause/resume", _plain_key(ord("p")), lambda app: app.toggle_pause()),
    KeyAction("Grid", "cells", "r", "Randomly strike cells", _plain_key(ord("r")), lambda app: app.random_strike()),
    KeyAction("Grid", "cells", "z", "Zero flame, reset energy to full where enabled", _plain_key(ord("z")), lambda app: app.zero_activity()),
    KeyAction("Grid", "cells", "Ctrl+Shift+Up", "Split at cursor and move upper cells up", _ctrl_shift_key(pygame.K_UP), lambda app: app.split_shift_state(0, -1)),
    KeyAction("Grid", "cells", "Ctrl+Shift+Down", "Split at cursor and move lower cells down", _ctrl_shift_key(pygame.K_DOWN), lambda app: app.split_shift_state(0, 1)),
    KeyAction("Grid", "cells", "Ctrl+Shift+Left", "Split at cursor and move left cells left", _ctrl_shift_key(pygame.K_LEFT), lambda app: app.split_shift_state(-1, 0)),
    KeyAction("Grid", "cells", "Ctrl+Shift+Right", "Split at cursor and move right cells right", _ctrl_shift_key(pygame.K_RIGHT), lambda app: app.split_shift_state(1, 0)),
    KeyAction("Grid", "cells", "Shift+Up", "Shift whole pattern up", _shift_key(pygame.K_UP), lambda app: app.shift_state(0, -1)),
    KeyAction("Grid", "cells", "Shift+Down", "Shift whole pattern down", _shift_key(pygame.K_DOWN), lambda app: app.shift_state(0, 1)),
    KeyAction("Grid", "cells", "Shift+Left", "Shift whole pattern left", _shift_key(pygame.K_LEFT), lambda app: app.shift_state(-1, 0)),
    KeyAction("Grid", "cells", "Shift+Right", "Shift whole pattern right", _shift_key(pygame.K_RIGHT), lambda app: app.shift_state(1, 0)),
    KeyAction("Grid", "cells", "Up", "Select previous parameter", lambda event: event.key == pygame.K_UP and not (event.mod & pygame.KMOD_SHIFT), lambda app: app.select_previous_var()),
    KeyAction("Grid", "cells", "Down", "Select next parameter", lambda event: event.key == pygame.K_DOWN and not (event.mod & pygame.KMOD_SHIFT), lambda app: app.select_next_var()),
    KeyAction("Grid", "cells", "Left", "Decrease selected parameter", lambda event: event.key == pygame.K_LEFT and not (event.mod & pygame.KMOD_SHIFT), lambda app: app.scale_selected_var(1 / 1.05)),
    KeyAction("Grid", "cells", "Right", "Increase selected parameter", lambda event: event.key == pygame.K_RIGHT and not (event.mod & pygame.KMOD_SHIFT), lambda app: app.scale_selected_var(1.05)),
]


STATIC_HELP = [
    HelpEntry("Mouse", "Left click on grid", "Toggle the enabled state of a cell"),
    HelpEntry("Mouse", "Left drag on grid", "Paint more cells with the same state"),
    HelpEntry("Mouse", "Right click on grid", "Strike a cell and retrigger the chart"),
    HelpEntry("Mouse", "Click a panel", "Move focus to grid, chart, or console"),
    HelpEntry("Console", "Enter", "Execute the current console command"),
    HelpEntry("Console", "Backspace", "Delete one character from the console input"),
    HelpEntry("Console", "ls", "List available snapshots"),
    HelpEntry("Console", "save NAME", "Save the current snapshot"),
    HelpEntry("Console", "load NAME", "Load a snapshot"),
    HelpEntry("Console", "name INDEX NAME", "Name a probe (shown on grid instead of its index)"),
    HelpEntry("Console", "strike X Y | strike LABEL", "Strike a cell by coordinate or probe label"),
    HelpEntry("Console", "set VAR VALUE", "Set a parameter directly"),
    HelpEntry("Console", "step N", "Advance N simulation ticks immediately"),
    HelpEntry("Console", "stats", "Print grid-wide activity summary"),
    HelpEntry("Console", "probes", "Print each probe's live energy/flame/illumination"),
    HelpEntry("Console", "inspect X Y", "Print one cell's live energy/flame/illumination"),
    HelpEntry("Console", "vars", "Print the current parameter values"),
    HelpEntry("Console", "quit", "Quit the simulator"),
    HelpEntry("Console", "clear", "Clear the enabled pattern and zero activity"),
    HelpEntry("Console", "enable X Y", "Enable a cell"),
    HelpEntry("Console", "disable X Y", "Disable a cell"),
    HelpEntry("Console", "settle", "Flame off, energy full where enabled"),
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

    for section in sections:
        lines.append(f"{section}:")
        for trigger, description in entries_by_section[section]:
            lines.append(f"{trigger:16} {description}")
        lines.append("")

    return lines
