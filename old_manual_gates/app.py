from __future__ import annotations

import copy
import dataclasses
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

from actions import COMMANDS, FOCUS_ORDER, build_help_lines, dispatch_keydown, find_command
import physics
from headless import grid_summary, probe_by_label
from model import SimulationConfig, State, centre_offset, resized_state
from persistence import list_snapshot_names, load_snapshot, save_snapshot
from recording import VideoRecorder
from views import CellsPanel, ChartPanel, ConsolePanel, HelpOverlay, WorldPanel


KIND_NAMES = {physics.EMPTY: "empty", physics.NORMAL: "normal", physics.GOOD: "good", physics.BAD: "bad"}

CONTROL_IN_PATH = Path("control_in.txt")
CONTROL_OUT_PATH = Path("control_out.log")


@dataclass
class UndoSnapshot:
    state: State
    probes: list[dict]
    config_vars: dict[str, float]
    selected_var: int
    show_only: str


FOCUS_CLICK_GRACE_S = 0.3   # a click this soon after the window gains focus is the focusing click


class _Tee:
    """Writes to several file-like sinks at once (used for sys.stdout)."""

    def __init__(self, *sinks) -> None:
        self.sinks = sinks

    def write(self, string: str) -> None:
        for sink in self.sinks:
            sink.write(string)

    def flush(self) -> None:
        for sink in self.sinks:
            sink.flush()


class SimulatorApp:
    def __init__(self, display_index: int = 0):
        pygame.init()
        pygame.key.set_repeat(500, 100)

        displays = pygame.display.get_desktop_sizes()
        self.display_index = max(0, min(display_index, len(displays) - 1))
        self.screen_width, self.screen_height = displays[self.display_index]
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            display=self.display_index,
            flags=pygame.FULLSCREEN | pygame.SCALED,
        )
        pygame.display.set_caption("Neuron 2025")

        self.config = SimulationConfig()
        self.selected_var = 0
        self.trainer = None
        self.speed = 1
        self.state = State((32, 32))
        self.show_only = ""
        self.running = True
        self.paused = False
        self.do_step = False
        self.help_visible = False
        self.dragging_state = False
        self.focus_name = "cells"
        self.last_mouse_pos = (0, 0)
        self.undo_stack: list[UndoSnapshot] = []
        self.ignore_next_textinput = False
        self.window_focused = True
        self.focus_gained_at = 0.0

        # four quarters: cells top left, world top right, chart bottom left, console bottom right
        half_w, half_h = self.screen_width // 2, self.screen_height // 2
        right, bottom = (self.screen_width - half_w, self.screen_height - half_h)
        side = min(half_w, half_h)
        self.cells_panel = CellsPanel(self.screen, (0, 0), (side, side), self.state.grid_size)
        self.world_panel = WorldPanel(self.screen, (half_w, 0), (right, half_h))
        self.chart_panel = ChartPanel(self.screen, (0, half_h), (half_w, bottom))
        self.console_panel = ConsolePanel(self.screen, (half_w, half_h), (right, bottom), self.submit)
        self.help_overlay = HelpOverlay(self.screen)
        self.help_lines = build_help_lines()
        self.recorder = VideoRecorder(fps=30)

        self.original_stdout = sys.stdout
        self.control_log = open(CONTROL_OUT_PATH, "a", buffering=1)
        sys.stdout = _Tee(self.console_panel, self.control_log)
        CONTROL_IN_PATH.write_text("")

        self.set_focus("cells")
        print("Hello everyone")
        self.try_load_snapshot("recent")

    def request_quit(self) -> None:
        self.running = False

    def set_focus(self, focus_name: str) -> None:
        self.focus_name = focus_name
        self.cells_panel.focus = focus_name == "cells"
        self.chart_panel.focus = focus_name == "chart"
        self.console_panel.focus = focus_name == "console"

    def cycle_focus(self) -> None:
        index = FOCUS_ORDER.index(self.focus_name)
        self.set_focus(FOCUS_ORDER[(index + 1) % len(FOCUS_ORDER)])

    def toggle_help(self) -> None:
        self.help_visible = not self.help_visible

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        print("Paused" if paused else "Resumed")

    def single_step(self) -> None:
        self.paused = True
        self.do_step = True

    def toggle_show_only(self, mode: str) -> None:
        self.show_only = "" if self.show_only == mode else mode

    def make_undo_snapshot(self) -> UndoSnapshot:
        return UndoSnapshot(
            state=self.state.clone(),
            probes=copy.deepcopy(self.chart_panel.probes),
            config_vars=copy.deepcopy(self.config.vars),
            selected_var=self.selected_var,
            show_only=self.show_only,
        )

    def push_undo_state(self) -> None:
        self.undo_stack.append(self.make_undo_snapshot())

    def restore_undo_snapshot(self, snapshot: UndoSnapshot) -> None:
        self.state = snapshot.state
        self.cells_panel.set_grid_size(self.state.grid_size)
        self.chart_panel.set_probes(copy.deepcopy(snapshot.probes))
        self.config.vars = copy.deepcopy(snapshot.config_vars)
        self.selected_var = snapshot.selected_var
        self.show_only = snapshot.show_only

    def undo(self) -> None:
        if not self.undo_stack:
            print("Nothing to undo")
            return
        snapshot = self.undo_stack.pop()
        self.restore_undo_snapshot(snapshot)
        print("Undone")

    def current_cell(self) -> tuple[bool, tuple[int, int]]:
        hit, grid_x, grid_y = self.cells_panel.find_cell(self.last_mouse_pos)
        return hit, (grid_x, grid_y)

    def set_enabled(self, xy: tuple[int, int], enabled: bool) -> None:
        self.state.set_kind(xy, physics.NORMAL if enabled else physics.EMPTY)

    def get_enabled(self, xy: tuple[int, int]) -> bool:
        return bool(self.state.kind_array[xy] != physics.EMPTY)

    def strike_cell(self, cell_xy: tuple[int, int]) -> bool:
        return self.state.strike(cell_xy, self.config)

    def name_probe_under_mouse(self) -> None:
        hit, cell_xy = self.current_cell()
        if not hit:
            return
        index = self.chart_panel.find_probe_index(cell_xy)
        if index is None:
            print("No probe under mouse")
            return
        self.set_focus("console")
        self.console_panel.input = f"name {index} "
        self.ignore_next_textinput = True

    def name_probe(self, index_str: str, name: str) -> None:
        try:
            index = int(index_str)
        except ValueError:
            print(f"Invalid probe index '{index_str}'")
            return
        probes = self.chart_panel.probes
        if not (0 <= index < len(probes)):
            print(f"No probe at index {index}")
            return
        self.push_undo_state()
        probes[index]["label"] = name
        print(f"Named probe {index} as '{name}'")

    def clear_enabled(self) -> None:
        self.push_undo_state()
        self.state.set_all_kinds(physics.EMPTY)
        self.state.reset_energy_and_flame()
        print("Cleared")

    def fill_enabled(self) -> None:
        self.push_undo_state()
        self.state.set_all_kinds(physics.NORMAL)
        self.state.reset_energy_and_flame()
        print("Filled")

    def zero_activity(self) -> None:
        self.push_undo_state()
        self.state.reset_energy_and_flame()
        print("Settled: flame off, energy full where enabled")

    def unlearn(self) -> None:
        self.push_undo_state()
        self.state.reset_weights()
        print("Unlearned: every weight reset to 1.0, all heat cleared")

    def console_grid(self, size_str: str) -> None:
        try:
            size = int(size_str)
        except ValueError:
            print(f"Invalid size '{size_str}'")
            return
        if not 1 <= size <= 1024:
            print(f"Size {size} out of range 1..1024")
            return
        self.push_undo_state()
        dx, dy = centre_offset(self.state.grid_size, (size, size))
        self.state = resized_state(self.state, (size, size))
        self.cells_panel.set_grid_size(self.state.grid_size)
        self.chart_panel.shift_probes(dx, dy, self.state)
        print(f"Grid is now {size}x{size}")

    def random_strike(self) -> None:
        self.push_undo_state()
        count = int(self.state.grid_size[0] * self.state.grid_size[1] / 10)
        for _ in range(count):
            self.strike_cell((random.randrange(self.state.grid_size[0]), random.randrange(self.state.grid_size[1])))

    def console_shift(self, dx_str: str, dy_str: str) -> None:
        try:
            dx, dy = int(dx_str), int(dy_str)
        except ValueError:
            print(f"Invalid shift '{dx_str} {dy_str}'")
            return
        self.push_undo_state()
        self.state.shift(dx, dy)
        self.chart_panel.shift_probes(dx, dy, self.state)
        print(f"Shifted by ({dx}, {dy})")

    def split_shift_state(self, dx: int, dy: int) -> None:
        hit, cell_xy = self.current_cell()
        if not hit:
            return
        self.push_undo_state()
        self.state.split_shift(dx, dy, cell_xy)
        self.shift_split_probes(dx, dy, cell_xy)

    def shift_split_probes(self, dx: int, dy: int, cursor_xy: tuple[int, int]) -> None:
        cursor_x, cursor_y = cursor_xy
        updated = []
        for probe in self.chart_panel.probes:
            probe = copy.deepcopy(probe)
            probe_x, probe_y = probe["xy"]
            if dx > 0 and probe_x >= cursor_x:
                probe_x += 1
            elif dx < 0 and probe_x <= cursor_x:
                probe_x -= 1
            elif dy > 0 and probe_y >= cursor_y:
                probe_y += 1
            elif dy < 0 and probe_y <= cursor_y:
                probe_y -= 1

            if 0 <= probe_x < self.state.grid_size[0] and 0 <= probe_y < self.state.grid_size[1]:
                probe["xy"] = (probe_x, probe_y)
                updated.append(probe)
        self.chart_panel.set_probes(updated)

    def select_previous_var(self) -> None:
        self.selected_var = (self.selected_var - 1) % len(self.config.vars)
        self.print_var_list()

    def select_next_var(self) -> None:
        self.selected_var = (self.selected_var + 1) % len(self.config.vars)
        self.print_var_list()

    def submit_selected_var_scaled(self, factor: float) -> None:
        name = self.selected_var_name()
        self.submit(f"set {name} {self.config.get(name) * factor:.6g}")

    def selected_var_name(self) -> str:
        return sorted(self.config.vars)[self.selected_var % len(self.config.vars)]

    def print_var_list(self) -> None:
        print()
        for name in sorted(self.config.vars):
            print(f"{name} {self.config.vars[name]}{' <-' if name == self.selected_var_name() else ''}")

    def check_config_safety(self) -> None:
        if self.config.stuck_on_risk():
            print(
                "Warning: SUPPLY/S >= MIN_FLAME * FLAME_CONSUME -- "
                "a cell can burn forever without extinguishing"
            )

    def chart_zoom_in(self) -> None:
        self.chart_panel.set_timescale_s(self.chart_panel.timescale_s / 1.25)

    def chart_zoom_out(self) -> None:
        self.chart_panel.set_timescale_s(self.chart_panel.timescale_s * 1.25)

    def toggle_chart_retrigger(self) -> None:
        self.chart_panel.retrig = not self.chart_panel.retrig

    def toggle_recording(self) -> None:
        if self.recorder.active:
            output = self.recorder.stop()
            print(f"Recording stopped: {output}")
            return

        try:
            output = self.recorder.start(self.screen.get_size())
            print(f"Recording started: {output}")
        except Exception as exc:
            print(f"Recording failed to start: {exc}")

    def list_snapshots(self) -> None:
        names = list_snapshot_names()
        print(" ".join(names))

    def save_snapshot(self, name: str) -> None:
        save_snapshot(name, self.state, self.chart_panel.probes, self.config)
        print(f"Saved {name}.json")

    def try_load_snapshot(self, name: str) -> None:
        try:
            self.load_snapshot(name)
        except Exception as exc:
            print(f"Error loading file '{name}': {exc}")

    def load_snapshot(self, name: str) -> None:
        state, probes, vars_dict = load_snapshot(name)
        self.push_undo_state()
        self.state = state
        self.cells_panel.set_grid_size(self.state.grid_size)
        self.chart_panel.set_probes(probes)
        self.config.reset()
        self.config.update_from_dict(vars_dict)
        print(f"Loaded {name}.json")
        self.check_config_safety()

    def console_set_kind(self, x_str: str, y_str: str, kind: int) -> None:
        try:
            xy = (int(x_str), int(y_str))
        except ValueError:
            print(f"Invalid coordinates '{x_str} {y_str}'")
            return
        if not (0 <= xy[0] < self.state.grid_size[0] and 0 <= xy[1] < self.state.grid_size[1]):
            print(f"Coordinates {xy} out of range")
            return
        self.push_undo_state()
        self.state.set_kind(xy, kind)
        print(f"{xy} is now {KIND_NAMES[kind]}")

    def console_probe(self, x_str: str, y_str: str) -> None:
        try:
            xy = (int(x_str), int(y_str))
        except ValueError:
            print(f"Invalid coordinates '{x_str} {y_str}'")
            return
        if not (0 <= xy[0] < self.state.grid_size[0] and 0 <= xy[1] < self.state.grid_size[1]):
            print(f"Coordinates {xy} out of range")
            return
        self.push_undo_state()
        self.chart_panel.add_probe(xy)
        print(f"Added probe at {xy}")

    def console_deleteprobe(self, x_str: str, y_str: str) -> None:
        try:
            xy = (int(x_str), int(y_str))
        except ValueError:
            print(f"Invalid coordinates '{x_str} {y_str}'")
            return
        self.push_undo_state()
        self.chart_panel.delete_probe(xy)
        print(f"Deleted probe at {xy}")

    def console_strike(self, target: str) -> None:
        probe = probe_by_label(self.chart_panel.probes, target)
        if probe is None:
            print(f"No probe labelled '{target}'")
            return
        self.push_undo_state()
        ok = self.strike_cell(probe["xy"])
        self.chart_panel.trig()
        if ok:
            print(f"Struck '{target}' at {probe['xy']}")
        else:
            print(f"Cannot strike '{target}' at {probe['xy']}: disabled or insufficient energy")

    def console_strike_xy(self, x_str: str, y_str: str) -> None:
        try:
            xy = (int(x_str), int(y_str))
        except ValueError:
            print(f"Invalid coordinates '{x_str} {y_str}'")
            return
        if not (0 <= xy[0] < self.state.grid_size[0] and 0 <= xy[1] < self.state.grid_size[1]):
            print(f"Coordinates {xy} out of range")
            return
        self.push_undo_state()
        ok = self.strike_cell(xy)
        self.chart_panel.trig()
        if ok:
            print(f"Struck {xy}")
        else:
            print(f"Cannot strike {xy}: disabled or insufficient energy")

    def console_set(self, name: str, value_str: str) -> None:
        if name not in self.config.vars:
            print(f"Unknown var '{name}'")
            return
        try:
            value = float(value_str)
        except ValueError:
            print(f"Invalid value '{value_str}'")
            return
        self.push_undo_state()
        self.config.set(name, value)
        self.print_var_list()
        self.check_config_safety()

    def console_run(self, seconds_str: str) -> None:
        try:
            seconds = float(seconds_str)
        except ValueError:
            print(f"Invalid duration '{seconds_str}'")
            return
        dt = 1 / 50.0
        n = max(0, min(round(seconds / dt), 5000))
        self.paused = True
        for _ in range(n):
            self.state.update(dt, self.config)
            self.chart_panel.update(dt, self.state)
        print(f"Ran {n * dt:.2f}s ({n} ticks), paused")

    def console_stats(self) -> None:
        lit, total_flame = grid_summary(self.state)
        energy = self.state.energy_array[self.state.kind_array != physics.EMPTY]
        print(f"lit_cells={lit} total_flame={total_flame:.4f}")
        if energy.size:
            print(f"energy(cells): min={energy.min():.4f} mean={energy.mean():.4f} max={energy.max():.4f}")

    def console_probes(self) -> None:
        for index, probe in enumerate(self.chart_panel.probes):
            xy = probe["xy"]
            label = probe.get("label") or str(index)
            e = float(self.state.energy_array[xy])
            f = float(self.state.flame_array[xy])
            i = float(self.state.illumination_array[xy])
            print(f"{label:12s} {xy}  energy={e:.4f} flame={f:.4f} illumination={i:.4f}")

    def console_inspect(self, x_str: str, y_str: str) -> None:
        try:
            xy = (int(x_str), int(y_str))
        except ValueError:
            print(f"Invalid coordinates '{x_str} {y_str}'")
            return
        if not (0 <= xy[0] < self.state.grid_size[0] and 0 <= xy[1] < self.state.grid_size[1]):
            print(f"Coordinates {xy} out of range")
            return
        s = self.state
        print(f"{xy} kind={KIND_NAMES[int(s.kind_array[xy])]} energy={s.energy_array[xy]:.4f} flame={s.flame_array[xy]:.4f} "
              f"weight={s.weight_array[xy]:.4f} illumination={s.illumination_array[xy]:.4f} trace={s.trace_array[xy]:.3f} "
              f"attention={s.attention_array[xy]:.2f} heat={s.heat_array[xy]:.3f}")

    def print_commands(self) -> None:
        print("Console commands:")
        for command in COMMANDS:
            print(f"{command.usage:20} {command.description}")

    def submit(self, string: str) -> None:
        """Echo a command as if typed at the console, then run it."""
        print(f"> {string}")
        self.execute_console_command(string)

    def submit_at_mouse(self, template: str) -> None:
        hit, (x, y) = self.current_cell()
        if hit:
            self.submit(template.format(x=x, y=y))

    def execute_console_command(self, string: str) -> None:
        words = string.strip().split()
        if not words:
            return
        command = find_command(words)
        if command is None:
            print(f"Unrecognised command '{string}'")
            return
        command.handler(self, words[1:])

    def handle_mouse_button_down(self, event: pygame.event.Event) -> None:
        pos = pygame.mouse.get_pos()
        self.last_mouse_pos = pos

        if self.chart_panel.is_click_within(pos):
            self.set_focus("chart")
        elif self.cells_panel.is_click_within(pos):
            self.set_focus("cells")
        elif self.console_panel.is_click_within(pos):
            self.set_focus("console")

        hit, cell_xy = self.current_cell()
        if not hit:
            return

        if event.button == 1 and pygame.key.get_mods() & pygame.KMOD_CTRL:
            is_teacher = self.state.kind_array[cell_xy] == physics.GOOD
            self.submit(f"{'disable' if is_teacher else 'good'} {cell_xy[0]} {cell_xy[1]}")
            self.dragging_state = None
        elif event.button == 1:
            self.push_undo_state()
            self.set_enabled(cell_xy, not self.get_enabled(cell_xy))
            self.dragging_state = self.get_enabled(cell_xy)
        else:
            self.push_undo_state()
            self.strike_cell(cell_xy)
            self.chart_panel.trig()

    def handle_mouse_motion(self, event: pygame.event.Event) -> None:
        self.last_mouse_pos = pygame.mouse.get_pos()
        if event.buttons[0] and self.dragging_state is not None:
            hit, cell_xy = self.current_cell()
            if hit:
                self.set_enabled(cell_xy, self.dragging_state)

    def handle_event(self, event: pygame.event.Event) -> None:
        self.last_mouse_pos = pygame.mouse.get_pos()

        if event.type == pygame.QUIT:
            self.request_quit()
            return

        # The click that brings the window into focus only focuses it. Depending on the platform, it arrives just before
        # or just after the focus event, so ignore clicks until that event and for a moment after it.
        if event.type == pygame.WINDOWFOCUSGAINED:
            self.window_focused, self.focus_gained_at = True, time.monotonic()
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.window_focused = False
        if event.type == pygame.MOUSEBUTTONDOWN and (not self.window_focused or time.monotonic() - self.focus_gained_at < FOCUS_CLICK_GRACE_S):
            self.dragging_state = None      # and don't paint if the mouse is dragged while still held
            return

        if self.help_visible:
            if event.type == pygame.KEYDOWN:
                self.help_visible = False
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.help_visible = False
                return
            if event.type in (pygame.TEXTINPUT, pygame.MOUSEMOTION):
                return

        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.handle_mouse_button_down(event)
        elif event.type == pygame.MOUSEMOTION:
            self.handle_mouse_motion(event)
        elif event.type == pygame.TEXTINPUT:
            if self.ignore_next_textinput:
                self.ignore_next_textinput = False
            elif self.console_panel.focus:
                self.console_panel.add_input_char(event.text)
        elif event.type == pygame.KEYDOWN:
            if self.console_panel.focus:
                if event.key in [pygame.K_TAB, pygame.K_ESCAPE]:
                    dispatch_keydown(self, event)
                    return
                if event.key in [pygame.K_RETURN, pygame.K_BACKSPACE]:
                    self.console_panel.add_input_char(chr(event.key))
                return
            dispatch_keydown(self, event)

    def update(self, delta_s: float) -> None:
        if self.trainer is not None and self.trainer.state is not self.state:
            self.trainer = None
            print("Training stopped (the grid was replaced)")
        if (not self.paused) or self.do_step:
            for _ in range(self.speed):
                if self.trainer is None:
                    self.state.update(delta_s, self.config)
                    self.chart_panel.update(delta_s, self.state)
                    continue
                summary = self.trainer.step(1 / 50, self.config)
                self.chart_panel.update(1 / 50, self.state)
                if summary is not None:
                    self.report_trial(summary)
        self.do_step = False

    def report_trial(self, summary: dict) -> None:
        w = self.state.weight_array[self.trainer.plastic]
        text = summary.get('text') or (
            f"{summary['input']} -> x {'fired' if summary['x'] else '-'}, y {'fired' if summary['y'] else '-'}, active {summary['active']:.0%}"
            + (f", attended {summary['attended']} cells" if 'attended' in summary else ""))
        print(f"trial {self.trainer.trials}: {text}, weights {w.min():.2f}..{w.max():.2f} (mean {w.mean():.2f})")

    def console_twophase(self, arg: str) -> None:
        import twophase
        if arg == "stop":
            self.trainer = None
            print("Training stopped")
            return
        try:
            seed = int(arg)
        except ValueError:
            print(f"Invalid seed '{arg}'")
            return
        self.start_trainer(lambda: twophase.Demo(seed, self.state.grid_size, self.config))
        w, h = self.state.grid_size
        print(f"Learning on a random {w}x{h} sheet (seed {seed}), local rules only: 200 trials of adaptation (a and b "
              "both rewarded for reaching o) grow routes; then 150 trials teaching a, then 150 teaching b (only the "
              "taught input rewarded). Every 10 teaching trials, a and b are tested with teaching off. Blue shows "
              "attention travelling back; press w for weights (blue) and temperature (green)")

    def console_robot(self, args: list[str]) -> None:
        import robot
        if args[:1] == ["stop"]:
            self.trainer = None
            print("Robot stopped")
            return
        try:
            seed = int(args[0]) if args else 0
            food = args[1] if len(args) > 1 else "red"
            assert food in ("red", "blue")
        except (ValueError, AssertionError):
            print("Usage: robot SEED [red|blue]")
            return
        self.start_trainer(lambda: robot.Demo(seed, food, self.config))
        print(f"Robot (seed {seed}): {food} is food. The sheet (top left) is the robot's brain: sensors on its rim "
              "are struck as they see colour, and each actuator pushes the robot when it fires. Eating strikes the "
              "GOOD or BAD teacher cells. Top right is the world; food is ringed in white")

    def start_trainer(self, make) -> None:
        """Replace the grid with an experiment's own, from a clean slate of the shared parameters."""
        self.push_undo_state()
        self.config.reset()                     # the shared parameters, which `set` then changes live
        self.trainer = make()
        self.state = self.trainer.state
        self.cells_panel.set_grid_size(self.state.grid_size)
        self.chart_panel.set_probes([{"xy": xy, "label": label} for label, xy in self.trainer.at.items()])
        self.chart_panel.trig()

    def console_speed(self, n_str: str) -> None:
        try:
            n = int(n_str)
        except ValueError:
            print(f"Invalid speed '{n_str}'")
            return
        self.speed = max(1, min(n, 50))
        print(f"Speed {self.speed}x")

    def render(self, delta_s: float) -> None:
        self.screen.fill((0, 0, 0))
        self.console_panel.render(delta_s, self.paused)
        self.chart_panel.render(self.config)
        self.world_panel.render(getattr(self.trainer, "robot", None))
        temperature = getattr(self.trainer, "temperature", None)
        self.cells_panel.render(self.state, self.config, self.show_only, self.chart_panel.probes,
                                temperature, getattr(self.trainer, "max_temperature", 1.0))
        if self.help_visible:
            self.help_overlay.render(self.help_lines)
        self.recorder.add_frame(self.screen, delta_s)
        pygame.display.flip()

    def poll_control_input(self) -> None:
        try:
            content = CONTROL_IN_PATH.read_text()
        except OSError:
            return
        if not content:
            return
        CONTROL_IN_PATH.write_text("")
        for line in content.splitlines():
            line = line.strip()
            if line:
                print(f"[control] {line}")
                self.execute_console_command(line)

    def run(self, commands: list[str] = ()) -> None:
        """Run the simulator; commands are submitted first, as if typed at the console."""
        for command in commands:
            self.submit(command)
        this_frame_start = time.time()
        try:
            while self.running:
                last_frame_start = this_frame_start
                this_frame_start = time.time()
                delta = this_frame_start - last_frame_start
                if delta > 1 / 50.0:
                    delta = 1 / 50.0

                for event in pygame.event.get():
                    self.handle_event(event)

                self.poll_control_input()
                self.update(delta)
                self.render(delta)
        finally:
            if self.recorder.active:
                self.recorder.stop()
            try:
                save_snapshot("recent", self.state, self.chart_panel.probes, self.config)
            finally:
                sys.stdout = self.original_stdout
                self.control_log.close()
                pygame.quit()
