from __future__ import annotations

import copy
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

from actions import FOCUS_ORDER, build_help_lines, dispatch_keydown
from headless import grid_summary, probe_by_label
from model import SimulationConfig, State, resized_state
from persistence import list_snapshot_names, load_snapshot, save_snapshot
from recording import VideoRecorder
from views import CellsPanel, ChartPanel, ConsolePanel, HelpOverlay


CONTROL_IN_PATH = Path("control_in.txt")
CONTROL_OUT_PATH = Path("control_out.log")


@dataclass
class UndoSnapshot:
    state: State
    probes: list[dict]
    config_vars: dict[str, float]
    config_selected: int
    show_only: str


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

        self.cells_panel = CellsPanel(self.screen, (0, 0), (self.screen_height, self.screen_height), self.state.grid_size)
        self.chart_panel = ChartPanel(self.screen, (self.screen_height, 0), (self.screen_width - self.screen_height, int(self.screen_height / 2)))
        self.console_panel = ConsolePanel(
            self.screen,
            (self.screen_height, int(self.screen_height / 2)),
            (self.screen_width - self.screen_height, self.screen_height - int(self.screen_height / 2)),
            self.execute_console_command,
        )
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

    def toggle_pause(self) -> None:
        self.paused = not self.paused

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
            config_selected=self.config.selected,
            show_only=self.show_only,
        )

    def push_undo_state(self) -> None:
        self.undo_stack.append(self.make_undo_snapshot())

    def restore_undo_snapshot(self, snapshot: UndoSnapshot) -> None:
        self.state = snapshot.state
        self.cells_panel.set_grid_size(self.state.grid_size)
        self.chart_panel.set_probes(copy.deepcopy(snapshot.probes))
        self.config.vars = copy.deepcopy(snapshot.config_vars)
        self.config.selected = snapshot.config_selected
        self.show_only = snapshot.show_only

    def undo(self) -> None:
        if not self.undo_stack:
            print("Nothing to undo")
            return
        snapshot = self.undo_stack.pop()
        self.restore_undo_snapshot(snapshot)

    def current_cell(self) -> tuple[bool, tuple[int, int]]:
        hit, grid_x, grid_y = self.cells_panel.find_cell(self.last_mouse_pos)
        return hit, (grid_x, grid_y)

    def set_enabled(self, xy: tuple[int, int], state: bool) -> None:
        self.state.enabled_array[xy] = state
        self.state.energy_array[xy] = 0
        self.state.flame_array[xy] = 0

    def get_enabled(self, xy: tuple[int, int]) -> bool:
        return bool(self.state.enabled_array[xy])

    def strike_cell(self, cell_xy: tuple[int, int]) -> None:
        if self.state.enabled_array[cell_xy] and self.state.energy_array[cell_xy] >= self.config.get("MIN_STRIKE"):
            self.state.flame_array[cell_xy] = self.config.get("STRIKE_LEVEL")

    def add_probe_under_mouse(self) -> None:
        hit, cell_xy = self.current_cell()
        if hit:
            self.push_undo_state()
            self.chart_panel.add_probe(cell_xy)

    def delete_probe_under_mouse(self) -> None:
        hit, cell_xy = self.current_cell()
        if hit:
            self.push_undo_state()
            self.chart_panel.delete_probe(cell_xy)

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
        self.state.set_all_enableds(False)
        self.state.reset_energy_and_flame()

    def fill_enabled(self) -> None:
        self.push_undo_state()
        self.state.set_all_enableds(True)
        self.state.reset_energy_and_flame()

    def zero_activity(self) -> None:
        self.push_undo_state()
        self.state.reset_energy_and_flame()

    def increase_grid_size(self) -> None:
        if self.state.grid_size[0] < 1024:
            self.resize_grid((self.state.grid_size[0] * 2, self.state.grid_size[1] * 2))

    def decrease_grid_size(self) -> None:
        if self.state.grid_size[0] > 1:
            self.resize_grid((int(self.state.grid_size[0] / 2), int(self.state.grid_size[1] / 2)))

    def resize_grid(self, grid_size: tuple[int, int]) -> None:
        self.push_undo_state()
        self.state = resized_state(self.state, grid_size)
        self.cells_panel.set_grid_size(grid_size)
        self.chart_panel.shift_probes(0, 0, self.state)

    def random_strike(self) -> None:
        self.push_undo_state()
        count = int(self.state.grid_size[0] * self.state.grid_size[1] / 10)
        for _ in range(count):
            self.strike_cell((random.randrange(self.state.grid_size[0]), random.randrange(self.state.grid_size[1])))

    def shift_state(self, dx: int, dy: int) -> None:
        self.push_undo_state()
        self.state.shift(dx, dy)
        self.chart_panel.shift_probes(dx, dy, self.state)

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
        self.config.select_previous()
        self.print_var_list()

    def select_next_var(self) -> None:
        self.config.select_next()
        self.print_var_list()

    def scale_selected_var(self, factor: float) -> None:
        self.push_undo_state()
        self.config.scale_selected(factor)
        self.print_var_list()
        self.check_config_safety()

    def print_var_list(self) -> None:
        for line in self.config.describe_lines():
            print(line)

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
        self.config.update_from_dict(vars_dict)
        print(f"Loaded {name}.json")
        self.check_config_safety()

    def console_strike(self, target: str) -> None:
        probe = probe_by_label(self.chart_panel.probes, target)
        if probe is None:
            print(f"No probe labelled '{target}'")
            return
        self.push_undo_state()
        self.strike_cell(probe["xy"])
        self.chart_panel.trig()
        print(f"Struck '{target}' at {probe['xy']}")

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
        self.strike_cell(xy)
        self.chart_panel.trig()
        print(f"Struck {xy}")

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

    def console_step(self, n_str: str) -> None:
        try:
            n = int(n_str)
        except ValueError:
            print(f"Invalid step count '{n_str}'")
            return
        n = max(0, min(n, 5000))
        for _ in range(n):
            self.state.update(1 / 50.0, self.config)
            self.chart_panel.update(1 / 50.0, self.state)
        print(f"Stepped {n} ticks")

    def console_stats(self) -> None:
        lit, total_flame = grid_summary(self.state)
        enabled = self.state.enabled_array
        energy = self.state.energy_array[enabled]
        print(f"lit_cells={lit} total_flame={total_flame:.4f}")
        if energy.size:
            print(f"energy(enabled): min={energy.min():.4f} mean={energy.mean():.4f} max={energy.max():.4f}")

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
        enabled = bool(self.state.enabled_array[xy])
        e = float(self.state.energy_array[xy])
        f = float(self.state.flame_array[xy])
        i = float(self.state.illumination_array[xy])
        print(f"{xy} enabled={enabled} energy={e:.4f} flame={f:.4f} illumination={i:.4f}")

    def execute_console_command(self, string: str) -> None:
        words = string.strip().split()
        if not words:
            return
        if words[0] == "ls":
            self.list_snapshots()
        elif words[0] in {"help", "?"}:
            print("Console commands:")
            print("ls")
            print("save NAME")
            print("load NAME")
            print("name INDEX NAME")
            print("strike X Y | strike LABEL")
            print("set VAR VALUE")
            print("step N")
            print("stats")
            print("probes")
            print("inspect X Y")
            print("vars")
            print("quit")
        elif words[0] == "save" and len(words) >= 2:
            self.save_snapshot(words[1])
        elif words[0] == "load" and len(words) >= 2:
            self.try_load_snapshot(words[1])
        elif words[0] == "name" and len(words) >= 3:
            self.name_probe(words[1], " ".join(words[2:]))
        elif words[0] == "strike" and len(words) == 3:
            self.console_strike_xy(words[1], words[2])
        elif words[0] == "strike" and len(words) == 2:
            self.console_strike(words[1])
        elif words[0] == "set" and len(words) == 3:
            self.console_set(words[1], words[2])
        elif words[0] == "step" and len(words) == 2:
            self.console_step(words[1])
        elif words[0] == "stats" and len(words) == 1:
            self.console_stats()
        elif words[0] == "probes" and len(words) == 1:
            self.console_probes()
        elif words[0] == "inspect" and len(words) == 3:
            self.console_inspect(words[1], words[2])
        elif words[0] == "vars" and len(words) == 1:
            self.print_var_list()
        elif words[0] == "quit" and len(words) == 1:
            self.request_quit()
        else:
            print(f"Unrecognised command '{string}'")

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

        if event.button == 1:
            self.push_undo_state()
            self.set_enabled(cell_xy, not self.get_enabled(cell_xy))
            self.dragging_state = self.get_enabled(cell_xy)
        else:
            self.push_undo_state()
            self.strike_cell(cell_xy)
            self.chart_panel.trig()

    def handle_mouse_motion(self, event: pygame.event.Event) -> None:
        self.last_mouse_pos = pygame.mouse.get_pos()
        if event.buttons[0]:
            hit, cell_xy = self.current_cell()
            if hit:
                self.set_enabled(cell_xy, self.dragging_state)

    def handle_event(self, event: pygame.event.Event) -> None:
        self.last_mouse_pos = pygame.mouse.get_pos()

        if event.type == pygame.QUIT:
            self.request_quit()
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
        if (not self.paused) or self.do_step:
            self.state.update(delta_s, self.config)
            self.chart_panel.update(delta_s, self.state)
        self.do_step = False

    def render(self, delta_s: float) -> None:
        self.screen.fill((0, 0, 0))
        self.console_panel.render(delta_s, self.paused)
        self.chart_panel.render(self.config)
        self.cells_panel.render(self.state, self.show_only, self.chart_panel.probes)
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

    def run(self) -> None:
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
