from __future__ import annotations

import sys

import numpy as np
import pygame

import physics
from model import SimulationConfig, State


ENERGY_COLOUR = (255, 0, 0)
FLAME_COLOUR = (0, 255, 0)
ILLUMINATION_COLOUR = (255, 255, 255)
FLAME_DISPLAY_GAIN = 2.0   # typical peak flame (~0.43) shows near full green
CELL_FLOOR = 0.1           # a cell never looks darker than 10% grey, so burnt-out cells stay visible
TEACHER_TINT = 0.4         # teacher cells always carry 40% blue


class CellsPanel:
    def __init__(self, screen: pygame.Surface, screen_xy: tuple[int, int], screen_size: tuple[int, int], grid_size: tuple[int, int]):
        self.screen = screen
        self.xy = screen_xy
        self.size = screen_size
        self.focus = False
        self.probe_font = pygame.font.Font(pygame.font.match_font("couriernew"), 16)
        self.set_grid_size(grid_size)

    def set_grid_size(self, grid_size: tuple[int, int]) -> None:
        self.grid_size = grid_size
        self.pixel_scale = self.size[0] / self.grid_size[0]
        self.surface = pygame.Surface(self.grid_size)

    def render(self, state: State, config: SimulationConfig, show_only: str, probes: list[dict],
               temperature: np.ndarray | None = None, max_temperature: float = 1.0) -> None:
        """Red = energy, green = flame, blue = attended (attention passing back through it). 'E' and 'I' show energy and illumination
        in grey. 'W' shows weight in blue (WEIGHT_MAX = full) and, when a training experiment provides one, each
        cell's temperature in green (max_temperature = full)."""
        if show_only == "W":
            present = state.kind_array != 0
            rgb = np.zeros(state.grid_size + (3,))
            rgb[:, :, 2] = np.where(present, state.weight_array / config.get("WEIGHT_MAX"), 0.0)   # neutral 1.0 = half blue
            if temperature is not None:
                rgb[:, :, 1] = np.where(present, temperature / max_temperature, 0.0)
        elif show_only == "E":
            grey = state.energy_array
            rgb = np.repeat(grey[:, :, None], 3, axis=2)
            rgb[state.kind_array != 0] = np.maximum(rgb[state.kind_array != 0], CELL_FLOOR)
        elif show_only == "I":
            grey = state.illumination_array / (2 * config.get("MIN_STRIKE"))   # ignition threshold = mid-grey
            rgb = np.repeat(grey[:, :, None], 3, axis=2)
        else:
            rgb = np.zeros(state.grid_size + (3,))
            rgb[:, :, 0] = state.energy_array
            rgb[:, :, 1] = state.flame_array * FLAME_DISPLAY_GAIN
            rgb[:, :, 2] = state.attention_array > config.get("ATTENTION_REST")          # attended: passing attention on
            rgb[:, :, 2] += np.where(np.isin(state.kind_array, physics.TEACHERS), TEACHER_TINT, 0.0)
            rgb[state.kind_array != 0] = np.maximum(rgb[state.kind_array != 0], CELL_FLOOR)
        pygame.surfarray.blit_array(self.surface, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))

        pixels = pygame.transform.scale(self.surface, self.size)
        self.screen.blit(pixels, self.xy)

        if self.pixel_scale > 6:
            for x_value in range(self.grid_size[0] + 1):
                pygame.draw.line(
                    self.screen,
                    (32, 32, 32),
                    (self.xy[0] + x_value * self.pixel_scale, self.xy[1]),
                    (self.xy[0] + x_value * self.pixel_scale, self.xy[1] + self.size[1]),
                    1,
                )
            for y_value in range(self.grid_size[1] + 1):
                pygame.draw.line(
                    self.screen,
                    (32, 32, 32),
                    (self.xy[0], self.xy[1] + y_value * self.pixel_scale),
                    (self.xy[0] + self.size[0], self.xy[1] + y_value * self.pixel_scale),
                    1,
                )

        for index, probe in enumerate(probes):
            label = probe.get("label") or str(index)
            textpixels = self.probe_font.render(label, True, (255, 255, 255))
            self.screen.blit(
                textpixels,
                (
                    self.xy[0] + probe["xy"][0] * self.pixel_scale,
                    self.xy[1] + probe["xy"][1] * self.pixel_scale,
                ),
            )

        if self.focus and pygame.key.get_focused():     # only while the window has keyboard focus
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (self.xy[0], self.xy[1], self.size[0], self.size[1]),
                width=1,
            )

    def find_cell(self, screen_xy: tuple[int, int]) -> tuple[bool, int, int]:
        cell_x = int(screen_xy[0] / self.pixel_scale)
        cell_y = int(screen_xy[1] / self.pixel_scale)
        hit = (0 <= cell_x < self.grid_size[0]) and (0 <= cell_y < self.grid_size[1])
        return hit, cell_x, cell_y

    def is_click_within(self, xy: tuple[int, int]) -> bool:
        return (
            self.xy[0] <= xy[0] < self.xy[0] + self.size[0]
            and self.xy[1] <= xy[1] < self.xy[1] + self.size[1]
        )


class ChartPanel:
    def __init__(self, screen: pygame.Surface, xy: tuple[int, int], size: tuple[int, int]):
        self.screen = screen
        self.xy = xy
        self.size = size
        self.ybot = self.xy[1] + self.size[1]
        self.set_timescale_s(1.0)
        self.retrig = False
        self.time_since_trig: float | None = None
        self.focus = False
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16)
        self.probes: list[dict] = []

    def set_timescale_s(self, timescale_s: float) -> None:
        self.timescale_s = timescale_s
        self.scale = (self.size[0] / self.timescale_s, self.size[1] / 1.0)

    def render(self, config: SimulationConfig) -> None:
        pygame.draw.rect(self.screen, (0, 0, 32), (self.xy[0], self.xy[1], self.size[0], self.size[1]), width=0)
        self.screen.blit(
            self.font.render(f"{self.timescale_s:3.1f}s", True, (255, 255, 255)),
            (self.xy[0] + self.size[0] - 40, self.xy[1]),
        )
        self.screen.blit(self.font.render("ENERGY", True, ENERGY_COLOUR), (self.xy[0] + self.size[0] - 140, self.xy[1]))
        self.screen.blit(self.font.render("FLAME", True, FLAME_COLOUR), (self.xy[0] + self.size[0] - 140, self.xy[1] + 20))
        self.screen.blit(
            self.font.render("ILLUMINATION", True, ILLUMINATION_COLOUR),
            (self.xy[0] + self.size[0] - 140, self.xy[1] + 40),
        )

        min_strike_y = self.ybot - config.get("MIN_STRIKE") * self.scale[1]
        pygame.draw.line(self.screen, (64, 64, 64), (self.xy[0], min_strike_y), (self.xy[0] + self.size[0], min_strike_y))
        self.screen.blit(self.font.render("MIN_STRIKE", True, (64, 64, 64)), (self.xy[0], min_strike_y))

        min_flame_y = self.ybot - config.get("MIN_FLAME") * self.scale[1]
        self.screen.blit(self.font.render("MIN_FLAME", True, (64, 64, 64)), (self.xy[0], min_flame_y))
        pygame.draw.line(self.screen, (64, 64, 64), (self.xy[0], min_flame_y), (self.xy[0] + self.size[0], min_flame_y))

        for probe in self.probes:
            if len(probe["energy_chart"]) > 1:
                pygame.draw.lines(self.screen, ILLUMINATION_COLOUR, False, probe["illumination_chart"], 1)
                pygame.draw.lines(self.screen, ENERGY_COLOUR, False, probe["energy_chart"], 1)
                pygame.draw.lines(self.screen, FLAME_COLOUR, False, probe["flame_chart"], 1)

        if self.focus and pygame.key.get_focused():     # only while the window has keyboard focus
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (self.xy[0] + 1, self.xy[1] + 1, self.size[0] - 2, self.size[1] - 2),
                width=1,
            )

    def update(self, delta_s: float, state: State) -> None:
        if self.time_since_trig is None:
            return

        self.time_since_trig += delta_s
        x_value = self.xy[0] + self.time_since_trig * self.scale[0]
        if self.time_since_trig < self.timescale_s:
            for probe in self.probes:
                probe_xy = probe["xy"]
                probe["energy_chart"].append((x_value, self.ybot - state.energy_array[probe_xy] * self.scale[1]))
                probe["flame_chart"].append((x_value, self.ybot - state.flame_array[probe_xy] * self.scale[1]))
                probe["illumination_chart"].append((x_value, self.ybot - state.illumination_array[probe_xy] * self.scale[1]))
        elif self.retrig:
            self.trig()

    def trig(self) -> None:
        for probe in self.probes:
            probe["energy_chart"] = []
            probe["flame_chart"] = []
            probe["illumination_chart"] = []
        self.time_since_trig = 0

    def add_probe(self, cell_xy: tuple[int, int]) -> None:
        for probe in self.probes:
            if probe["xy"] == cell_xy:
                return
        self.probes.append(self.new_probe(cell_xy))

    @staticmethod
    def new_probe(xy: tuple[int, int], label: str | None = None) -> dict:
        probe = {"xy": tuple(xy), "energy_chart": [], "flame_chart": [], "illumination_chart": []}
        if label:
            probe["label"] = label
        return probe

    def find_probe_index(self, cell_xy: tuple[int, int]) -> int | None:
        for index, probe in enumerate(self.probes):
            if probe["xy"] == cell_xy:
                return index
        return None

    def delete_probe(self, cell_xy: tuple[int, int]) -> None:
        self.probes = [probe for probe in self.probes if probe["xy"] != cell_xy]

    def shift_probes(self, dx: int, dy: int, state: State) -> None:
        shifted = []
        for probe in self.probes:
            probe_xy = (probe["xy"][0] + dx, probe["xy"][1] + dy)
            if 0 <= probe_xy[0] < state.grid_size[0] and 0 <= probe_xy[1] < state.grid_size[1]:
                probe["xy"] = probe_xy
                shifted.append(probe)
        self.probes = shifted

    def set_probes(self, probes: list[dict]) -> None:
        self.probes = [self.new_probe(probe["xy"], probe.get("label")) for probe in probes]

    def is_click_within(self, xy: tuple[int, int]) -> bool:
        return (
            self.xy[0] <= xy[0] < self.xy[0] + self.size[0]
            and self.xy[1] <= xy[1] < self.xy[1] + self.size[1]
        )


class ConsolePanel:
    def __init__(self, screen: pygame.Surface, xy: tuple[int, int], size: tuple[int, int], execute):
        self.screen = screen
        self.xy = xy
        self.size = size
        self.char_width_pixels = 10
        self.char_height_pixels = 16
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16)
        self.chars_wide = int(size[0] / self.char_width_pixels)
        self.chars_high = int(size[1] / self.char_height_pixels)
        self.input = ""
        self.strings = ["" for _ in range(self.chars_high - 1)]
        self.fps_smoothing = 0.0
        self.focus = False
        self.execute = execute

    def scroll(self) -> None:
        self.strings = self.strings[1:] + [""]

    def add_console_char(self, char: str) -> None:
        if char == chr(10):
            self.scroll()
        else:
            if len(self.strings[-1]) >= self.chars_wide:
                self.scroll()
            self.strings[-1] += char

    def add_input_char(self, char: str) -> None:
        if char == chr(8):
            self.input = self.input[0:-1]
        elif char == chr(13):
            command, self.input = self.input, ""
            self.execute(command)
        else:
            self.input += char

    def write(self, string: str) -> None:
        sys.stderr.write(string)
        for char in string:
            self.add_console_char(char)

    def flush(self) -> None:
        pass

    def render(self, s_per_frame: float, is_paused: bool) -> None:
        render_strings = self.strings + [">" + self.input]
        for index, string in enumerate(render_strings):
            self.screen.blit(
                self.font.render(string, True, (255, 255, 255)),
                (self.xy[0], self.xy[1] + index * self.char_height_pixels),
            )

        self.fps_smoothing = self.fps_smoothing * 0.9 + s_per_frame * 0.1
        if is_paused:
            label = "PAUSED"
            colour = (255, 255, 255)
        else:
            label = f"{int(1 / self.fps_smoothing)}fps" if self.fps_smoothing else "..."
            colour = (64, 64, 0)
        self.screen.blit(
            self.font.render(label, True, colour),
            (self.xy[0] + self.size[0] - 70, self.xy[1]),
        )

        if self.focus and pygame.key.get_focused():     # only while the window has keyboard focus
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (self.xy[0] + 1, self.xy[1] + 1, self.size[0] - 2, self.size[1] - 2),
                width=1,
            )

    def is_click_within(self, xy: tuple[int, int]) -> bool:
        return (
            self.xy[0] <= xy[0] < self.xy[0] + self.size[0]
            and self.xy[1] <= xy[1] < self.xy[1] + self.size[1]
        )


class WorldPanel:
    """The robot's world, when one is running: the arena, its red and blue blocks (food ringed in white), and the
    robot, with a tick for each sensor and a dot for each actuator that lights up as it fires."""
    FLASH_S = 0.3

    def __init__(self, screen: pygame.Surface, xy: tuple[int, int], size: tuple[int, int]):
        self.screen = screen
        self.xy = xy
        self.size = size
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16)

    def render(self, robot) -> None:
        pygame.draw.rect(self.screen, (16, 16, 16), (*self.xy, *self.size))
        if robot is None:
            self.screen.blit(self.font.render("no world (try: robot 0)", True, (96, 96, 96)), (self.xy[0] + 10, self.xy[1] + 10))
            return
        w = robot.world
        side = min(self.size) - 20
        scale = side / w.size
        origin = (self.xy[0] + self.size[0] - side - 10, self.xy[1] + 10)
        at = lambda p: (origin[0] + p[0] * scale, origin[1] + p[1] * scale)
        pygame.draw.rect(self.screen, (64, 64, 64), (*origin, side, side), width=1)
        for xy, colour in robot.blocks:
            rgb = (220, 40, 40) if colour == 'red' else (40, 80, 255)
            pygame.draw.circle(self.screen, rgb, at(xy), max(3, scale * 0.6))
            if colour == robot.food:
                pygame.draw.circle(self.screen, (255, 255, 255), at(xy), max(4, scale * 0.6) + 2, width=1)
        centre = at(robot.xy)
        radius = w.radius * scale
        pygame.draw.circle(self.screen, (128, 128, 128), centre, radius, width=2)
        for sensor in robot.sensor_cells:
            c, s = np.cos(sensor['angle']), np.sin(sensor['angle'])
            pygame.draw.line(self.screen, (160, 160, 160), (centre[0] + c * radius, centre[1] + s * radius),
                             (centre[0] + c * (radius + 6), centre[1] + s * (radius + 6)), 2)
        for i, actuator in enumerate(robot.actuator_cells):
            c, s = np.cos(actuator['angle']), np.sin(actuator['angle'])
            lit = robot.t - robot.fired[i] < self.FLASH_S and robot.fired[i] > 0
            pygame.draw.circle(self.screen, (255, 255, 0) if lit else (64, 64, 0),
                               (centre[0] + c * radius * 0.7, centre[1] + s * radius * 0.7), 4 if lit else 2)
        food, poison = robot.counts()
        recent_food, recent_poison = robot.counts(robot.t - 300)
        lines = (f"t {robot.t:.0f}s, {robot.food} is food",
                 f"ate {food} food, {poison} poison",
                 f"last 5 min {recent_food} food, {recent_poison} poison")
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, (255, 255, 255)), (self.xy[0] + 6, self.xy[1] + 6 + 18 * i))


class HelpOverlay:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.title_font = pygame.font.Font(pygame.font.match_font("couriernew"), 24)
        self.body_font = pygame.font.Font(pygame.font.match_font("couriernew"), 18)

    def render(self, lines: list[str]) -> None:
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 210))
        self.screen.blit(overlay, (0, 0))

        width, height = self.screen.get_size()
        panel_rect = pygame.Rect(40, 40, width - 80, height - 80)
        pygame.draw.rect(self.screen, (20, 20, 20), panel_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), panel_rect, width=1)
        self.screen.blit(self.title_font.render("Help", True, (255, 255, 255)), (panel_rect.x + 20, panel_rect.y + 20))

        y_value = panel_rect.y + 60
        for line in lines:
            font = self.title_font if line.endswith(":") else self.body_font
            self.screen.blit(font.render(line, True, (255, 255, 255)), (panel_rect.x + 20, y_value))
            y_value += 26 if line.endswith(":") else 22
