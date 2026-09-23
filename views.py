from __future__ import annotations

import sys

import numpy as np
import pygame

from model import SimulationConfig, State


ENERGY_COLOUR = (0, 255, 0)
FLAME_COLOUR = (0, 0, 255)
ILLUMINATION_COLOUR = (255, 255, 255)


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

    def render(self, state: State, show_only: str, probes: list[dict]) -> None:
        white_int = 1 + 256 + 256 * 256
        if show_only:
            arrays = {
                "E": state.energy_array,
                "F": state.flame_array,
                "I": state.illumination_array,
            }
            arr = arrays[show_only]
            values = np.clip((arr * 256).astype(int), 0, 255)
            pygame.surfarray.blit_array(self.surface, values * white_int)
        else:
            energy = np.clip((state.energy_array * 256).astype(int), 0, 255)
            flame = np.clip((state.flame_array * 256).astype(int), 0, 255)
            pygame.surfarray.blit_array(
                self.surface,
                energy + 255 * 256 * flame + 255 * 256 * 256 * state.enabled_array,
            )

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

        if self.focus:
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

        if self.focus:
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
        self.probes.append(
            {
                "xy": cell_xy,
                "energy_chart": [],
                "flame_chart": [],
                "illumination_chart": [],
            }
        )

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
        self.probes = probes
        for probe in self.probes:
            probe["xy"] = tuple(probe["xy"])

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
            self.add_console_char(">")
            for existing_char in self.input:
                self.add_console_char(existing_char)
            self.scroll()
            self.execute(self.input)
            self.input = ""
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

        if self.focus:
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
