from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage


DEFAULT_VARS = {
    "COUPLING_DIST": 0.85,
    "COUPLING_GAIN": 6.0,
    "FLAME_CONSUME": 2.0,
    "FLAME_INERTIA": 1.0,
    "MIN_FLAME": 0.09,
    "MIN_STRIKE": 0.23,
    "STRIKE_LEVEL": 0.18,
    "SUPPLY/S": 0.14,
}


@dataclass
class SimulationConfig:
    vars: dict[str, float] = field(default_factory=lambda: copy.deepcopy(DEFAULT_VARS))
    selected: int = 0

    def sorted_keys(self) -> list[str]:
        return sorted(self.vars)

    def selected_key(self) -> str:
        keys = self.sorted_keys()
        return keys[self.selected % len(keys)]

    def get(self, name: str) -> float:
        return self.vars[name]

    def set(self, name: str, value: float) -> None:
        self.vars[name] = float(value)

    def get_selected(self) -> float:
        return self.vars[self.selected_key()]

    def set_selected(self, value: float) -> None:
        self.vars[self.selected_key()] = float(value)

    def select_previous(self) -> None:
        self.selected = (self.selected - 1) % len(self.vars)

    def select_next(self) -> None:
        self.selected = (self.selected + 1) % len(self.vars)

    def scale_selected(self, factor: float) -> None:
        self.set_selected(self.get_selected() * factor)

    def update_from_dict(self, values: dict[str, float]) -> None:
        for key, value in values.items():
            self.vars[key] = float(value)

    def describe_lines(self) -> list[str]:
        lines = [""]
        selected_key = self.selected_key()
        for key in self.sorted_keys():
            marker = " <-" if key == selected_key else ""
            lines.append(f"{key} {self.vars[key]}{marker}")
        return lines

    def stuck_on_risk(self) -> bool:
        """True if a burning cell can find a stable, non-extinguishing equilibrium.

        The relaxation step lets flame settle wherever consumption balances
        resupply: flame_eq = (SUPPLY/S) / FLAME_CONSUME. If that equilibrium
        sits at or above MIN_FLAME, a cell that finds it can burn forever
        without ever crossing the extinguish threshold.
        """
        return self.get("SUPPLY/S") >= self.get("MIN_FLAME") * self.get("FLAME_CONSUME")


class State:
    def __init__(self, grid_size: tuple[int, int]):
        self.grid_size = grid_size
        self.enabled_array = np.zeros(self.grid_size, dtype=bool)
        self.energy_array = np.zeros(self.grid_size, dtype=float)
        self.flame_array = np.zeros(self.grid_size, dtype=float)
        self.illumination_array = np.zeros(self.grid_size, dtype=float)

    def set_all_enableds(self, state: bool) -> None:
        self.enabled_array[:] = state

    def reset_energy_and_flame(self) -> None:
        """Reset to a clean baseline: flame off, energy full wherever enabled.

        Energy is filled rather than emptied so no residual "recently fired"
        trace (e.g. from a loaded snapshot's history) leaks into whatever
        reads energy as a proxy for recent activity.
        """
        self.energy_array[:] = 0
        self.energy_array[self.enabled_array] = 1.0
        self.flame_array[:] = 0
        self.illumination_array[:] = 0

    def update(self, delta_s: float, config: SimulationConfig) -> None:
        self.illumination_array = (
            config.get("COUPLING_GAIN")
            * scipy.ndimage.gaussian_filter(
                self.flame_array,
                sigma=config.get("COUPLING_DIST"),
                mode="constant",
            )
        )

        self.flame_array[
            (self.enabled_array == True)
            & (self.flame_array == 0)
            & (self.illumination_array >= config.get("MIN_STRIKE"))
        ] = config.get("STRIKE_LEVEL")

        self.flame_array[
            (self.flame_array < config.get("MIN_FLAME")) | (self.energy_array == 0)
        ] = 0

        flame_inertia = config.get("FLAME_INERTIA")
        self.flame_array = np.where(
            self.flame_array > 0,
            self.energy_array
            + (self.flame_array - self.energy_array)
            * math.exp(-1.0 / flame_inertia * delta_s),
            self.flame_array,
        )

        self.energy_array = np.maximum(
            0,
            self.energy_array - self.flame_array * config.get("FLAME_CONSUME") * delta_s,
        )

        self.energy_array[self.enabled_array] = np.minimum(
            1,
            self.energy_array[self.enabled_array] + config.get("SUPPLY/S") * delta_s,
        )

    def shift(self, dx: int, dy: int) -> None:
        def scroll(arr: np.ndarray) -> np.ndarray:
            arr = np.roll(arr, shift=dx, axis=0)
            arr = np.roll(arr, shift=dy, axis=1)
            if dx > 0:
                arr[0:dx, :] = 0
            if dx < 0:
                arr[dx:, :] = 0
            if dy > 0:
                arr[:, 0:dy] = 0
            if dy < 0:
                arr[:, dy:] = 0
            return arr

        self.enabled_array = scroll(self.enabled_array)
        self.energy_array = scroll(self.energy_array)
        self.flame_array = scroll(self.flame_array)
        self.illumination_array = scroll(self.illumination_array)

    def split_shift(self, dx: int, dy: int, cursor_xy: tuple[int, int]) -> None:
        cursor_x, cursor_y = cursor_xy

        def shift_x(arr: np.ndarray) -> np.ndarray:
            new_arr = arr.copy()
            if dx > 0:
                if cursor_x + 1 < arr.shape[0]:
                    new_arr[cursor_x + 1 :, :] = arr[cursor_x:-1, :]
                new_arr[cursor_x, :] = 0
            elif dx < 0:
                if cursor_x > 0:
                    new_arr[:cursor_x, :] = arr[1 : cursor_x + 1, :]
                new_arr[cursor_x, :] = 0
            return new_arr

        def shift_y(arr: np.ndarray) -> np.ndarray:
            new_arr = arr.copy()
            if dy > 0:
                if cursor_y + 1 < arr.shape[1]:
                    new_arr[:, cursor_y + 1 :] = arr[:, cursor_y:-1]
                new_arr[:, cursor_y] = 0
            elif dy < 0:
                if cursor_y > 0:
                    new_arr[:, :cursor_y] = arr[:, 1 : cursor_y + 1]
                new_arr[:, cursor_y] = 0
            return new_arr

        if dx != 0:
            self.enabled_array = shift_x(self.enabled_array)
            self.energy_array = shift_x(self.energy_array)
            self.flame_array = shift_x(self.flame_array)
            self.illumination_array = shift_x(self.illumination_array)

        if dy != 0:
            self.enabled_array = shift_y(self.enabled_array)
            self.energy_array = shift_y(self.energy_array)
            self.flame_array = shift_y(self.flame_array)
            self.illumination_array = shift_y(self.illumination_array)

    def paste(self, source_state: "State") -> None:
        def paste_array(foreground: np.ndarray, background: np.ndarray) -> None:
            bg_h, bg_w = background.shape
            fg_h, fg_w = foreground.shape

            start_y = max(0, (bg_h - fg_h) // 2)
            start_x = max(0, (bg_w - fg_w) // 2)
            end_y = min(bg_h, start_y + fg_h)
            end_x = min(bg_w, start_x + fg_w)

            fg_start_y = max(0, -((bg_h - fg_h) // 2))
            fg_start_x = max(0, -((bg_w - fg_w) // 2))
            fg_end_y = fg_start_y + (end_y - start_y)
            fg_end_x = fg_start_x + (end_x - start_x)

            background[start_y:end_y, start_x:end_x] = foreground[
                fg_start_y:fg_end_y, fg_start_x:fg_end_x
            ]

        paste_array(source_state.enabled_array, self.enabled_array)
        paste_array(source_state.energy_array, self.energy_array)
        paste_array(source_state.flame_array, self.flame_array)
        paste_array(source_state.illumination_array, self.illumination_array)

    def clone(self) -> "State":
        return copy.deepcopy(self)


def resized_state(source_state: State, new_grid_size: tuple[int, int]) -> State:
    target = State(new_grid_size)
    target.paste(source_state)
    return target
