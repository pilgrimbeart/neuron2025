from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

import physics


@dataclass
class SimulationConfig:
    vars: dict[str, float] = field(default_factory=lambda: copy.deepcopy(physics.DEFAULT_PARAMS))

    def get(self, name: str) -> float:
        return self.vars[name]

    def set(self, name: str, value: float) -> None:
        self.vars[name] = float(value)

    def update_from_dict(self, values: dict[str, float]) -> None:
        for key, value in values.items():
            self.vars[key] = float(value)

    def stuck_on_risk(self) -> bool:
        return physics.stuck_on_risk(self.vars)


class State:
    """The grid of cells, plus editing operations. The cell behaviour itself is in physics.py."""

    def __init__(self, grid_size: tuple[int, int]):
        self.grid_size = grid_size
        self.enabled_array = np.zeros(self.grid_size, dtype=bool)
        self.energy_array = np.zeros(self.grid_size, dtype=float)
        self.flame_array = np.zeros(self.grid_size, dtype=float)
        self.illumination_array = np.zeros(self.grid_size, dtype=float)

    def set_all_enableds(self, state: bool) -> None:
        self.enabled_array[:] = state

    def reset_energy_and_flame(self) -> None:
        """Flame off, energy full wherever enabled."""
        self.energy_array[:] = 0
        self.energy_array[self.enabled_array] = 1.0
        self.flame_array[:] = 0
        self.illumination_array[:] = 0

    def update(self, delta_s: float, config: SimulationConfig) -> None:
        self.energy_array, self.flame_array, self.illumination_array = physics.step(
            self.enabled_array, self.energy_array, self.flame_array, config.vars, delta_s
        )

    def strike(self, xy: tuple[int, int], config: SimulationConfig) -> bool:
        if not physics.can_ignite(self.enabled_array[xy], self.energy_array[xy], config.vars):
            return False
        self.flame_array[xy] = config.get("STRIKE_LEVEL")
        return True

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
        """Copy source_state in, centred: source cell (x, y) lands at (x, y) + centre_offset."""
        offset = centre_offset(source_state.grid_size, self.grid_size)

        def paste_array(foreground: np.ndarray, background: np.ndarray) -> None:
            dst, src = [], []
            for off, fg_len, bg_len in zip(offset, foreground.shape, background.shape):
                start, end = max(0, off), min(bg_len, off + fg_len)
                dst.append(slice(start, end))
                src.append(slice(start - off, end - off))
            background[tuple(dst)] = foreground[tuple(src)]

        paste_array(source_state.enabled_array, self.enabled_array)
        paste_array(source_state.energy_array, self.energy_array)
        paste_array(source_state.flame_array, self.flame_array)
        paste_array(source_state.illumination_array, self.illumination_array)

    def clone(self) -> "State":
        return copy.deepcopy(self)


def centre_offset(old_size: tuple[int, int], new_size: tuple[int, int]) -> tuple[int, int]:
    return ((new_size[0] - old_size[0]) // 2, (new_size[1] - old_size[1]) // 2)


def resized_state(source_state: State, new_grid_size: tuple[int, int]) -> State:
    target = State(new_grid_size)
    target.paste(source_state)
    return target
