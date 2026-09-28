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

    # Per-cell arrays and the value an empty or newly exposed cell holds.
    ARRAYS = {"kind": physics.EMPTY, "energy": 0.0, "flame": 0.0, "weight": 1.0, "illumination": 0.0, "modulator": 0.0}

    def __init__(self, grid_size: tuple[int, int]):
        self.grid_size = grid_size
        self.kind_array = np.zeros(grid_size, dtype=np.int8)
        for name, blank in self.ARRAYS.items():
            if name != "kind":
                setattr(self, f"{name}_array", np.full(grid_size, blank, dtype=float))

    def arrays(self):
        return [(name, blank, getattr(self, f"{name}_array")) for name, blank in self.ARRAYS.items()]

    def set_kind(self, xy: tuple[int, int], kind: int) -> None:
        """Place or remove one cell, fresh: full energy, no flame, neutral weight."""
        self.kind_array[xy] = kind
        self.energy_array[xy] = 1.0 if kind != physics.EMPTY else 0.0
        self.flame_array[xy] = 0.0
        self.weight_array[xy] = 1.0

    def set_all_kinds(self, kind: int) -> None:
        self.kind_array[:] = kind
        self.weight_array[:] = 1.0

    def reset_energy_and_flame(self) -> None:
        """Flame off, energy full wherever there is a cell."""
        self.energy_array[:] = np.where(self.kind_array != physics.EMPTY, 1.0, 0.0)
        self.flame_array[:] = 0
        self.illumination_array[:] = 0
        self.modulator_array[:] = 0

    def update(self, delta_s: float, config: SimulationConfig) -> None:
        self.energy_array, self.flame_array, self.weight_array, self.illumination_array, self.modulator_array = physics.step(
            self.kind_array, self.energy_array, self.flame_array, self.weight_array, config.vars, delta_s
        )

    def strike(self, xy: tuple[int, int], config: SimulationConfig) -> bool:
        if not physics.can_ignite(self.kind_array[xy], self.energy_array[xy], config.vars):
            return False
        self.flame_array[xy] = config.get("STRIKE_LEVEL")
        return True

    def shift(self, dx: int, dy: int) -> None:
        for name, blank, arr in self.arrays():
            moved = np.full_like(arr, blank)
            src_x = slice(max(0, -dx), arr.shape[0] - max(0, dx))
            dst_x = slice(max(0, dx), arr.shape[0] - max(0, -dx))
            src_y = slice(max(0, -dy), arr.shape[1] - max(0, dy))
            dst_y = slice(max(0, dy), arr.shape[1] - max(0, -dy))
            moved[dst_x, dst_y] = arr[src_x, src_y]
            setattr(self, f"{name}_array", moved)

    def split_shift(self, dx: int, dy: int, cursor_xy: tuple[int, int]) -> None:
        """Insert a blank row or column at the cursor, moving the cells on one side of it outward."""
        axis, d, c = (0, dx, cursor_xy[0]) if dx != 0 else (1, dy, cursor_xy[1])
        for name, blank, arr in self.arrays():
            arr = np.moveaxis(arr, axis, 0)
            new = arr.copy()
            if d > 0:
                new[c + 1:] = arr[c:-1]
            else:
                new[:c] = arr[1:c + 1]
            new[c] = blank
            setattr(self, f"{name}_array", np.moveaxis(new, 0, axis))

    def paste(self, source_state: "State") -> None:
        """Copy source_state in, centred: source cell (x, y) lands at (x, y) + centre_offset."""
        offset = centre_offset(source_state.grid_size, self.grid_size)
        dst, src = [], []
        for off, fg_len, bg_len in zip(offset, source_state.grid_size, self.grid_size):
            start, end = max(0, off), min(bg_len, off + fg_len)
            dst.append(slice(start, end))
            src.append(slice(start - off, end - off))
        for name, _, arr in self.arrays():
            arr[tuple(dst)] = getattr(source_state, f"{name}_array")[tuple(src)]

    def clone(self) -> "State":
        return copy.deepcopy(self)


def centre_offset(old_size: tuple[int, int], new_size: tuple[int, int]) -> tuple[int, int]:
    return ((new_size[0] - old_size[0]) // 2, (new_size[1] - old_size[1]) // 2)


def resized_state(source_state: State, new_grid_size: tuple[int, int]) -> State:
    target = State(new_grid_size)
    target.paste(source_state)
    return target
