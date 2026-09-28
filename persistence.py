from __future__ import annotations

import copy
import glob
import json
from pathlib import Path

import numpy as np

from model import SimulationConfig, State


SNAPSHOT_SUFFIX = ".json"
SNAPSHOT_DIR = Path("patterns")


def snapshot_path(filename: str) -> Path:
    return SNAPSHOT_DIR / f"{filename}{SNAPSHOT_SUFFIX}"


def list_snapshot_names() -> list[str]:
    return [
        path.stem
        for path in sorted(SNAPSHOT_DIR.glob(f"*{SNAPSHOT_SUFFIX}"))
    ]


def save_snapshot(
    filename: str,
    state: State,
    probes: list[dict],
    config: SimulationConfig,
) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    obj = {
        "kind": state.kind_array.tolist(),
        "energy": state.energy_array.tolist(),
        "flame": state.flame_array.tolist(),
        "weight": state.weight_array.tolist(),
        "probes": [{"xy": list(p["xy"]), **({"label": p["label"]} if p.get("label") else {})} for p in probes],
        "vars": copy.deepcopy(config.vars),
    }
    with open(snapshot_path(filename), "wt") as handle:
        json.dump(obj, handle, indent=4)


def load_snapshot(filename: str) -> tuple[State, list[dict], dict[str, float]]:
    with open(snapshot_path(filename), "rt") as handle:
        obj = json.load(handle)

    state = State((len(obj["kind"]), len(obj["kind"][0])))
    state.kind_array = np.array(obj["kind"], dtype=np.int8)
    state.energy_array = np.array(obj["energy"], dtype=float)
    state.flame_array = np.array(obj["flame"], dtype=float)
    state.weight_array = np.array(obj["weight"], dtype=float)

    probes = obj.get("probes", [])
    for probe in probes:
        probe["xy"] = tuple(probe["xy"])

    return state, probes, obj.get("vars", {})
