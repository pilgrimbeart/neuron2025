"""Checks that every cell rule in rules/ keeps the golden rules (python test_rules.py):

  - it imports nothing that knows about the grid, the body or the world (only numpy, numba, math and other rules);
  - its update takes exactly (me, mean, count, inp, out): it is given nothing else;
  - every cell is born the same, bar the numbers declared random at birth;
  - it runs on two very different sizes, with every array index checked (a rule's arrays are copies of its own
    numbers and its neighbours' means, so an index past their end can't reach another cell's numbers; the check
    makes such an index fail here rather than read garbage).
And that the body and world code (anything not in rules/ or sheet.py) touches the sheet only through cells(), give(),
get(), take() and view().
"""

from __future__ import annotations

import os

os.environ["NUMBA_BOUNDSCHECK"] = "1"   # every rule runs with its array indices checked: none can reach past its own

import ast
import importlib
import inspect
import pathlib
import sys

import numpy as np

import sheet

HERE = pathlib.Path(__file__).parent
ALLOWED = {"numpy", "numba", "math", "__future__", "rules"}      # rules: one rule may build on another


def imports(path: pathlib.Path) -> set:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add((node.module or "").split(".")[0])
    return out


def check_rule(path: pathlib.Path) -> list:
    problems = []
    bad = imports(path) - ALLOWED
    if bad:
        problems.append(f"imports {sorted(bad)}")
    rule = importlib.import_module(f"rules.{path.stem}")
    params = list(inspect.signature(rule.update.py_func).parameters)
    if params != ["me", "mean", "count", "inp", "out"]:
        problems.append(f"update takes {params}")
    for n in (9, 64):
        s = sheet.Sheet(rule, n, seed=1)
        for name, i in s.index.items():
            if name not in rule.BIRTH_NOISE and np.ptp(s._s[:, :, i]) > 0:
                problems.append(f"{name} differs between cells at birth")
        s.step(20)
    return problems


def check_body(path: pathlib.Path) -> list:
    text = path.read_text()
    return [f"reaches into the sheet ({word})" for word in ("._s", "._inp", "._mean", "_ticks") if word in text]


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    failed = False
    for path in sorted((HERE / "rules").glob("*.py")):
        problems = check_rule(path)
        print(f"rule {path.stem}: {'ok' if not problems else '; '.join(problems)}")
        failed |= bool(problems)
    for path in sorted(HERE.glob("*.py")):
        if path.name not in ("sheet.py", "test_rules.py"):
            problems = check_body(path)
            print(f"body {path.stem}: {'ok' if not problems else '; '.join(problems)}")
            failed |= bool(problems)
    raise SystemExit(1 if failed else 0)
