# Agent Notes

This is a private project.

When changing the codebase, prefer a complete rip-up-and-rewrite with breaking changes when that leads to a cleaner, smaller, more coherent design.

Do not favor incremental edits that preserve legacy structure at the cost of accumulating spaghetti.

Minimalism, clarity, and a single obvious architecture are preferred over compatibility with earlier internal layouts.

No one else works on this project. Always work directly on `main` -- commit and push there, don't create feature branches or PRs.

Commit and push only when the user says "Push".

## Working on the simulator

- The goal is emergent learning: a sheet of identical cells, running one evolved local rule, that learns within its lifetime. The final test is the robot learning which colour is food. See the README.
- `cell.py` is the whole of how a cell behaves; the rule itself is found by evolution (`evolve.py`, `mapelites.py`) and saved in `kernels/`. Keep `cell.py` small, and aim for evolved rules that can be said in a sentence or two.
- **Stochastic updates only.** Every cell updates at random times (each tick, each cell updates with some probability, otherwise keeps its state): never all cells in lockstep. Lockstep lets rules depend on a shared rhythm and makes artefacts (flipping every tick, checkerboard oscillations). This applies to everything new; the robot's cell step (`cell.py`) predates it and still updates in lockstep, and must change before more robot work.
- Every rule must also be robust to some noise in its cells' numbers.
- **Any size.** Every rule must work on a sheet of any size: only the time it takes may grow with the size, never whether it works. No constant may silently assume a size (a fixed wait, a cap on a count, a range, a threshold tuned to one sheet); stages hand over on "this stage's work is done here", not after a fixed time. Test every capability on at least two very different sizes (e.g. 49 and 512).
- Everything outside the cell is body and world (`robot.py`). Local behaviour only, no global god: inputs are sparse kicks (pulses), never continuous levels; cells must never know the sheet's size or position beyond what their neighbours tell them; and learning must work with events in random order, not a crafted training sequence.
- Check evolved results on fresh worlds and at other tick sizes, not just the worlds they were evolved in, before reporting them; fixed evaluation worlds and a fixed tick let flukes and artefacts in.
- When the body or the sheet size changes, evolved rules no longer apply: keep them aside (as `kernels/grid12/`) and evolve again from the start of the curriculum. Penalties that depend on the number of cells or the size of a rule may need scaling.
- Measure, and trace failures to a cause before fixing them. Record findings and dead ends in `LESSONS.md`.
- Keep experimental setups that are still in use side by side in the code (sharing `cell.py`), rather than ripping one up for the next. When a setup is finished, tag it in git (`exp/NAME`), list it under "Experiments" in the README and name the tag in `LESSONS.md`, so it can be brought back exactly.
- Before reporting a behaviour as working, also show it live: drive the running app (`python app.py`) through the control channel (`control_in.txt` / `control_out.log`) so the user can watch.
- The hand-designed system (excitable medium, gates, `verify.py`) is in `old_manual_gates/` and still runs from there (`cd old_manual_gates && python neuron.py`); its old rules (shared `DEFAULT_PARAMS`, `verify.py` before committing) apply only to changes there.
