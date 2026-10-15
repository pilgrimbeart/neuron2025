# Agent Notes

This is a private project.

When changing the codebase, prefer a complete rip-up-and-rewrite with breaking changes when that leads to a cleaner, smaller, more coherent design.

Do not favor incremental edits that preserve legacy structure at the cost of accumulating spaghetti.

Minimalism, clarity, and a single obvious architecture are preferred over compatibility with earlier internal layouts.

No one else works on this project. Always work directly on `main` -- commit and push there, don't create feature branches or PRs.

Commit and push only when the user says "Push".

## Working on the simulator

- The goal is emergent learning: a sheet of identical cells, running one local rule, that builds itself into a brain and learns within its lifetime. The test is the creature learning which colour is food. See the README.
- The current approach is development in stages: each capability (`frame.py`, `paths.py`, `learn.py`, ...) is designed by hand and proven on its own, then composed (`develop.py`, `creature.py`), each stage handing over by a local trigger. Keep each rule small enough to say in a sentence or two, and the numbers cells share to a minimum. The earlier evolved cell rule (`cell.py`, `evolve.py`, `mapelites.py`, rules in `kernels/`) is kept and still runs.
- **The cell.** All cells start identical, differing only by independent random noise. A cell may hold as many numbers, in as many layers, as it likes, and a layer may read any of its own cell's layers; each update sees only that cell's own numbers and the numbers its neighbours share (its **shared numbers**: count them, the tag included, and keep them few). Neighbours see a shared number exactly as the cell holds it, noise included: information must sit in steps much larger than the noise, in a modest range, never in an encoding that squeezes it below the noise or stretches it over many orders of magnitude (sharing exp(-5 d) with the noise added to d first, so the mean gave the smallest d, was a cheat). All of a cell's state lives in the sheet's arrays, in the cell kernels, never in the harness (no Python dicts of "the red cell's balance"). The harness is only body and world: it delivers kicks and tastes to, and reads outputs from, cells that have named themselves, elementwise, and never decides anything from a sheet-wide view (no `argwhere` over the sheet to decide what happens; watching and measuring may use one). `python creature.py --check` checks the first part.
- **Stochastic updates only.** Every cell updates at random times (each tick, each cell updates with some probability, otherwise keeps its state): never all cells in lockstep. Lockstep lets rules depend on a shared rhythm and makes artefacts (flipping every tick, checkerboard oscillations). This applies to everything new; the robot's cell step (`cell.py`) predates it and still updates in lockstep, and must change before more robot work.
- Every rule must also be robust to some noise in its cells' numbers.
- **Any size** (the aim; for now the creature is held to two sizes, 49 and 97, with one body). Every rule must work on a sheet of any size: only the time it takes may grow with the size, never whether it works. No constant may silently assume a size (a fixed wait, a cap on a count, a range, a threshold tuned to one sheet); stages hand over on "this stage's work is done here", not after a fixed time. Test every capability on at least two very different sizes (e.g. 49 and 512).
- Everything outside the cell is body and world (`creature.py`'s `Body`, `robot.py`). Local behaviour only, no global god: inputs are sparse kicks (pulses), never continuous levels; cells must never know the sheet's size or position beyond what their neighbours tell them; and learning must work with events in random order, not a crafted training sequence.
- Check evolved results on fresh worlds and at other tick sizes, not just the worlds they were evolved in, before reporting them; fixed evaluation worlds and a fixed tick let flukes and artefacts in.
- When the body or the sheet size changes, evolved rules no longer apply: keep them aside (as `kernels/grid12/`) and evolve again from the start of the curriculum. Penalties that depend on the number of cells or the size of a rule may need scaling.
- Measure, and trace failures to a cause before fixing them. Record findings and dead ends in `LESSONS.md`.
- Keep experimental setups that are still in use side by side in the code, sharing modules, rather than ripping one up for the next. When a setup is finished, tag it in git (`exp/NAME`), list it under "Experiments" in the README and name the tag in `LESSONS.md`, so it can be brought back exactly.
- Before reporting a behaviour as working, also show it live so the user can watch: the experiment's `--watch` view, or for the evolved robot the running app (`python app.py`), driven through its control channel (`control_in.txt` / `control_out.log`).
- The hand-designed system (excitable medium, gates, `verify.py`) is in `old_manual_gates/` and still runs from there (`cd old_manual_gates && python neuron.py`); its old rules (shared `DEFAULT_PARAMS`, `verify.py` before committing) apply only to changes there.
