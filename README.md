# Neuron 2025

The aim is **emergent learning**: a sheet of identical cells, running one simple local rule, that learns within its own lifetime. The final test is a round robot, run by such a sheet, that bumbles round an arena of red and blue blocks and learns, within a run, which colour is food and which is poison.

The cell rule is **evolved**. It was first designed by hand: an excitable medium with gates built from first principles and hand-made learning rules. That work still runs, in `old_manual_gates/`. Its rules learned routing, but not the robot task.

- `LESSONS.md`: what we've learned with the evolved rule, and the principles carried over from the hand-designed one.
- `AGENTS.md`: how we work on the project.

## The cell (`cell.py`)

Each cell holds four numbers, each kept within −1..1:
- **v:** fast. Sensor pulses kick it, and an actuator fires when its v rises through 0.5.
- **w:** fast, free for evolution to use, e.g. as recovery.
- **m:** ten times slower, free to use as memory.
- **t:** fast. Only tastes kick it, up for food and down for poison, like a neuromodulator rather than a spike.

Every tick, each variable moves by its rate (0.1, 0.1, 0.01, 0.1) × dt times a quadratic polynomial of the cell's own variables and the mean of each over its neighbours (up to 8). dt is the simulated time per tick (`World.dt`, 1 by default). A rule is those polynomials' coefficients (180 of them). The aim is for most to end up zero, so that the rule can be said in a sentence or two (`cell.describe`).

A cell knows only its own variables and its neighbours'. The sheet is a list of cells with neighbour lists, and the body decides who neighbours whom. For speed, a genome's lives are stepped together, and the rule is written out from `TERMS` as one expression per variable so that it vectorises; the physics is the same.

## The robot (`robot.py`)

- **The brain** is a disc of 448 cells on a 24×24 grid.
- **Six sensor groups** sit round the rim, spaced out so that structure can form around them. Each group has:
  - a red and a blue input cell, on the rim, 3 cells either side;
  - a thruster, 3 cells in from the rim.
- **Sensing:** a sensor sees colour in the direction it faces, fading with distance. Its input cell is kicked at random at a rate that rises with what it sees (a Poisson process): sparse pulses, never levels.
- **Moving:** each thruster pushes the robot away from its own side as its v rises through 0.5. So approaching what one side sees needs the far side's thruster, and signals must cross the disc.
- **Eating:** touching a block eats it. Taste kicks t in the red and blue input cells of the sensor group facing the block (where the colour signals enter, since taste can't be carried far), and in the middle of the disc: up for food, down for poison. The strengths of the two are body genes that evolution can set.
- **Lives** start from near-blank noise (±0.1 in every variable). Each life has its own random stream, so a world's food and poison lives are identical until the first taste. Counts are kept per third of the life.

Tasks, shortest first:

| Task | World |
|---|---|
| `move` | no blocks |
| `approach` | one block, which is food |
| `taste` | one red block, food in one life and poison in the other |
| `choose` | one red and one blue block, one of them food |
| `forage` | several of each |
| `graze` | 4 of each, all food, seen only near by (so the nearest dominates) |
| `discriminate` | as graze, one colour food and the other poison; long lives (tens of tastes of each) |

## Evolving (`evolve.py`, `mapelites.py`)

The curriculum, each stage starting from the last:

```bash
python evolve.py move 60
python evolve.py approach 250 --from kernels/move.json --sigma 0.5 --sparsity 0.001
python evolve.py graze 150 --from kernels/approach.json --sigma 0.1 --sparsity 0.001
python mapelites.py 10 --task discriminate --seed kernels/graze.json     # hours; --resume to continue
```

**Scoring, common to both:**
- **Tick sizes:** every candidate is scored at dt 1, 0.5 and 0.25, taking the worst, so no rule can rely on the tick size. Validation checks 0.125 too, on fresh worlds.
- **Activity costs:** each thruster firing, and the sheet's spike rate above one spike per cell per 100 units of time. Sparse activity is favoured. Move and approach are exempt (getting the robot going comes first).
- **Learning tasks** (taste, choose, discriminate) are lived once with each colour as food. They're scored on the worse of the two lives, plus a bonus for any rise in food share from the first third of a life to the last. Discriminate scores only the last third, so gradual learning gets credit.

**The two programs:**
- **`evolve.py`** climbs one hill with CMA-ES. Each candidate lives in 8 worlds. Each generation it validates the search's mean on 32 fresh worlds, and saves it to `kernels/TASK.json` whenever it beats the best so far.
- **`mapelites.py`** searches widely. It keeps the best rule for each kind of behaviour (meals per life, and how eating food and eating poison change from the first third of a life to the last), so stepping stones towards learning survive even when they score below "eat everything".
  - **Racing:** tick sizes are added one at a time, coarsest first, and only for candidates that could still beat their niche's best. Most candidates only ever live at dt 1.
  - **Seeding:** `--seed` adds saved rules (kernels, with small variations, or a whole archive `.pkl`).
  - **Outputs:** `kernels/map_TASK.pkl` (the archive), `map_TASK.json` (the best score), `map_TASK_learner.json` (the best learner on its own worlds) and `map_TASK_validated.json` (the best learner on fresh worlds, at every tick size).

**Rules in `kernels/`:** the current search's outputs, for the current 24×24 body. Rules from earlier bodies are kept for reference:
- `kernels/grid12/`: the 12×12 body (commit `1dd377d` and before), including the taste learners;
- `kernels/grid24_mean/`: the 24×24 body with a separate taste cell (its move, approach and graze rules still work, since they don't depend on taste);
- `kernels/grid48/`: the 48×48 body.

## Watching (`app.py`)

```bash
python app.py approach -c "task approach" -c "speed 1" -c "pause"     # then type resume
```

The screen is in four quarters:
- **Top-left, the cells:**
  - `view all` (the default): v green when positive and red when negative, m in blue; `view v`, `view w`, `view m` or `view t`: that variable alone, in grey (−1 black, 0 grey, +1 white);
  - special cells are lettered: R and B (red and blue inputs, which also receive taste), M (thrusters), c (centre taste cells);
  - hovering over a cell describes it in the top-right corner, with its current variables.
- **Top-right, the world:** food is ringed in white. The panel shows meals, thruster firings, and spikes per cell per 100 units of time.
- **Bottom-left:** each thruster's v over time.
- **Bottom-right:** the console. Commands can also be written to `control_in.txt`, and output goes to `control_out.log`.

Commands: `kernel NAME`, `task NAME`, `food red|blue` (in taste, `food blue` makes the red block poison), `seed N`, `dt X`, `restart`, `speed N`, `pause`, `resume`, `step N`, `view all|v|w|m|t`, `rule`, `quit`. Clicking a panel gives it the keyboard: with the cells panel focused (white outline), the space bar does `step 1`; click the console to type.

## Where things stand

**Achieved on the 12×12 sheet:**
- move, approach and graze, at every tick size;
- a one-bit switch: in the taste task (one red block), one bite decides it, and after poison the robot keeps away. This works at every tick size, and sparsely (pulse trains rather than broad waves). But it's sensitisation, not association: in forage it eats less of both colours.

**Not found:** learning which colour is food, in choose (1–3 meals per life, so one-shot) or in discriminate (tens of tastes per life). With six sensor groups crammed together and cells that see only their neighbours' mean, there seemed to be no room for structure that tells red from blue.

**Growing the sheet** (`LESSONS.md`, "Growing the sheet"):
- **24×24:** graze reached 32 meals per 2,000 time units, twice the 12×12 best, and early rules build lasting, circuit-like structure from noise. But 10 hours of discriminate found no colour learning.
- **48×48:** worse (approach 1.24, graze 16 meals) and no sparser, so we went back to 24×24.

**Now:** discriminate on 24×24, with one change: taste arrives at the input cells, where the colour signals enter. The hand-designed seed experiments showed colour-specific learning needs that (`LESSONS.md`, "Designing a seed rule by hand"). The search is seeded with the 24×24 graze rule and the previous discriminate archive.

## Next steps

1. **Discriminate with taste at the inputs:** see whether colour learning appears now.
2. **Understand the circuits:** how much they vary across seeds, whether they settle, which variables form them, and whether signals travel along them from sensors to thrusters.
3. **If colour learning still fails,** the next candidates:
   - **Gradient inputs:** each variable's east-minus-west and south-minus-north difference as well as its mean (612 coefficients, about 3× the cost; a shared compass).
   - **The fuller cell (parked):** each neighbour's variables as separate inputs, so a cell can tell which neighbour fired.
4. **Organism-wide memory:** spread what one side learned to the others.
5. **Make rules describable:** prune terms one by one while the score holds.
6. **Side-quest, time:** the medium is continuous (no global clock). Two alternatives are worth trying later:
   - rewarding sheets that settle into a stable, low-activity oscillation emerging from the cells (allowed: a property of the medium, not of the tick);
   - a synchronous medium like a cellular automaton (score at dt 1 only), as in Life or Rule 110: cheaper, but less biological.
7. **Parked, in git history:** the fuller cell above (2,812 coefficients), a sheet that grows from one central cell along a square spiral, and an untested PyTorch version of the simulation for running many lives at once on a GPU. It's the commit "Parked: ..." just before "Revert ...": `git revert` the revert to bring it back. A GPU only pays once the simulation is batched across genomes too.

## Requirements

Python 3, NumPy, Numba, pygame, and cma (for CMA-ES), in `.venv/`.
