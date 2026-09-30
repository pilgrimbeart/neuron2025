# Neuron 2025

The aim is **emergent learning**: a sheet of identical cells, running one simple local rule, that learns within its own lifetime. The final test is a round robot, run by such a sheet, that bumbles round an arena of red and blue blocks and learns, within a run, which colour is food and which is poison.

The cell rule is **evolved**. It was first designed by hand: an excitable medium with gates built from first principles and hand-made learning rules. That work still runs, in `old_manual_gates/`. Its rules learned routing, but not the robot task.

- `LESSONS.md`: what we've learned with the evolved rule, and the principles carried over from the hand-designed one.
- `AGENTS.md`: how we work on the project.

## The cell (`cell.py`)

Each cell holds three numbers, each kept within −1..1:
- **v:** fast. Sensor pulses and tastes kick it, and an actuator fires when its v rises through 0.5.
- **w:** fast, free for evolution to use, e.g. as recovery.
- **m:** ten times slower, free to use as memory.

Every tick, each variable moves by its rate (0.1, 0.1, 0.01) times a quadratic polynomial of the cell's own variables and the mean of each over its neighbours (up to 8). A rule is those polynomials' coefficients (84 of them). The aim is for most to end up zero, so that the rule can be said in a sentence or two (`cell.describe`).

A cell knows only its own variables and its neighbours'. The sheet is a list of cells with neighbour lists, and the body decides who neighbours whom.

## The robot (`robot.py`)

- **The brain** is a disc of cells on a 12×12 grid.
- **Six sensor groups** sit round the rim, each a red and a blue input cell with a thruster between them.
- **Sensing:** a sensor sees colour in the direction it faces, fading with distance. Each tick its input cell is kicked with a probability that rises with what it sees: sparse pulses, never levels.
- **Moving:** each thruster pushes the robot away from its own side as its v rises through 0.5. So approaching what one side sees needs the far side's thruster, and signals must cross the disc.
- **Eating:** touching a block eats it. Taste kicks the cell beside the sensor group that touched it, and the middle of the disc: up for food, down for poison. The strengths of the two are body genes that evolution can set.

Tasks, shortest first:

| Task | World |
|---|---|
| `move` | no blocks |
| `approach` | one block, which is food |
| `taste` | one red block, food in one life and poison in the other |
| `choose` | one red and one blue block, one of them food |
| `forage` | several of each |

## Evolving (`evolve.py`, `mapelites.py`)

```bash
python evolve.py move 60
python evolve.py approach 500 --from kernels/move.json --sigma 0.5
python evolve.py taste 800 --from kernels/approach.json --sigma 0.02 --sigma-m 0.5 --sparsity 0.001
python mapelites.py 8 --task taste            # 8 hours; --resume to continue
```

- **`evolve.py`** climbs one hill with CMA-ES. Each generation it scores the rule at the search's mean on fixed validation worlds, and saves it to `kernels/TASK.json` whenever it beats the best so far.
- **`mapelites.py`** searches widely. It keeps the best rule for each kind of behaviour (meals per life, and how eating food and eating poison change during a life), so stepping stones towards learning survive even when they score below "eat everything". It writes these files, and overwrites them as it goes:
  - `kernels/map_TASK.pkl`: the archive;
  - `kernels/map_TASK.json`: the best-scoring rule;
  - `kernels/map_TASK_learner.json`: the best learner on its own worlds;
  - `kernels/map_TASK_validated.json`: the best learner that also learns on fresh worlds.

Rules in `kernels/`:

| File | What it is |
|---|---|
| `move.json`, `approach.json` | the move and approach rules; approach eats about 2.6 blocks per life |
| `approach_vw.json` | the approach rule as evolved, before m existed |
| `taste_learner.json` | the first rule that learns (the taste task), fixed as a reference |
| `map_taste*.json` | outputs of the latest MAP-Elites run |

## Watching (`app.py`)

```bash
python app.py taste_learner -c "task taste" -c "speed 3"
```

The screen is in four quarters:
- **Top-left, the cells:** v green when positive and red when negative, m (or w) in blue. Input cells are outlined red and blue, thrusters yellow, taste cells white.
- **Top-right, the world:** food is ringed in white.
- **Bottom-left:** each thruster's v over time.
- **Bottom-right:** the console. Commands can also be written to `control_in.txt`, and output goes to `control_out.log`.

Commands: `kernel NAME`, `task NAME`, `food red|blue` (in taste, `food blue` makes the red block poison), `seed N`, `restart`, `speed N`, `pause`, `resume`, `view m|w`, `rule`, `quit`.

## Where things stand

- **Works:** move and approach evolve in minutes.
- **Learns, via MAP-Elites:** the taste task. On fresh worlds, poison is eaten 85% less in the second half of a life, while food is eaten more, and the robot keeps away from poison rather than freezing. The memory is the w of the taste cell that received the bad taste. So it is local: only the side that bit has learned.
- **Not yet:** choose, which needs colour-specific learning ("red is bad, blue is good"), and forage.

## Next steps

1. **Choose:** run MAP-Elites on choose, seeded with the taste run's archive.
2. **Organism-wide memory:** spread what one side learned to the others, e.g. through centre cells.
3. **A cell "type" variable,** set early in life and then fixed, if a task needs cells with different jobs.
4. **Make rules describable:** penalise m (so it rests at 0 unless something is learned), then prune terms one by one while the score holds.
5. **Speed:**
   - `cell.py` now steps cells from neighbour lists, one term at a time across all cells, with no per-tick allocation. It's checked to behave the same (statistically, since the dynamics are chaotic), but not yet benchmarked against the previous version, in real and CPU time.
   - The laptop's Intel Arc GPU is visible to WSL (`/dev/dxg`). Using it would need Intel's compute runtime and PyTorch's Intel GPU backend, and a batched rewrite of the simulation.

## Requirements

Python 3, NumPy, Numba, pygame, and cma (for CMA-ES), in `.venv/`.
