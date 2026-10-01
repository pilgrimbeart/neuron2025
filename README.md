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

A cell knows only its own variables and its neighbours'. The sheet is a list of cells with neighbour lists, and the body decides who neighbours whom.

## The robot (`robot.py`)

- **The brain** is a disc of cells on a 12×12 grid.
- **Six sensor groups** sit round the rim, each a red and a blue input cell with a thruster between them.
- **Sensing:** a sensor sees colour in the direction it faces, fading with distance. Each tick its input cell is kicked with a probability that rises with what it sees: sparse pulses, never levels.
- **Moving:** each thruster pushes the robot away from its own side as its v rises through 0.5. So approaching what one side sees needs the far side's thruster, and signals must cross the disc.
- **Eating:** touching a block eats it. Taste kicks t in the cell beside the sensor group that touched it, and in the middle of the disc: up for food, down for poison. The strengths of the two are body genes that evolution can set.

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

Both score every candidate at two tick sizes (dt 1 and 0.5) and take the worse, so that no rule can rely on the tick size, and they validate at 0.25 as well. Both charge for activity: each thruster firing, and the sheet's spike rate above one spike per cell per 100 units of time, so sparse activity is favoured.

- **`evolve.py`** climbs one hill with CMA-ES. Each generation it scores the rule at the search's mean on fixed validation worlds, and saves it to `kernels/TASK.json` whenever it beats the best so far.
- **`mapelites.py`** searches widely. It keeps the best rule for each kind of behaviour (meals per life, and how eating food and eating poison change during a life), so stepping stones towards learning survive even when they score below "eat everything". It writes these files, and overwrites them as it goes:
  - `kernels/map_TASK.pkl`: the archive;
  - `kernels/map_TASK.json`: the best-scoring rule;
  - `kernels/map_TASK_learner.json`: the best learner on its own worlds;
  - `kernels/map_TASK_validated.json`: the best learner that also learns on fresh worlds, at every tick size.

  `--seed` adds saved rules to the first batches: kernels (with small variations) or a whole archive (`.pkl`).

Rules in `kernels/`:

| File | What it is |
|---|---|
| `move.json`, `approach.json` | the move and approach rules, at every tick size; approach eats about 3 blocks per life |
| `robust_learner.json` | the first rule that learns the taste task at every tick size; dense (travelling waves, about 6 spikes per cell per 100) |
| `sparse_robust_learner.json` | the best so far: learns the taste task at every tick size, sparsely (about 2.6 spikes per cell per 100; pulse trains rather than broad waves) |
| `dense_scorer.json` | the best taste score before activity was charged for |
| `map_taste*.json` | outputs of the latest MAP-Elites run |

## Watching (`app.py`)

```bash
python app.py sparse_robust_learner -c "task taste" -c "view t" -c "speed 4"
```

The screen is in four quarters:
- **Top-left, the cells:** v green when positive and red when negative, m (or w) in blue. Input cells are outlined red and blue, thrusters yellow, taste cells white.
- **Top-right, the world:** food is ringed in white.
- **Bottom-left:** each thruster's v over time.
- **Bottom-right:** the console. Commands can also be written to `control_in.txt`, and output goes to `control_out.log`.

Commands: `kernel NAME`, `task NAME`, `food red|blue` (in taste, `food blue` makes the red block poison), `seed N`, `dt X`, `restart`, `speed N`, `pause`, `resume`, `view m|w|t`, `rule`, `quit`. The world panel shows meals, thruster firings and spikes per cell per 100 units of time.

## Where things stand

- **Works:** move and approach evolve in minutes, at every tick size.
- **Learns, via MAP-Elites:** the taste task, at every tick size, and sparsely. With the best rule (`sparse_robust_learner.json`), the first taste decides it: after poison it stays away, after food it keeps eating. Two things made that possible: giving taste its own variable t, and scoring at more than one tick size. Charging for activity turned the broad travelling waves into something closer to pulse trains.
- **Not yet:** choose and forage. The taste learner can't tell colours apart: in forage it eats both alike and less of both over time, because in the taste task there was only ever one colour.

## Next steps

1. **Choose:** MAP-Elites on choose, seeded with the taste run's archive (`--seed kernels/map_taste.pkl`).
2. **Organism-wide memory:** spread what one side learned to the others.
3. **Make rules describable:** prune terms one by one while the score holds.
4. **Parked, in git history:** a fuller cell (each neighbour's variables as separate inputs, every product of two inputs: 2,812 coefficients), a sheet that grows from one central cell along a square spiral, and an untested PyTorch version of the simulation for running many lives at once on a GPU. It is the commit "Parked: ..." just before "Revert ...": `git revert` the revert to bring it back.
5. **Speed:** this laptop manages about 500 lives per second on 8 cores. Renting a GPU only pays once the simulation is batched (the parked PyTorch version).

## Requirements

Python 3, NumPy, Numba, pygame, and cma (for CMA-ES), in `.venv/`.
