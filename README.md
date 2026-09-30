# Neuron 2025

The aim is **emergent learning**: a sheet of identical cells, running one simple local rule, that learns within its own lifetime. The final test is a round robot, run by such a sheet, that bumbles round an arena of red and blue blocks and learns, within a run, which colour is food and which is poison.

Earlier, the cell rule was designed by hand: an excitable medium with gates built from first principles and hand-made learning rules. That work, which still runs, is in `old_manual_gates/`. Its rules learned routing but not the robot task. Now the cell rule is **evolved**. What we learned along the way is in `LESSONS.md`.

## The cell (`cell.py`)

Each cell holds three numbers, each kept within −1..1:
- **v:** fast. Sensor pulses and tastes kick it, and an actuator fires when its v rises through 0.5.
- **w:** fast, free for evolution to use, e.g. as recovery.
- **m:** ten times slower, free to use as memory.

Every tick, each variable moves by its rate (0.1, 0.1, 0.01) times a quadratic polynomial of the cell's own variables and the mean of each over its 8 neighbours. A rule is those polynomials' coefficients (84 of them). The aim is for most to end up zero, so that the rule can be said in a sentence or two (`cell.describe`).

## The robot (`robot.py`)

- **The brain** is a 12×12 disc of cells.
- **Six sensor groups** sit round the rim, each a red and a blue input cell with a thruster between them.
- **Sensing:** a sensor sees colour in the direction it faces. Each tick its input cell is kicked with a probability that rises with what it sees: sparse pulses, never levels.
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

`evolve.py` climbs one hill with CMA-ES. `mapelites.py` searches widely: it keeps the best rule for each kind of behaviour (meals per life, and how eating food and eating poison change during a life), so stepping stones towards learning survive even when they score below "eat everything". Its best learner is checked on fresh worlds as it runs, and saved to `kernels/map_taste_validated.json` only if it learns there too.

Rules are saved in `kernels/`.

## Watching (`app.py`)

```bash
python app.py approach -c "task approach" -c "speed 2"
```

The screen is in four quarters:
- **Top-left, the cells:** v green when positive and red when negative, m (or w) in blue. Input cells are outlined red and blue, thrusters yellow, taste cells white.
- **Top-right, the world:** food is ringed in white.
- **Bottom-left:** each thruster's v over time.
- **Bottom-right:** the console. Commands can also be written to `control_in.txt`, and output goes to `control_out.log`.

Commands: `kernel NAME`, `task NAME`, `food red|blue`, `seed N`, `restart`, `speed N`, `pause`, `resume`, `view m|w`, `rule`, `quit`.

## Requirements

Python 3, NumPy, Numba, pygame, and cma (for CMA-ES).
