# Neuron 2025

The aim is **emergent learning**: a sheet of identical cells, running one simple local rule, that learns within its own lifetime. The final test is a round robot, run by such a sheet, that bumbles round an arena of red and blue blocks and learns, within a run, which colour is food and which is poison.

The cell rule is **evolved**. It was first designed by hand: an excitable medium with gates built from first principles and hand-made learning rules. That work still runs, in `old_manual_gates/`. Its rules learned routing, but not the robot task.

**The picture we're aiming for:** a big sheet, a few sensors and actuators wired to its edges, and the sheet left to run. It grows its own sparse network, with most cells completely inactive, either by starting dense and pruning to paths (as slime mould does), or by growing paths through a quiet medium along gradients and then pruning them by use (as nervous systems do). When something new has to be learned, the growth and pruning can start again where needed.

- `LESSONS.md`: what we've learned with the evolved rule, and the principles carried over from the hand-designed one.
- `AGENTS.md`: how we work on the project.

## The cell (`cell.py`)

Each cell holds six numbers, each kept within −1..1:
- **v:** fast. Sensor pulses kick it, and an actuator fires when its v rises through 0.5.
- **w:** fast, free for evolution to use, e.g. as recovery.
- **m:** ten times slower, free to use as memory.
- **t:** fast. Only tastes kick it, up for food and down for poison, like a neuromodulator rather than a spike.
- **e** (fast) and **z** (ten times slower): free for learning to use, e.g. as an eligibility trace and a gate on where learning happens.

Every tick, each variable moves by its rate (0.1, 0.1, 0.01, 0.1, 0.1, 0.01) × dt times a polynomial of the cell's own variables and the mean of each over its neighbours (up to 8): 1, those 12 inputs, their pairwise products and squares, and the cubic products of the cell's own variables (so a three-factor learning rule such as e·t·z is one term). dt is the simulated time per tick (`World.dt`, 1 by default). A rule is those polynomials' coefficients (147 terms per variable, 882 in all). The aim is for most to end up zero, so that the rule can be said in a sentence or two (`cell.describe`). Rules saved with fewer variables or without the cubic terms load with the missing coefficients at zero, and behave as before.

Search can be limited to some variables (complexification): `--variables v,w` evolves only v's and w's equations in terms of each other (the rest stay zero), and `--freeze v,w,m,t` keeps the seed's rule among those variables as it was, so only terms involving the new ones are searched (in `evolve.py` and `mapelites.py`).

A cell knows only its own variables and its neighbours'. The sheet is a list of cells with neighbour lists, and the body decides who neighbours whom. For speed, a genome's lives are stepped together, and the rule is written out from `TERMS` term by term, one loop over lives per variable, so that it vectorises; the physics is the same.

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
| `forage` | several of each (`World.blocks`, 4 by default) |
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
- **Activity costs:** each thruster firing, and the sheet's spike rate (a spike is a cell's v rising through 0.5) above one spike per cell per 100 units of time. So it's oscillation that costs: a cell held high costs nothing. Move and approach are exempt (getting the robot going comes first).
- **Learning tasks** (taste, choose, discriminate) are lived once with each colour as food. They're scored on the worse of the two lives, plus a bonus for any rise in food share from the first third of a life to the last. Discriminate scores only the last third, so gradual learning gets credit.

**The two programs:**
- **`evolve.py`** climbs one hill with CMA-ES. Each candidate lives in 8 worlds. Each generation it validates the search's mean on 32 fresh worlds, and saves it to `kernels/TASK.json` whenever it beats the best so far.
- **`mapelites.py`** searches widely. It keeps the best rule for each kind of behaviour (meals per life, and how eating food and eating poison change from the first third of a life to the last), so stepping stones towards learning survive even when they score below "eat everything".
  - **Racing:** tick sizes are added one at a time, coarsest first, and only for candidates that could still beat their niche's best. Most candidates only ever live at dt 1.
  - **Seeding:** without `--seed` the search starts from random rules. With it, it starts from saved rules (kernels, or a whole archive `.pkl`) and variations of them, and keeps mutations small (up to 0.1, against 0.5 from scratch), since a working robot survives small changes to many terms but not large ones.
  - **Outputs:** `kernels/map_TASK.pkl` (the archive), `map_TASK.json` (the best score), `map_TASK_learner.json` (the best learner on its own worlds) and `map_TASK_validated.json` (the best learner on fresh worlds, at every tick size).

**Rules in `kernels/`:** searches write their outputs here, for the current 24×24 body. Also kept:
- `kernels/hand_seed.json`: the hand-designed rule (`LESSONS.md`, "Designing a seed rule by hand"); it learns on a bare sheet but doesn't move the robot;
- `kernels/seeded/`: discriminate searches on the current body seeded with the 24×24 graze rule (no colour learning).

Rules from earlier bodies:
- `kernels/grid12/`: the 12×12 body (commit `1dd377d` and before), including the taste learners;
- `kernels/grid24_mean/`: the 24×24 body with a separate taste cell (its move, approach and graze rules still work, since they don't depend on taste);
- `kernels/grid48/`: the 48×48 body;
- `kernels/vw_only/`: move, approach and graze with only v and w in use (`--variables v,w`): they move but hardly forage.

## Watching (`app.py`)

```bash
python app.py approach -c "task approach" -c "speed 1" -c "pause"     # then type resume
```

The screen is in four quarters:
- **Top-left, the cells:**
  - `view all` (the default): v green when positive and red when negative, m in blue; `view v` (or w, m, t, e, z): that variable alone, in grey (−1 black, 0 grey, +1 white);
  - special cells are lettered: R and B (red and blue inputs, which also receive taste), M (thrusters), c (centre taste cells);
  - hovering over a cell describes it in the top-right corner, with its current variables.
- **Top-right, the world:** food is ringed in white. The panel shows meals, thruster firings, and spikes per cell per 100 units of time.
- **Bottom-left:** each thruster's v over time.
- **Bottom-right:** the console. Commands can also be written to `control_in.txt`, and output goes to `control_out.log`.

Commands: `kernel NAME`, `task NAME`, `food red|blue` (in taste, `food blue` makes the red block poison), `seed N`, `dt X`, `restart`, `speed N`, `pause`, `resume`, `step N`, `view all|v|w|m|t|e|z`, `rule`, `quit`. Clicking a panel gives it the keyboard: with the cells panel focused (white outline), the space bar does `step 1`; click the console to type.

## Where things stand

**Achieved on the 12×12 sheet:**
- move, approach and graze, at every tick size;
- a one-bit switch: in the taste task (one red block), one bite decides it, and after poison the robot keeps away. This works at every tick size, and sparsely (pulse trains rather than broad waves). But it's sensitisation, not association: in forage it eats less of both colours.

**Achieved on 24×24:** move, approach and graze (32 meals per 2,000 time units), with sheets that build lasting, circuit-like structure from noise.

**Not found anywhere: learning which colour is food.** What we've tried for discriminate, all without a learner that holds up on fresh worlds (`LESSONS.md` has each):
- 12×12, 24×24 and 48×48 sheets (48×48 was worse, so we went back to 24×24);
- taste arriving at the input cells, where the colour signals enter;
- seeding with a hand-designed rule that learns on a bare sheet (it doesn't move the robot);
- complexification: a v,w-only curriculum (it moves but hardly forages), then the four-variable forager frozen with two new variables (e, z) and cubic terms for learning (aborted: too slow, and its mutations wrecked the forager; both since fixed).

**What we've learned about the search itself:**
- The curriculum carries one lineage: each stage hands one rule to the next, so learning can only build on one particular forager, chosen for eating, not for being teachable.
- Most random rules don't move: their sheets go quiet within a few firings, so they taste nothing, however much food is near (`World.blocks`).

## Next goal

**Grow wiring across a bare sheet: one network joining every source and every sink.** No spikes, no robot, no meaning attached to the ends: the first step towards a sheet that grows its own wiring (`LESSONS.md`, "Growing wiring"). Not a path for every pair (12 inputs × 6 thrusters would be 72), but one shared network in which every source can reach every sink through junctions, where paths converge and diverge; later, learning lives at the junctions (how much of a signal goes down each branch), while the wiring itself is grown by development.

`paths.py`, in two tricks: **sinks send out waves**, and **each cell remembers which neighbour the wave came from** (a breadcrumb). A source is on the path, and so is any cell a path cell's breadcrumb points to: a path is the trail of breadcrumbs from a source to the nearest sink. When the waves stop, cells forget, and the path vanishes. (Subtext: waves rest after passing, so they only go outwards; a breadcrumb is kept unless the wave now comes clearly shorter another way; ends are pulsed; cells update at random times; distances carry noise.)

```bash
python paths.py --watch --sources "8,45 24,45 40,45" --sinks "12,2 36,2"   # s / k add an end at the mouse (or remove one there), r restarts
python paths.py --picture net.png --sources "8,45 24,45 40,45" --sinks "12,2 36,2"
```

**Done so far:**
- **A first rule, by flow** (a chemical drained at the sink, conductance growing with flow; git tag `exp/paths-flow`): a single thin path, repaired locally when an end moves, but in about 600,000 ticks.
- **Distances by the minimum of the neighbours** (in git history before the wave rule): fast, but stale values circulated after an end went ("count to infinity"), and every fix cost stability elsewhere.
- **The wave rule (breadcrumbs):** every source joins its nearest sink in every test scene, seed and tick size, with noise and random update times; paths are as short as straight lines; when a sink moves, the path re-forms on the new shortest route; when sinks go, paths vanish. Works on 96×96 too.

**Persistent, quiet wiring** (on by default): a sink calls only while hungry (no pulse has reached it lately), and a path lives as long as its sink thanks it (an acknowledgement passed back up the path for each pulse that arrives). Once wired, the sheet falls silent and the paths stay; damage, moved or removed ends make sinks hungry or sources lonely, and the network re-wires, then falls quiet again.

**Crossing and learning (experiments in `LESSONS.md`, "Growing wiring"):** crossed wiring for approach works with one set of variables per thruster (all 12 inputs of a robot-like ring reach their opposite thruster), or with no crossing at all if thrusters pull towards their own side (one channel). One bit was learned on a grown path: pulses along it, a trace at each input, and a bad taste closing the gate of a recently active input; the poison colour stopped reaching the thruster after its first visit, the other carried on.

**Next:** quiet networks that stay in place once formed (plasticity: today a path exists only while waves keep coming); every source reaching every sink rather than only the nearest (a network with junctions); then carry pulses along the paths and put learning at the junctions.

**Then, the search for colour learning resumes** with what this teaches, either as a cell family with grown wiring and its few knobs evolved, or with the random search below.

**After that (some or all):**
1. **Search from random rules in a target-rich world, screening out the hopeless cheaply.** MAP-Elites on discriminate, all six variables free, with many blocks of each colour (`World.blocks`, e.g. 40), so any rule that keeps moving tastes often. In front of each full evaluation, a short life (about 500 time units, 2 worlds, dt 1): a candidate whose thrusters have stopped firing can't taste, so can't learn, and is rejected at about 1/200 of the cost. To build: the screen and `--blocks` for `mapelites.py`.
2. **If movers don't learn, the dish** (an alternative, sketched): a bare 12×12 sheet with two inputs and an output laid out like one sensor group, visits to either input in random order each ending in a taste (good after one, bad after the other, swapped between lives), learning measured as the change in output response. About a tenth of a robot candidate's cost; learners would then be put in the robot.
3. **Thin the world** once something learns: fewer blocks, step by step (a curriculum on the world rather than on behaviour).
4. **Understand the circuits:** how much they vary across seeds, whether they settle, which variables form them, and whether signals travel along them from sensors to thrusters.
5. **Make rules describable:** prune terms one by one while the score holds; the coefficient penalty (`--sparsity`) is too weak to make rules sparse.
6. **Organism-wide memory:** spread what one side learned to the others.
7. **Other cells, in reserve:** gradient (Sobel) inputs; a small ReLU network instead of the polynomial; the fuller cell (each neighbour's variables as separate inputs).
8. **Side-quest, time:** the medium is continuous (no global clock). Two alternatives are worth trying later:
   - rewarding sheets that settle into a stable, low-activity oscillation emerging from the cells (allowed: a property of the medium, not of the tick);
   - a synchronous medium like a cellular automaton (score at dt 1 only), as in Life or Rule 110: cheaper, but less biological.
9. **Parked, in git history:** the fuller cell above (2,812 coefficients), a sheet that grows from one central cell along a square spiral, and an untested PyTorch version of the simulation for running many lives at once on a GPU. It's the commit "Parked: ..." just before "Revert ...": `git revert` the revert to bring it back. A GPU only pays once the simulation is batched across genomes too.

## Experiments

Each finished experimental setup is tagged in git (`exp/NAME`) and named in `LESSONS.md`, so it can be brought back exactly (`git checkout exp/NAME`, or `git worktree add ../NAME exp/NAME` to have it alongside). Setups still in use live side by side in the code, sharing `cell.py`.
- `exp/robot-curriculum`: the robot curriculum (move, approach, graze, then discriminate) on the 24×24 disc, with the six-variable cubic cell and complexification.

## Requirements

Python 3, NumPy, Numba, pygame, and cma (for CMA-ES), in `.venv/`.
