# A brain that builds itself, then learns

Like the "lizard" of Growing Neural Cellular Automata, this is a sheet of identical cells, each born the same bar
random noise, each running one local rule and showing its neighbours a few numbers. Unlike the lizard, the sheet
doesn't grow a picture. It grows a nervous system: it works out where it is, names its organs, wires its sensors to
its thrusters, and then the creature it drives learns, within its own lifetime, which of two colours is food.

```bash
python creature.py                 # watch a life (r: new life, space: pause, up/down: speed, Esc: quit)
python creature.py --measure 16    # 16 lives with learning and 16 without, headless (about 2 minutes)
python creature.py --check         # every cell is born the same, bar noise
```

Needs Python 3 with numpy, numba and pygame.

Note: these cells read each neighbour's shown numbers separately (the smallest, the one a breadcrumb names), which
breaks the project's first rule, that a cell sees only the average of its neighbours. `../../demo/` rebuilds the
creature under that rule.

## The files

- **`cell.py`: everything a cell is and does.** 54 numbers per cell. All neighbours can read of a cell is what it
  shows (`show`): 18 numbers, each neighbour's separately. Every tick every cell's timers run. Then each cell, with chance 0.9, updates from its
  own numbers and what its neighbours show; never all at once.
- **`creature.py`: the body, the world and the views.** The body touches the brain only at cells, by the names
  those cells gave themselves. It kicks the cell named R when its red sensor fires (B for blue) and delivers tastes
  to R and B. It fires a thruster for each pulse arriving at the cell named A (or V).

## What happens in a life

1. **Where am I.** The four corners know they are corners: they have three neighbours. They elect an origin: the
   smallest random tag wins. Whole-number counts in from each edge then give every cell its x and y, as fractions
   of the sheet. Cells name themselves organs by them:
   - R and B on the north edge, 6 cells apart;
   - A on the south edge;
   - V near the north-east corner.
2. **Wiring.** A and V call: each sends out waves that count their steps. Every cell remembers which neighbour the
   wave came from, as a breadcrumb. The trail of breadcrumbs from R and from B is a path. There is one channel per
   target, so the routes cross freely. Pulses run down the paths, each sink thanks every pulse back up its path,
   and a thanked path stays.
3. **Choosing.** When a sensor kicks R or B, that cell sends a pulse towards A (forward, at what it sees) with
   chance b, its balance. Otherwise the pulse goes towards V (back off and turn). Born b = 0.9, so a new-born
   creature is drawn to anything it sees.
4. **Learning.**
   - **The trace.** Every cell keeps how long it has been since it was kicked (its trace) and the news: the age of
     the freshest kick it has heard of. If a cell hears of a kick fresher than its own, it forgets its own. So of R
     and B, only the more recently kicked keeps a trace. The news can be watched spreading from one to the other.
   - **A meal.** A meal kicks the eaten colour's cell once (its flavour). The creature chews with its eyes shut, so
     the news has time to cross.
   - **The taste.** The taste then moves each input's b by taste × trace, bad tastes negative. This is a
     three-factor rule: just active, a meal, its taste. A bad taste also makes R and B send towards V for a while,
     so the creature visibly recoils.

No stage waits a fixed time: each starts when the one before has produced what it needs.

## Result

The table shows poison's share of meals in each third of life after the creature starts moving: 16 lives of 8,000
ticks, half with red as poison and half with blue.

|  | first third | second | last | balance at the end |
|---|---|---|---|---|
| learning on | 43% | 27% | 24% | poison 0.03, food 1.00 |
| learning off | 50% | 51% | 45% | both 0.90 |

Some poison is still eaten after learning, for two reasons:
- **Latency.** A pulse takes roughly 50 ticks to cross the brain, so the creature acts on what it saw a while ago.
- **Both colours in view.** When both are in sight at once, food drives it forward into the poison too.

The balances themselves come out right in every life.

## Where it came from

This directory is the standalone demo. The research code is in the repository's root:
- each capability (`frame.py`, `paths.py`, `learn.py`) proven on its own, at other sizes too;
- the findings, in `LESSONS.md`.
