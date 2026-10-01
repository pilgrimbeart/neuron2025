# Lessons

What we tried with the evolved cell rule, what worked, what didn't and why, so settled questions aren't reopened. The lessons from the hand-designed system (excitable medium, gates, hand-made learning rules) are in `old_manual_gates/LESSONS.md`.

## Principles carried over from the hand-designed system

- **Local behaviour only, no global god.** No global clock, no broadcast signals, no harness switching the cells' behaviour. The only things from outside are the body's inputs: sensor pulses and taste.
- **No hidden scale.** Cells must not know the sheet's size or their position in it; only their own state and their neighbours'.
- **Inputs spike.** Sparse kicks, never continuous levels.
- **Random inputs, nothing crafted.** Learning must work with events arriving in random order, not a training sequence designed to make it work.
- **Excitable media percolate.** In a connected threshold medium, a pulse either dies or floods everything, and tuning the threshold only moves the boundary between the two. A flood fires every actuator at once, so their pushes cancel.
- **Reward needs variability.** Deterministic trials give the same outcome every time, so reward has nothing to select between.

## Evolving the cell rule (cell.py, evolve.py, mapelites.py)

- **The rule is a polynomial per cell variable, found by evolution;** the aim is still a rule that can be said in a sentence or two.
- **The body can give the answer away.** With each actuator between its own sensor's inputs and pulling towards its own side, "copy your neighbour" is already a line-follower: only the rim was used. Actuators are now thrusters (they push away from their side), so approach needs signals to cross the disc.
- **Discrete time invites tricks.** With whole-step updates, the evolved sheet flipped every cell between −1 and +1 each tick and steered by phase differences. A rate per variable (0.1) makes the rule an equation of motion: firing became a real excursion and recovery, and activity sparse (6% of cells above firing level while approaching).
- **Move and approach evolve in seconds to minutes with CMA-ES; learning didn't.** Choose and taste stayed on the "eat everything" plateau (50% food in both halves of lives), whether taste landed in the middle or at the point of contact, with or without the slow variable m.
  - Scoring the worse of a red-food and a blue-food life made standing still (0) beat eating everything, so poison costs half, as a stepping stone.
  - The approach rule sits on a narrow ridge: at a search spread of 0.1 its neighbours die and the search collapses to doing nothing. From scratch it collapses too.
  - CMA-ES needs no gradient, but it still moves only towards improvements it samples; on a plateau its rankings are decided by noise.
- **A wide search found learning.** MAP-Elites keeps the best rule for each kind of behaviour (meals per life, and how eating food and eating poison change during a life), with the two taste strengths as body genes. Within about 1.5 hours it found a rule that learns the taste task: on 128 fresh worlds, the one block is eaten 1.23 then 1.70 times per half-life when it is food, and 1.04 then 0.16 when it is poison. After a bad taste it keeps away from the block (about 21 units, against 13 when it is food), rather than freezing.
- **Fixed evaluation worlds let flukes in.** The first "learner" (+0.48 rise in food share on 16 worlds) scored 0 on fresh ones: it ate everything early and then stopped. Learning now only counts when both halves of lives have meals, and the best learner is re-checked on fresh worlds.
- **Where the learner keeps its memory: in w, in the taste cells.** After a bad taste, the taste cell that was kicked lowers its w and holds it for the rest of the life; nothing else differs between food and poison lives. So the memory is local, where the taste arrived, and a side the robot hasn't bitten from hasn't learned.
- **m wasn't used as memory or as a cell type.** It settles to about −0.47 in every cell within 200 ticks: a constant offset, which through the m·x terms shifts every coefficient alike. No differentiation into kinds of cell appeared; nothing asked for it.
- **A taste does leave a lasting difference in a sheet that hasn't learned, but it's chaotic** (a changed oscillation), not a usable memory.
- **The evolved rules depend on the tick size, so they aren't continuous-time systems.** With a time step dt (rates are per unit time, pulses a Poisson process), the approach rule and the learner work only at the dt they were evolved at (1). At dt 0.5 the sheet goes quiet (no thruster firing at all); at dt 2 approach eats 1.55 blocks instead of 2.6, and the learner barely eats. Their activity comes from overshoot at the coarse step, a subtler version of the flip-every-tick trick. Rules have to be evolved across tick sizes to rule this out.

## Speed

- **Cells as a list with neighbour lists, the rule applied one term at a time across all cells, no allocation per tick:** 3.6× faster in one process (17.5 → 5.0 ms per 2000-tick life), 2.4× on all 8 cores (185 → 448 lives per second). No swapping or major page faults in either version.
- **All 8 cores give only 2.7× one core** (540 against 203 lives per second): each life costs 4.9 ms of CPU alone and 13.8 ms with 8 running. The Core Ultra 7 266V has 4 performance and 4 low-power cores and slows its clock under full load. More workers still help, less each time (4 workers: 461 lives per second).
