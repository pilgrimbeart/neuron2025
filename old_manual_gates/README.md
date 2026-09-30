# neuron2025: the hand-designed system

This is the earlier, hand-designed version of the project, superseded by the evolved cell rule (see `../README.md`). It still runs from this folder, using the project's `.venv`: `cd old_manual_gates && ../.venv/bin/python neuron.py`.

`neuron2025` is an interactive Python sandbox for building and probing a 2D field of excitable cells. You "wire" the medium by enabling cells on a grid, inject energy into those cells continuously, and then strike individual locations to watch pulses ignite, propagate, sustain, die out, and interact.

The aim is **emergent learning**: a medium whose own local dynamics, given some reward signal, reshape it so that useful pulse pathways form and persist, with local rules only (no global signal, no outside hand beyond the training inputs and outputs). The cell model is deliberately tiny (all of it is in `physics.py`). The bundled patterns (`line`, `oneway`, `and`, `or`, `xor`, `cross`, `osc`, `inhibit`) show that one shared parameter set, derived from first principles, supports propagation, one-way gates, logic, crossings, oscillators and inhibition; `twophase.py` shows a random sheet learning, by reward alone, which of two inputs should drive an output.

## What The Simulator Models

Each cell of the grid is empty, **normal**, or a teacher: **good** or **bad**. A cell has:

- an energy store that replenishes over time,
- a flame value representing how strongly it is currently burning,
- a weight: how readily it responds to light (1 = normal). This is what learning changes,
- a fire trace: 1 when it ignites, then fading. Its neighbours can read it. There is no clock: two traces fade at the same rate, so comparing them says only which of two cells fired first and how far apart, never how long ago, so nothing depends on how big the sheet is,
- attention: whether it is currently being credited (or blamed) as a cause of something a teacher responded to, and passing that back, with its valence: reward or punishment,
- heat: how much more readily than usual it fires by chance while learning (its temperature above a base).

It receives **illumination** from other normal cells' flames nearby, through a Gaussian coupling. A teacher cell gives no light and is never lit; it only burns when struck, and striking it pays attention to whatever just fired beside it: a **good** one as a reward, a **bad** one as a punishment. Outside the cells there are only strikes (inputs, and the teachers) and one input, `teaching`, which switches learning on.

On each simulation step (this is the whole of `physics.py`):

1. illumination is computed from other normal cells' flames nearby (a cell's own flame doesn't count),
2. dark cells with at least `STRIKE_LEVEL` energy ignite if weight × illumination crosses a strike threshold; while teaching, they may also ignite by chance, more readily the nearer they are to that threshold and the hotter they are,
3. weak flames extinguish,
4. burning cells relax toward the energy available to them,
5. flames consume energy, and a flame whose fuel runs out goes out,
6. enabled cells refill with fresh energy,
7. a cell that ignited sets its fire trace to 1; all traces fade,
8. attention passes one hop back, from effects to causes, by comparing fire traces (see "Learning" below),
9. while teaching: cells credited as causes of a reward strengthen (of a punishment, weaken), cells that fired beside a rewarded route without causing it (backflow) weaken, every firing costs a little, and near misses that go unrewarded warm up.

Manual strikes follow the same fuel rule: a cell needs at least `STRIKE_LEVEL` energy to be struck.

The implementation uses elapsed real time rather than fixed frame steps, and the flame response is written in a time-invariant form so the dynamics are less dependent on frame rate. The workbook `timeinvariance.xlsx` contains the derivation notes behind that choice.

## Requirements

- Python 3
- `pygame`
- `numpy`
- `numba` (compiles the per-cell rules in `physics.py`; the first run in a process takes a few seconds)
- `ffmpeg` on your `PATH` if you want video recording

Install them however you prefer, for example:

```bash
pip install pygame numpy numba
```

## Running

Launch the simulator with:

```bash
python neuron.py
```

It opens fullscreen on the first detected display. You can optionally pass a display index:

```bash
python neuron.py 1
```

On startup the program tries to load `patterns/recent.json`, and on exit it saves the current session back to that file.

To run console commands at startup (after that load), pass them with `-c`, as many as you like:

```bash
python neuron.py -c "twophase 3" -c "speed 20"
```

## Code Layout

The simulator is now split into a small set of modules rather than one large file:

- `physics.py`: how a cell behaves — the update rule, the strike rule and the shared default parameters. Everything else is UI and support.
- `neuron.py`: startup (optional argument: display index)
- `app.py`: top-level application loop and coordination
- `model.py`: the grid of cells and its editing operations (shift, resize, paste), and the parameter container
- `views.py`: grid, chart, console, and help overlay rendering
- `actions.py`: console command and key-binding registries, and generated help text
- `persistence.py`: JSON snapshot loading and saving
- `recording.py`: video export through `ffmpeg`
- `headless.py`: load and run patterns without pygame, for scripted experiments and the regression suite
- `gates.py`: generates every bundled pattern's layout from code (`python gates.py` rewrites them)
- `verify.py`: the regression suite (`python verify.py`)
- `twophase.py`: the learning experiment: a random sheet taught by reward alone (`python twophase.py`, or `twophase SEED` in the app)
- `robot.py`: a round robot run by a sheet, in a world of red and blue blocks, learning which colour is food (`python robot.py`, or `robot SEED` in the app)
- `patterns/`: saved simulator snapshots and bundled example layouts
- `LESSONS.md`: alternatives we considered and why we settled where we did, in brief, so settled questions aren't reopened

## Interface Layout

The window has four quarters:

- top left: the excitable cell grid,
- top right: the robot's world, while `robot` runs (the arena, its blocks with food ringed in white, and the robot with its sensors as ticks and its actuators as dots that flash yellow as they fire),
- bottom left: a live chart/oscilloscope for probed cells,
- bottom right: a small text console for save/load commands and variable inspection.

The grid view can show the combined state or isolate one field:

- default view: red = energy, green = flame (×2), blue = attended (attention passing back through it). An enabled cell at rest is bright red and an empty one black; a cell just ignited looks yellow and turns green as its energy drains; a recovering cell glows dim red. Cells never drop below 10% grey, and teacher cells (good and bad) always carry an extra 40% blue,
- `e`: energy only, grayscale,
- `i`: illumination (light from other cells) only, grayscale, with the ignition threshold at mid-grey,
- `w`: weight in blue: neutral (1.0) is half blue, the maximum (2.0) full blue. This is where learning shows. While `twophase` runs, each cell's temperature shows in green (`T_MAX` = full green).

Chart traces use the same colours: energy red, flame green, illumination white.

## Basic Interaction

### Mouse

- Left click on a cell: toggle it between empty and normal. A new cell starts with full energy and neutral weight.
- Left-drag: paint more cells with the same enabled/disabled state.
- Ctrl+click on a cell: toggle it between a good teacher and empty (`good X Y` / `disable X Y`).
- Right click on a cell: strike it if it has enough energy, and trigger the chart timebase.
- Click a panel: move keyboard focus between grid, chart, and console.

### Global Keys

- `Tab`: cycle focus between grid, chart, and console.
- `Esc` or `Ctrl+C`: quit (`quit`).
- `h` or `?`: toggle the help overlay.
- `u` or `U`: undo the last edit (`undo`). Undo depth is unlimited.
- `v`: start or stop MP4 recording of the whole window into `videos/`.

### Keyboard While Grid Has Focus

- `p`: pause or resume (`pause` / `resume`).
- `Space`: advance one simulation step while paused.
- `c`: clear the enabled pattern and zero activity (`clear`).
- `f`: fill the whole grid with enabled cells (`fill`).
- `z`: reset activity without changing the enabled pattern — flame off, energy full wherever enabled (`settle`). Useful before a trial so no residual activity (e.g. from a loaded snapshot) leaks in. Weights are left alone.
- `Shift+Z`: forget all learning, resetting every weight to 1.0 and all heat to 0 (`unlearn`).
- `r`: randomly strike roughly one tenth of the cells.
- `a`: add a chart probe at the cell under the mouse (`probe X Y`).
- `d`: delete a chart probe at the cell under the mouse (`deleteprobe X Y`).
- `n`: name the chart probe at the cell under the mouse (focuses the console with a `name` command pre-filled; type the name and press Enter).
- `Shift` + arrow keys: shift the whole pattern and its probes (`shift DX DY`).
- `Ctrl` + `Shift` + arrow keys: split at the cursor by inserting a blank row or column and moving only the cells on that side of the cursor.
- `=`: zoom in by halving the grid size, keeping the pattern centred (`grid SIZE`).
- `-`: zoom out by doubling the grid size, keeping the pattern centred (`grid SIZE`).
- `e`: toggle energy-only view.
- `i`: toggle illumination-only view.
- `w`: toggle weight-only view.
- arrow keys: select and adjust model parameters.
  - `Up` / `Down`: choose which parameter is selected,
  - `Left` / `Right`: scale the selected parameter down or up by 5% (`set VAR VALUE`).

### Keyboard While Chart Has Focus

- `=`: zoom in on time.
- `-`: zoom out on time.
- `t`: toggle retrigger mode when the chart reaches the right edge.

### Console Commands

```text
help                 List console commands
ls                   List saved snapshots
save NAME            Save the current snapshot
load NAME            Load a snapshot
name INDEX NAME...   Name a probe (shown on grid instead of its index)
strike LABEL         Strike the cell under a labelled probe
strike X Y           Strike a cell
set VAR VALUE        Set a parameter
run SECONDS          Pause, advance by SECONDS immediately, stay paused
pause                Pause the simulation
resume               Resume the simulation
stats                Print grid-wide activity summary
probes               Print each probe's live energy/flame/illumination
inspect X Y          Print one cell's kind, energy, flame, weight, illumination, trace, attention and heat
params               Print the current parameter values
clear                Clear the enabled pattern and zero activity
fill                 Enable every cell
enable X Y           Make a normal cell
good X Y             Make a GOOD teacher cell: gives no light, never lit; striking it rewards what just fired beside it
bad X Y              Make a BAD teacher cell: gives no light, never lit; striking it punishes what just fired beside it
disable X Y          Remove a cell
settle               Flame off, energy full where enabled
unlearn              Reset every cell's weight to 1.0 and its heat to 0 (forget all learning)
probe X Y            Add a chart probe
deleteprobe X Y      Delete the chart probe at a cell
twophase SEED        Run the learning experiment live: adaptation grows routes, then reward teaches a, then b
twophase stop        Stop the learning experiment
robot SEED           Run the robot, with red as food: a sheet in a round body learning which colour is food
robot SEED FOOD      Run the robot with FOOD (red or blue) as food
robot stop           Stop the robot
speed N              Run N simulation steps per frame (1..50)
grid SIZE            Resize the grid to SIZE x SIZE, keeping the pattern centred
shift DX DY          Shift the whole pattern and its probes
undo                 Undo the last change
quit                 Quit the simulator
```

Commands are defined once, in `actions.COMMANDS`, which also generates `help` and the help overlay. Keys that do the same thing as a command (shown in brackets above) submit that command, so it is echoed as `> command` exactly as if typed, and appears in `control_out.log`.

This works with `patterns/*.json`, so `load xor` reads `patterns/xor.json`.
When the console has focus, normal typing stays local to the console; `Tab` and `Esc` still work globally.

### External Control Channel

While the simulator is running, an external process (e.g. an AI agent) can
drive and inspect the *same live session* the console commands above act on:

- Append console command lines to `control_in.txt` (repo root, gitignored).
  Each frame the app reads any new lines, runs them through the same command
  handler as the on-screen console, echoes `[control] <command>` so you can
  see what ran, then clears the file.
- All console output (from typed commands, control-channel commands, and
  ordinary `print()`s) is mirrored to `control_out.log` (also gitignored),
  regardless of how the process's own stdout/stderr are connected.

This is the same command language either way — typing in the console and
appending to `control_in.txt` are two entry points to one dispatcher, so
you can watch an external agent's actions happen live on screen.

## Bundled Example States

The bundled patterns are generated by `gates.py`, all use the shared parameters, and are all checked by `verify.py`. Each has labelled probes for its terminals, so `strike LABEL` works on them.

- `patterns/line.json`: the simplest case — a straight line from `in` to `out`.
- `patterns/oneway.json`: a one-way ("diode") gate — a pulse crosses `a → b` but not `b → a`.
- `patterns/and.json`: an AND gate — `out`/`edge` fire only when `a` and `b` are struck together; either alone does nothing, and nothing flows back from the output.
- `patterns/or.json`: an OR gate — two diodes feeding a shared relay, with an output leg to the grid edge. Either input fires the output; neither input can fire the other.
- `patterns/xor.json`: an XOR gate — `out`/`edge` fire when exactly one of `a`, `b` is struck; nothing fires when both are struck within ~0.75 s of each other. All three terminals are on the grid edge. See "Building XOR" below.
- `patterns/cross.json` (48×48): a crossing — a pulse from `a` (left edge) leaves only at `a_out` (right edge), and a pulse from `b` (top edge) leaves only at `b_out` (bottom edge). Simultaneous pulses leave at both; otherwise keep them ≈ 10 s apart. See "Crossing Two Pulse Streams" below.
- `patterns/osc.json`: a ring oscillator — strike `start` once and a pulse circulates forever, emitting at `out` about every 10 s. The ring contains a diode so the pulse can only go one way, and is long enough (60 cells) that it always returns to recovered cells. `start` joins just after the diode, so the half of the launch pulse that heads backwards dies at once.
- `patterns/inhibit.json`: an inhibitor — a pulse at `in` reaches `out` unless a pulse arrives at `inh` at the same time. `inh` is a modulation input: on its own it never produces output, and never leaves through `in`. It is one inhibit unit from XOR, with `inh` as the veto.
- `patterns/recent.json`: the most recently saved working state. This file is intentionally ignored by git.

Each save stores:

- each cell's kind (empty, normal or teacher),
- current energy, flame and weight values,
- any chart probes,
- the current global parameter values.

That means saves are full simulator snapshots, not just static patterns.

## Tunable Parameters

The live parameter set includes:

- `COUPLING_DIST`: spatial spread of illumination,
- `COUPLING_GAIN`: strength of coupling between flames and neighbouring cells,
- `FLAME_CONSUME`: how quickly flame depletes energy,
- `FLAME_INERTIA`: how quickly flame follows available energy,
- `MIN_FLAME`: sustaining threshold (a weaker flame goes out),
- `MIN_STRIKE`: ignition threshold,
- `STRIKE_LEVEL`: initial flame level when a cell ignites, and the energy a cell needs before it can ignite,
- `SUPPLY/S`: energy replenishment rate.

The intended workflow is exploratory: draw a structure, strike it, watch the chart, and tune parameters until the behaviour becomes useful or surprising.

Parameters fall into three different kinds, worth telling apart when tuning:

- **Thresholds/amplitudes** (`MIN_STRIKE`, `STRIKE_LEVEL`, `MIN_FLAME`): fixed values a live quantity is compared against or reset to.
- **Spatial parameters** (`COUPLING_DIST`, `COUPLING_GAIN`): how far and how strongly a burning cell's illumination reaches.
- **Rates and time constants** (`FLAME_INERTIA`, `FLAME_CONSUME`, `SUPPLY/S`): how fast things happen in real time. Multiplying `FLAME_INERTIA` by `k` and dividing `FLAME_CONSUME` and `SUPPLY/S` by `k` preserves every peak and threshold crossing and just makes everything take `k` times as long. Note this cannot change *how many cells* a pulse travels per recovery period — see below.

### The Stuck-On Trap

A burning cell has a reachable equilibrium `flame_eq = (SUPPLY/S) / FLAME_CONSUME`. If that is at or above `MIN_FLAME`, a cell that settles there burns forever. The condition is:

```
SUPPLY/S >= MIN_FLAME * FLAME_CONSUME   -->   at risk of a cell burning forever
```

`physics.stuck_on_risk()` checks this, and the app warns whenever it's true after a `load` or a parameter change. Fix it by raising `MIN_FLAME` or lowering `SUPPLY/S`, not by raising `FLAME_CONSUME` (which also lowers peak flame and erodes propagation).

### Designing Gates From First Principles

The bundled gates and the default parameters are derived, not searched for. Five quantities matter:

- **θ, the ignition threshold in flame units:** `θ = MIN_STRIKE / (COUPLING_GAIN · w_orth)`, where `w_orth` is the Gaussian weight of an orthogonal neighbour. A cell next to one burning neighbour ignites when that neighbour's flame reaches θ. Gain and `MIN_STRIKE` only ever act through this ratio.
- **f_peak:** the peak flame of a burning cell, set by the pulse shape (`FLAME_INERTIA`, `FLAME_CONSUME`, `STRIKE_LEVEL`, `MIN_FLAME`).
- **r = w_diag / w_orth = exp(−1 / 2σ²)**, with σ = `COUPLING_DIST`.
- **Hop time** ≈ `FLAME_INERTIA · ln((1 − STRIKE_LEVEL) / (1 − θ))`, slightly shortened by the pull from the cell two back.
- **Refractory time** ≈ burn duration + energy recovery, which scales with `1 / SUPPLY/S`.

The rules:

1. **Lines propagate:** θ < f_peak.
2. **Diode/AND window:** one diagonal neighbour must *not* ignite a cell, two must: `r · f_peak < θ < 2r · f_peak`. The window is widest at σ ≈ 0.85 (r = 0.5); put θ near its geometric centre, ≈ 0.7 · f_peak, for about ±40% margin.
3. **Hysteresis:** `STRIKE_LEVEL ≥ 2 · MIN_FLAME`, and `STRIKE_LEVEL < θ`.
4. **No stuck-on:** `SUPPLY/S` comfortably below `MIN_FLAME · FLAME_CONSUME`.
5. **No re-entry at junctions.** A cell that has just burned out must not relight while its neighbours are still burning, or junctions become oscillators. The fuel rule guarantees this: a burnt-out cell has no energy and can't ignite again until it has recovered `STRIKE_LEVEL`, which takes `STRIKE_LEVEL / SUPPLY/S` (≈ 0.8 s, several hops). By then its neighbours have burned out too. (Before the fuel rule, re-entry forced `θ > 3 · MIN_FLAME` at every T-junction, which held `MIN_FLAME` and hence recovery speed down.)
6. **Keep unrelated lines ≥ 3 apart.** A fully lit line two cells away delivers about 25% of an orthogonal neighbour's illumination (`(k2/k1)(1 + 2·k1/k0)` with the 1-D kernel weights). That is enough to make a parallel line catch fire almost all at once.

**The diode.** Narrow coupling means a cell can't reach two cells away, but an orthogonal neighbour can ignite a cell while a single diagonal one can't (rule 2). So the input line forks and wraps around the target, touching it only diagonally from two sides. Forward, the two diagonal cells burn together and ignite the target; backward, the burning target reaches each fork cell through one diagonal only, which is too weak:

```
. x x . . . .
x x . T x x x      input line on the left, T = target, output on the right
. x x . . . .
```

**AND is the same geometry** with the two diagonal feeders coming from separate inputs, so it is one-way by construction. **OR** is two diodes feeding a shared relay, with the output branching from the relay's middle.

Current shared parameters: `COUPLING_DIST=0.85 COUPLING_GAIN=6 FLAME_CONSUME=3 FLAME_INERTIA=1 MIN_FLAME=0.06 MIN_STRIKE=0.2 STRIKE_LEVEL=0.12 SUPPLY/S=0.144`, giving θ ≈ 0.30 ≈ 0.7 · f_peak, hops of ≈ 0.18 s, a burn of ≈ 1.1 s (a wake of ≈ 6 cells), and a refractory time of ≈ 5.9 s. Every gate still works with `COUPLING_GAIN` changed by ±20%, and at 30–100 fps.

**The wake.** A cell burns until its fuel runs out, so the burn is set by how fast flame consumes energy (`FLAME_CONSUME` against `FLAME_INERTIA`), not by `MIN_FLAME`. Before the fuel rule, an exhausted cell kept "burning" on inertia alone, decaying from ≈ 0.34 down to `MIN_FLAME` over about τ · ln(0.34 / `MIN_FLAME`). With the τ = 1 s that slow propagation needs, that fuel-less tail was half of every burn and tripled the wake (18 cells).

**Cells per refractory period** (refractory time ÷ hop time, ≈ 33 here) is the number that sets how long a delay line must be to hold one signal back until another has recovered. It is dimensionless, so uniform time scaling can't change it; only pulse *shape* and σ can. Narrow σ helps because with wide coupling, cells several steps back keep pushing the front forward. Rule 4 limits it: recovery speed `SUPPLY/S` is capped at `MIN_FLAME · FLAME_CONSUME`, and `MIN_FLAME` is capped at `STRIKE_LEVEL / 2` by rule 3.

### How the Constraints Interact

Every rule above is a statement about where θ sits relative to the pulse shape, so they compete for the same range:

- **Slow propagation versus margins.** A slow hop means θ sits high on flame's rising flank (rule 1). That squeezes the line's own margin, and it makes hop time very sensitive to anything that shifts θ: a ±20% change in gain moves hop time from 0.29 s to 0.12 s. Centring θ in the diode window (rule 2) is the compromise.
- **Hysteresis versus recovery.** The stuck-on limit (rule 4) caps `SUPPLY/S` at `MIN_FLAME · FLAME_CONSUME`, and rule 3 caps `MIN_FLAME` at `STRIKE_LEVEL / 2`. A low `STRIKE_LEVEL` gives slow hops (good) but also slow recovery, so delay lines stay at ≈ 33 cells per refractory period.
- **`STRIKE_LEVEL` does three jobs.** It is where each flame starts (so it sets hop time with θ), the fuel a cell needs before it can ignite (so it sets how long a burnt-out cell stays refractory), and, through rule 3, the ceiling on `MIN_FLAME` and so on recovery speed. Keeping it below θ also means a partly recovered cell that relights (its flame starts at `STRIKE_LEVEL` and it soon runs out of fuel) can't ignite its neighbours. Those harmless fizzles show up wherever a diode target is hit twice in quick succession.
- **Timing windows mix gain-sensitive and gain-insensitive quantities.** Path delays scale with hop time, which is very gain-sensitive. The coincidence window (≈ 1.3 s, set by how long two flames overlap) and the refractory time (≈ 5.9 s) are set by pulse shape and barely move. Any circuit that races two paths must put its delay near the geometric centre of the window it needs, because the hop time can vary by a factor of about 2.4 across ±20% gain.
- **Equal path lengths into a coincidence junction.** At low gain the two-diagonal margin is thin, so an AND only fires if its inputs arrive together. Give both inputs the same path length.

### Building XOR

XOR is not monotonic, so it needs inhibition. The only inhibition this medium has is refractoriness: a region that has just burned can't carry a wave. `patterns/xor.json` has the same structure as the classic four-NAND XOR, `out = (a inhibited by AND) OR (b inhibited by AND)`:

- Each input feeds an **AND** along the bottom edge, and separately takes a serpentine **delay line** up to its own **long-armed diode**. The two diodes feed the **OR** relay, and the output leaves from the relay's middle to the top edge.
- The AND's output (the veto) runs up the middle and **fans out** to each diode's inner arm, close to the diode target. Each branch passes through a small one-way diode, so that one input's pulse can't run back up the veto line and out of the other input.

One input: its pulse reaches both diode arms together and fires the diode, and the OR carries it out. Both inputs: the veto burns each diode's inner arm first. Its backward wave annihilates the input's pulse head-on, and the diode target only ever sees single arms, far enough apart in time that it never fires. Two timing conditions must hold, and both scale with hop time, which is the gain-sensitive quantity:

- The veto must lead the input's pulse at the entry point by more than the coincidence window and less than the refractory time. Anything from ≈ 8 to over 26 hops passes the checks; the delay lines set ≈ 14 hops (≈ 2.5 s), near the middle of that range.
- The veto's forward wave (inner arm) and its backward wave (round the split and up the outer arm) must reach the diode more than a coincidence window apart. So the veto enters close to the diode target, and the outer arm is 2 cells longer than the inner. It can't be much longer, because at low gain a single input's two arms must still arrive together.

The gate works with `COUPLING_GAIN` changed by ±20% and at 30–100 fps. Inputs up to ≈ 0.75 s apart count as "both"; from ≈ 1 s apart the output fires once.

**Topology.** `a`, `b` and the output all sit on the grid edge. That depends on the architecture: vetoing the *merged* OR signal (connections A→AND, A→OR, B→AND, B→OR, AND→junction, OR→junction, plus all three terminals on the boundary) forms K3,3, which can't be drawn in a plane. Vetoing each input separately, before the merge, has the planar four-NAND graph: going round the outside, the order is a, a's diode, OR/out, b's diode, b, AND.

### Crossing Two Pulse Streams

**No single-layer crossing can be free of interference.** With short-range coupling, a pulse only moves between orthogonal neighbours, or jumps one empty cell through a diagonal pair (and that gap cell is enclosed by the pulse's own cells). So a's route from the left edge to the right edge is an unbroken barrier, and b's route from top to bottom must share cells with it or run orthogonally alongside it. Running alongside means a's pulse ignites b's line. Sharing cells means the shared cells are refractory for a while after either pulse. Wider coupling doesn't help, because anything that throws enough light across a's line lights a's line at least as strongly. Every crossing therefore has a dead time of about one refractory period.

**`patterns/cross.json` routes correctly outside that dead time, and for simultaneous pulses.**

- **Routing.** a and b enter an L-shaped relay at its two ends (each through a diode). The merged pulse reaches two inhibit units, each a long-armed diode like the ones in XOR. The unit leading right to `a_out` is vetoed by b; the one leading down to `b_out` is vetoed by a. b's veto runs through the top-right quarter of the grid and a's through the bottom-left, so nothing crosses and all four terminals sit on the edges. A small 8-cell bump on each input, after its veto branch, gives the veto its ≈ 14-hop lead.
- **The "both" case.** Without help, simultaneous pulses veto each other and nothing comes out. So the leg to each inhibit unit attaches near the *opposite* input's end of the relay: the `b_out` leg near a's end, the `a_out` leg near b's end. A single pulse reaches its near leg at once and the far leg only after crossing the relay (≈ 16 hops, longer than a coincidence window at any gain in range). Simultaneous pulses reach both legs together. A spur from each leg ends diagonally beside a junction `both` (the AND geometry), which therefore fires only for simultaneous pulses. Its output feeds both output lines through one-way diodes, beyond the vetoed units.

It works with `COUPLING_GAIN` changed by ±20% and at 30–100 fps, with no pulse ever leaving through an input. With both inputs struck, measured by how far apart the pulses are:

| separation | result |
|---|---|
| ≤ 0.5 s (≤ 1 s if a is second) | both outputs fire, via `both` |
| 0.75–3 s | both pulses lost: outside the coincidence window, but each veto still catches the other's pulse |
| 4–8 s | the first is routed, the second is lost |
| ≈ 8.5–9 s | the second is **misrouted** to the first's output: its veto reaches an arm that is still recovering, relights only the entry cell, and the merged pulse gets through once that cell has recovered |
| ≥ 9.5 s | both routed correctly |

Routing each input across the relay to its output makes the dead time about 3 s longer than without the `both` detector. That cost is unavoidable in this design, because the relay crossing must outlast the coincidence window. A crossing that handles every separation correctly is impossible here (see above). Three XORs (c = a⊕b, then `a_out` = c⊕b and `b_out` = c⊕a) would also give simultaneous pulses, at roughly three times the size, and would still have a dead time.

**Requirements for every saved gate.** All gate patterns share exactly the same `vars`, so any of them can be wired together on one grid. Each must also keep working with a margin of error on those parameters: currently `COUPLING_GAIN` changed by ±20% and frame rates from 30 to 100 fps. Design new gates at the existing parameters rather than tuning bespoke values. If a parameter change is unavoidable, re-verify every gate against it.

### Learning

Learning is reward only: the network acts by itself, and a teacher can only say "yes, that" after the output has fired. It never makes anything fire. Everything it sets off happens cell to cell (steps 2 and 7–9 of `physics.py`):

- **Attention runs back from the reward to its causes.** Striking the teacher cell beside the output makes it attended. An attended cell reads its neighbours' fire traces, finds the one that fired first in the second before it (two cells out if no adjacent one did, as light reaches that far), and offers a level; a neighbour that fired, listening to the attended neighbour that fired soonest after it (the one it could have caused), accepts if its own trace is at or below that level. So attention travels back one hop per step along the route the pulse actually took, even through a flood, and a cell that fired after its neighbour (backflow) is never taken for a cause. After passing attention on, a cell is refractory for a while, so one reward sends one wave.
- **Credit and blame.** A cell that accepts attention gains weight. A cell that fired a moment after a credited neighbour and wasn't accepted is backflow: the valued route drove it, but it led nowhere; it loses a little weight. That closes a competing input's route exactly where it joins the rewarded one. Every firing costs a little, so unused activity fades.
- **Exploration.** While teaching, a cell can fire by chance, far more readily when it is nearly lit enough. Each cell has its own temperature: it warms when it nearly took part (lit, but didn't fire) and no attention came within reach, cools when attention does, and slowly settles back. So chance firing concentrates at the edge of activity that isn't paying off, such as a closed gate beside a stopped pulse.
- **Teaching on or off.** The `teaching` input switches chance firing and all learning on. With it off the network is deterministic and nothing changes.

`twophase.py` is the test: a random sheet with inputs `a` and `b` and output `o`. Adaptation (both inputs rewarded for reaching `o`) grows routes; then teaching `a` alone makes `a` pass and `b` not, and teaching `b` switches it. On 16 random 32×32 sheets, with every rule local: routes grow on all 16, teaching `a` blocks `b` on all 16, and the full test (learn `a`, then switch to `b`, each checked with teaching off) passes on 13. On 64×64 sheets (same rules; only the experimenter's timing scales) it passes on 3 of 8. AND (a bad teacher punishing single inputs) doesn't work yet. How we got here, including what failed, is in `LESSONS.md`.

`robot.py` is the final test's framework: a round robot whose brain is a disc of ordinary cells, in a walled arena of red and blue blocks. In each run one colour is food and the other poison, and the robot has to learn which within the run. Eight directional sensors on the rim strike a red or a blue stub of cells at a rate that rises with how much of that colour they see (sparse pulses: a cell takes seconds to refuel); eight actuators on the rim, each between its sensor's two stubs, push the robot towards their side each time they fire, so the fresh sheet has an innate approach to any colour. Eating strikes GOOD or BAD teacher cells beside every actuator. The current rules do not learn this yet: the robot eats both colours alike, and the firing cost wears the sheet down until it stops (`LESSONS.md`). `python robot.py SEEDS MINUTES` compares learning against a no-learning control; `robot SEED` or `robot SEED FOOD` runs it live.

### Regression Suite

**Run `python verify.py` every time `physics.py`, the shared parameters, or any pattern changes, and make it pass before committing.**

```bash
python verify.py            # every pattern, and learning
python verify.py xor cross  # just these
python verify.py learning   # just learning
```

It never waits for real time: the model always advances by an explicit time step, so the suite steps at a fixed frame rate as fast as the CPU allows (a few hundred times real time per core) and spreads the runs across all cores.

For each pattern, in order starting with `line`, `verify.py` checks that the saved file uses `physics.DEFAULT_PARAMS` and matches its layout and probes in `gates.py`. It then runs every truth-table case headlessly at the nominal parameters, with `COUPLING_GAIN` changed by −20% and +20%, and at 30 and 100 fps. Every expected output must fire the right number of times, every cell must ignite at most once per pulse, and nothing may still be burning at the end, except for `osc`, which must keep running. It prints PASS/FAIL per pattern and exits non-zero on any failure.

Learning is checked with `twophase.py`: on 8 fixed random sheets, at least 6 must grow routes, learn `a` (then `a` reaches `o` and `b` doesn't, with teaching off) and switch to `b`. That takes a few minutes and dominates the run; the pattern checks take seconds.

To add or change a pattern, edit its function in `gates.py`, add its cases to `verify.CASES`, run `python gates.py NAME`, run `python verify.py`, and then check it live through the control channel.

### Debugging and Exploration Workflow

Two complementary ways to work with the simulator programmatically:

- **The control channel** (`control_in.txt` / `control_out.log`, see above) drives the *live, on-screen* session — use it for anything you want a human watching to be able to see happen, and for the final verification of any change.
- **`headless.py`** (`load_pattern`, `simulate`) drives an offline `State` with no pygame dependency — use it in throwaway scripts for fast sweeps across many hypothetical geometries or parameter values where nobody needs to watch each one, before settling on a design.

## Project Status

A compact simulator, a set of verified patterns on one shared parameter set, a regression suite, and a first result in emergent learning with local rules only (`twophase.py`, "Learning" above). There is no packaging or formal file format spec; the module files in this folder are the reference implementation.
