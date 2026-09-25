# neuron2025

`neuron2025` is an interactive Python sandbox for building and probing a 2D field of excitable cells. You "wire" the medium by enabling cells on a grid, inject energy into those cells continuously, and then strike individual locations to watch pulses ignite, propagate, sustain, die out, and interact.

The core project is the sandbox itself: a place to explore local excitation, depletion, coupling, and pattern behaviour in an excitable medium. Logic-gate-style structures are the first bundled demo use-case rather than the whole scope of the project. The saved examples in this repository include `or`, `xor`, `inhibit`, and an accelerating path, all aimed at testing whether useful pulse-based logic can emerge from those local dynamics.

## What The Simulator Models

Each enabled cell has:

- an energy store that replenishes over time,
- a flame value representing how strongly it is currently burning,
- an illumination value produced by Gaussian coupling from nearby flames.

On each simulation step:

1. illumination is computed from neighbouring flame intensity,
2. dark enabled cells ignite if illumination crosses a strike threshold,
3. weak or exhausted flames extinguish,
4. burning cells relax toward the energy available to them,
5. flames consume energy,
6. enabled cells refill with fresh energy.

The implementation uses elapsed real time rather than fixed frame steps, and the flame response is written in a time-invariant form so the dynamics are less dependent on frame rate. The workbook `timeinvariance.xlsx` contains the derivation notes behind that choice.

## Requirements

- Python 3
- `pygame`
- `numpy`
- `scipy`
- `ffmpeg` on your `PATH` if you want video recording

Install them however you prefer, for example:

```bash
pip install pygame numpy scipy
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

## Code Layout

The simulator is now split into a small set of modules rather than one large file:

- `neuron.py`: thin compatibility entrypoint
- `main.py`: startup
- `app.py`: top-level application loop and coordination
- `model.py`: simulation state and parameters
- `views.py`: grid, chart, console, and help overlay rendering
- `actions.py`: keybinding registry and generated help text
- `persistence.py`: JSON snapshot loading and saving
- `recording.py`: video export through `ffmpeg`
- `headless.py`: run a pattern without pygame — load, step, strike, trace probes, grid-wide summaries. Used for scripted analysis/training and for fast exploratory sweeps across many hypothetical parameter values, as distinct from the control channel below (see "Debugging and Exploration Workflow").
- `patterns/`: saved simulator snapshots and bundled example layouts

## Interface Layout

The window has three regions:

- left: the excitable cell grid,
- top-right: a live chart/oscilloscope for probed cells,
- bottom-right: a small text console for save/load commands and variable inspection.

The grid view can show the combined state or isolate one field:

- default composite view: enabled cells in red, energy in blue, flame in green,
- `E`: energy only, grayscale,
- `I`: illumination only, grayscale.

## Basic Interaction

### Mouse

- Left click on a cell: toggle whether the cell is enabled. Enabling fills its energy to full; disabling zeroes it.
- Left-drag: paint more cells with the same enabled/disabled state.
- Right click on a cell: strike it if it has enough energy, and trigger the chart timebase.
- Click a panel: move keyboard focus between grid, chart, and console.

### Global Keys

- `Tab`: cycle focus between grid, chart, and console.
- `Esc` or `Ctrl+C`: quit.
- `h` or `?`: toggle the help overlay.
- `u` or `U`: undo the last edit. Undo depth is unlimited.
- `v`: start or stop MP4 recording of the whole window into `videos/`.

### Keyboard While Grid Has Focus

- `p`: pause or resume.
- `Space`: advance one simulation step while paused.
- `c`: clear the enabled pattern and zero activity.
- `f`: fill the whole grid with enabled cells.
- `z`: reset activity without changing the enabled pattern — flame off, energy full wherever enabled. Useful before a training run so no residual activity history (e.g. from a loaded snapshot) leaks into a fresh trial.
- `r`: randomly strike roughly one tenth of the cells.
- `a`: add a chart probe at the cell under the mouse.
- `d`: delete a chart probe at the cell under the mouse.
- `n`: name the chart probe at the cell under the mouse (focuses the console with a `name` command pre-filled; type the name and press Enter).
- `Shift` + arrow keys: shift the whole pattern and its probes.
- `Ctrl` + `Shift` + arrow keys: split at the cursor by inserting a blank row or column and moving only the cells on that side of the cursor.
- `=`: zoom in by reducing grid size.
- `-`: zoom out by increasing grid size.
- `e`: toggle energy-only view.
- `i`: toggle illumination-only view.
- arrow keys: select and adjust model parameters.
  - `Up` / `Down`: choose which parameter is selected,
  - `Left` / `Right`: scale the selected parameter down or up by 5%.

### Keyboard While Chart Has Focus

- `=`: zoom in on time.
- `-`: zoom out on time.
- `t`: toggle retrigger mode when the chart reaches the right edge.

### Console Commands

```text
help
?
ls
save NAME
load NAME
name INDEX NAME
strike X Y | strike LABEL
set VAR VALUE
run SECONDS
stats
probes
inspect X Y
params
quit
clear
enable X Y
disable X Y
settle
probe X Y
deleteprobe X Y
```

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

The repository includes several saved layouts:

- `patterns/oneway.json`: a one-way ("diode") gate — a pulse crosses `a → b` but not `b → a`.
- `patterns/and.json`: an AND gate — `out`/`edge` fire only when `a` and `b` are struck together; either alone does nothing, and nothing flows back from the output.
- `patterns/or.json`: an OR gate — two diodes feeding a shared relay, with an output leg to the grid edge. Either input fires the output; neither input can fire the other.
- `patterns/xor.json`: an XOR gate — `out`/`edge` fire when exactly one of `a`, `b` is struck; nothing fires when both are struck within ~1.75 s of each other. See "Building XOR" below.
- `patterns/inhibit.json`: an inhibitory structure, described in the original notes as the closest pulse-based analogue of NOT.
- `patterns/accel.json`: an accelerating or amplifying path experiment.
- `patterns/osc.json`: an oscillator example.
- `patterns/recent.json`: the most recently saved working state. This file is intentionally ignored by git.

Each save stores:

- the enabled-cell mask,
- current energy values,
- current flame values,
- any chart probes,
- the current global parameter values.

That means saves are full simulator snapshots, not just static patterns.

## Tunable Parameters

The live parameter set includes:

- `COUPLING_DIST`: spatial spread of illumination,
- `COUPLING_GAIN`: strength of coupling between flames and neighbouring cells,
- `FLAME_CONSUME`: how quickly flame depletes energy,
- `FLAME_INERTIA`: how quickly flame follows available energy,
- `MIN_FLAME`: sustaining threshold,
- `MIN_STRIKE`: ignition threshold,
- `STRIKE_LEVEL`: initial flame level when a cell ignites,
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

`SimulationConfig.stuck_on_risk()` checks this, and the app warns whenever it's true after a `load` or a parameter change. Fix it by raising `MIN_FLAME` or lowering `SUPPLY/S`, not by raising `FLAME_CONSUME` (which also lowers peak flame and erodes propagation) — and check rule 5 below when raising `MIN_FLAME`.

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
5. **No re-entry at junctions:** automatic ignition has no energy check, so a cell that has just gone out relights if its still-burning neighbours' combined tails reach θ. Those tails are near `MIN_FLAME`, so with `n` simultaneously-burning orthogonal neighbours you need `θ > n · MIN_FLAME` — n = 3 at any fork or T-junction — plus margin. Violating this turns junctions into oscillators. Count diagonal neighbours at weight r and any nearby parallel line (rule 6) toward n.
6. **Keep unrelated lines ≥ 3 apart.** A fully lit line two cells away delivers about 25% of an orthogonal neighbour's illumination (`(k2/k1)(1 + 2·k1/k0)` with the 1-D kernel weights). That is enough to make a parallel line catch fire almost all at once, and to tip nearby junctions over rule 5.

**The diode.** Narrow coupling means a cell can't reach two cells away, but an orthogonal neighbour can ignite a cell while a single diagonal one can't (rule 2). So the input line forks and wraps around the target, touching it only diagonally from two sides. Forward, the two diagonal cells burn together and ignite the target; backward, the burning target reaches each fork cell through one diagonal only, which is too weak:

```
. x x . . . .
x x . T x x x      input line on the left, T = target, output on the right
. x x . . . .
```

**AND is the same geometry** with the two diagonal feeders coming from separate inputs, so it is one-way by construction. **OR** is two diodes feeding a shared relay, with the output branching from the relay's middle.

Current shared parameters: `COUPLING_DIST=0.85 COUPLING_GAIN=6 FLAME_CONSUME=2 FLAME_INERTIA=1 MIN_FLAME=0.09 MIN_STRIKE=0.23 STRIKE_LEVEL=0.18 SUPPLY/S=0.14`, giving θ ≈ 0.35 ≈ 0.7 · f_peak ≈ 3.9 · `MIN_FLAME`, hops of ≈ 0.16 s, and a refractory time of ≈ 7.6 s. Every gate still works with `COUPLING_GAIN` changed by ±20%, and at 30–100 fps.

**Cells per refractory period** (refractory time ÷ hop time, ≈ 48 here) is the number that sets how long a delay line must be to hold one signal back until another has recovered. It is dimensionless, so uniform time scaling can't change it; only pulse *shape* and σ can. Narrow σ helps because with wide coupling, cells several steps back keep pushing the front forward. Rules 4 and 5 pull against each other (low `MIN_FLAME` versus fast recovery), which is what limits it.

### How the Constraints Interact

Every rule above is a statement about where θ sits relative to the pulse shape, so they compete for the same range:

- **Slow propagation versus margins.** A slow hop means θ sits high on flame's rising flank (rule 1). That squeezes the line's own margin, and it makes hop time very sensitive to anything that shifts θ: a ±20% change in gain moves hop time from 0.29 s to 0.12 s. Centring θ in the diode window (rule 2) is the compromise.
- **Hysteresis versus recovery.** Rule 5 wants `MIN_FLAME` well below θ, but the stuck-on limit (rule 4) caps `SUPPLY/S` at `MIN_FLAME · FLAME_CONSUME`, so a low `MIN_FLAME` means slow recovery and long refractory times. Together these limit how few cells a delay line can have (≈ 48 per refractory period here).
- **`STRIKE_LEVEL` does two jobs.** Keeping it ≥ 2 · `MIN_FLAME` (rule 3) gives a clean ignition, and keeping it below θ means a depleted cell that gets relit (it only ever reaches `STRIKE_LEVEL` before fizzling) can never ignite its neighbours. Those harmless fizzles show up wherever a diode target is hit twice in quick succession.
- **Timing windows mix gain-sensitive and gain-insensitive quantities.** Path delays scale with hop time, which is very gain-sensitive. The coincidence window (≈ 2–2.5 s, set by how long two flames overlap) and the refractory time (≈ 7.6 s) are set by pulse shape and barely move. Any circuit that races two paths must put its delay near the geometric centre of the window it needs, because the hop time can vary by a factor of about 2.4 while that window only spans about 2.8×.
- **Equal path lengths into a coincidence junction.** At low gain the two-diagonal margin is thin, so an AND only fires if its inputs arrive together. Give both inputs the same path length.

### Building XOR

XOR is not monotonic, so it needs inhibition. The only inhibition this medium has is refractoriness: a region that has just burned can't carry a wave. The gate is:

- an **OR** of the inputs, whose output travels a long route (the delay line) to a **split** into two strands four cells apart;
- a **coincidence junction** (the AND/diode geometry) where the two strands meet diagonally, feeding the output line;
- an **AND** of the inputs (the veto), whose output joins strand 1 close to the junction.

One input: the OR signal arrives on both strands together and fires the junction. Both inputs: the veto burns strand 1 first, and its backward wave annihilates the OR signal head-on. The junction then sees two single strands more than a coincidence window apart, so it never fires. This only works if the veto leads the OR signal by more than the coincidence window (≈ 2.5 s) and less than the refractory time (≈ 7 s). The delay route is sized to put it at ≈ 4 s, and the gate works with gain changed by ±20%. Inputs more than ≈ 2 s apart count as separate events, and the output fires once.

**Topology.** The connections A→AND, A→OR, B→AND, B→OR, AND→junction and OR→junction, plus A, B and the output all reaching the grid edge, form K3,3, which can't be drawn in a plane. With one layer of cells, some terminal of an AND/OR-based XOR must be enclosed. In `patterns/xor.json` it is `b`, and it has a stub to make its path length match `a`'s. Composing XOR freely with other gates will eventually need a crossover.

**All gate patterns share exactly the same `vars`**, so any of them can be wired together on one grid. Design new gates at the existing parameters rather than tuning bespoke values, and check them the same way: every truth-table case, every cell igniting exactly once per pulse, and nothing still lit afterwards.

### Debugging and Exploration Workflow

Two complementary ways to work with the simulator programmatically:

- **The control channel** (`control_in.txt` / `control_out.log`, see above) drives the *live, on-screen* session — use it for anything you want a human watching to be able to see happen, and for the final verification of any change.
- **`headless.py`**, run directly in throwaway scripts, drives an offline `State`/`SimulationConfig` with no pygame dependency — use it for fast, disposable sweeps across many hypothetical geometries or parameter values where nobody needs to watch each one, before settling on a design to build and verify live.

## Project Status

This repository is currently a compact multi-file simulator plus a handful of saved experiments. It does not include packaging, tests, or a formal file format spec; `neuron.py` is a thin entrypoint and the module files in the project root are the reference implementation.
