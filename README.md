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

- `patterns/or.json`: a minimal OR gate — two `oneway` gates (see below) feeding a shared relay segment, with an output leg reaching to the grid edge. Either input fires the output; neither input can fire the other.
- `patterns/xor.json`: an XOR-style arrangement.
- `patterns/inhibit.json`: an inhibitory structure, described in the original notes as the closest pulse-based analogue of NOT.
- `patterns/accel.json`: an accelerating or amplifying path experiment.
- `patterns/osc.json`: an oscillator example.
- `patterns/oneway.json`: a one-way ("diode") gate — a pulse crosses `a → b` but not `b → a`. See "Building a One-Way Gate" below for how it works.
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

- **Thresholds/amplitudes** (`MIN_STRIKE`, `STRIKE_LEVEL`, `MIN_FLAME`): fixed values a live quantity is compared against or reset to. Changing these changes *what counts as enough*, not how fast things happen.
- **Spatial parameters** (`COUPLING_DIST`, `COUPLING_GAIN`): how far and how strongly a burning cell's illumination reaches. Widening these is what lets several cells jointly reach further than any one of them alone (see "Building a One-Way Gate" below).
- **Rates and time constants** (`FLAME_INERTIA`, `FLAME_CONSUME`, `SUPPLY/S`): govern how fast things happen in real time. `FLAME_INERTIA` is a time constant (bigger = slower response); `FLAME_CONSUME` and `SUPPLY/S` are per-second rates (bigger = faster). Because the whole model runs on elapsed real time rather than fixed frame steps, these three can be scaled together as a group to slow down or speed up a pattern's *pace* without changing anything about what it actually does: multiply `FLAME_INERTIA` by `k` and divide `FLAME_CONSUME` and `SUPPLY/S` by `k`, and every peak value and threshold crossing is preserved — the whole thing just takes `k` times as long in wall-clock time. Useful when a pattern's pulses happen faster than a human can follow.

### The Stuck-On Trap

A burning cell has a real, reachable equilibrium: `flame_eq = (SUPPLY/S) / FLAME_CONSUME`. If that value is at or above `MIN_FLAME`, a cell that settles there never crosses the extinguish threshold and burns forever — silently consuming energy and never re-arming to be struck again. The condition is:

```
SUPPLY/S >= MIN_FLAME * FLAME_CONSUME   -->   at risk of a cell burning forever
```

`SimulationConfig.stuck_on_risk()` checks this, and the app prints a warning whenever it's true after a `load` or a parameter change. The bundled defaults and patterns are tuned to stay clear of it.

If you do hit it, the fix is `MIN_FLAME`, not `FLAME_CONSUME`. Raising `FLAME_CONSUME` closes the trap but also lowers a cell's peak flame — which quietly erodes how far a pulse can propagate (single-hop illumination margin can drop to almost nothing) — a regression that's easy to miss until a specific pattern happens to need that margin. Raising `MIN_FLAME` instead only changes the extinguish threshold and leaves peak flame/propagation reach untouched.

### Building a One-Way Gate

Under default coupling, one cell reliably ignites an *adjacent* cell (illumination comfortably clears `MIN_STRIKE`), but cannot ignite a cell one gap further away (illumination falls off too fast to reach threshold) — even a wide fan of cells contributing at once isn't enough. Widening `COUPLING_DIST`/`COUPLING_GAIN` changes the balance: a single cell still can't jump a one-cell gap, but three cells in a row *can*, because their combined illumination clears the threshold at the far side even though any one of them alone falls short.

That asymmetry is a one-way gate ("diode"): a line widens into a 3-cell fan, crosses an empty gap cell, and narrows back to a single target cell. A pulse arriving at the fan crosses the gap; a pulse arriving at the lone target cell cannot gather enough cells to send anything back. `patterns/oneway.json` is exactly this.

**Gotcha — accidental fans.** Any 3 (or more) roughly-in-line cells near a gap can jump it, not just the intended fan — including cells you placed there for an unrelated reason. `patterns/or.json` (two `oneway` gates sharing an output) first broke this way: a target cell plus a 2-cell start of its onward output leg formed an unintentional 3-cell column at exactly gap-distance from the *other* gate's fan, and jumped that gap too, silently defeating the one-way property. Checking cell-by-cell distance to the nearest gap isn't enough — it's the *combined* illumination from every enabled cell within range that matters, and a handful of stray cells add up to the same effect as a deliberate fan.

The fix that worked: don't let the two gates' targets sit edge-to-edge. Give them a plain relay segment between the two targets, and branch any onward line from the *middle* of that segment — far enough from both fans (several cells of clearance, not one) that nothing there can contribute meaningfully to either gap-jump. `patterns/or.json` uses exactly this shape. Whatever the layout, verify isolation empirically (strike one side, confirm every cell on the *other* side — including any new pattern you add later — never lights up) rather than trusting that a geometry "looks" far enough apart.

### Debugging and Exploration Workflow

Two complementary ways to work with the simulator programmatically:

- **The control channel** (`control_in.txt` / `control_out.log`, see above) drives the *live, on-screen* session — use it for anything you want a human watching to be able to see happen, and for the final verification of any change.
- **`headless.py`**, run directly in throwaway scripts, drives an offline `State`/`SimulationConfig` with no pygame dependency — use it for fast, disposable sweeps across many hypothetical geometries or parameter values where nobody needs to watch each one, before settling on a design to build and verify live.

## Project Status

This repository is currently a compact multi-file simulator plus a handful of saved experiments. It does not include packaging, tests, or a formal file format spec; `neuron.py` is a thin entrypoint and the module files in the project root are the reference implementation.
