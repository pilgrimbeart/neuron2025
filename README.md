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

- Left click on a cell: toggle whether the cell is enabled.
- Left-drag: paint more cells with the same enabled/disabled state.
- Right click on a cell: strike it if it has enough energy, and trigger the chart timebase.
- Click a panel: move keyboard focus between grid, chart, and console.

### Global Keys

- `Tab`: cycle focus between grid, chart, and console.
- `Esc` or `Ctrl+C`: quit.
- `h` or `?`: toggle the help overlay.
- `v`: start or stop MP4 recording of the whole window into `videos/`.

### Keyboard While Grid Has Focus

- `p`: pause or resume.
- `Space`: advance one simulation step while paused.
- `c`: clear the enabled pattern and zero activity.
- `f`: fill the whole grid with enabled cells.
- `z`: zero energy and flame without changing the enabled pattern.
- `r`: randomly strike roughly one tenth of the cells.
- `a`: add a chart probe at the cell under the mouse.
- `d`: delete a chart probe at the cell under the mouse.
- `Shift` + arrow keys: shift the whole pattern and its probes.
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

The console is intentionally minimal:

```text
help
?
ls
save NAME
load NAME
```

This works with `patterns/*.json`, so `load xor` reads `patterns/xor.json`.
When the console has focus, normal typing stays local to the console; `Tab` and `Esc` still work globally.

## Bundled Example States

The repository includes several saved layouts:

- `patterns/or.json`: an OR-style arrangement.
- `patterns/xor.json`: an XOR-style arrangement.
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

## Project Status

This repository is currently a compact multi-file simulator plus a handful of saved experiments. It does not include packaging, tests, or a formal file format spec; `neuron.py` is a thin entrypoint and the module files in the project root are the reference implementation.
