_The brief this project started from, kept as written. (Since then: numpy and numba rather than JAX, and development in stages designed by hand alongside evolution; see the README and LESSONS.md.)_

# Emergent Neural Fabric

## Goal

Build a small experimental platform to explore whether **very simple local rules can produce useful, understandable neural-like computation and learning**.

The objective is not merely to evolve a controller that performs well. The objective is to discover **simple local rules from which useful global organisation emerges**.

A good result should ideally be understandable and explainable, not just performant.

---

## Core idea

A simple virtual creature moves in a 2D world containing objects with sensory properties and consequences.

Inside the creature is a homogeneous 2D cellular substrate.

Each cell:

* runs exactly the same local update rule;
* has a small vector of mutable state;
* communicates only with nearby cells;
* has no absolute position;
* may interact with mutable local edge/connection state.

The shared update rule is the thing being evolved.

During a creature's lifetime, the rule is fixed, but **cell and connection state can change freely**.

---

## What should emerge

Do not hard-code conventional neural-network architecture.

The substrate should make it possible for evolution to discover mechanisms such as:

* signal propagation;
* wiring / high-conductance pathways;
* crossings and routing;
* memory;
* modulation of one cell's behaviour by another;
* changing thresholds, gain or response speed;
* plastic connectivity;
* oscillation or persistent activity.

These should emerge from simple local dynamics rather than being implemented as named mechanisms.

Avoid permanently private per-cell parameters where possible. If something such as threshold, responsiveness or connection strength varies between cells, prefer representing it as mutable state that other cells can influence.

---

## Learning, not fixed behaviour

The creature must not be able to evolve fixed meanings for sensory cues.

For example, objects may have arbitrary sensory identities, while their **valence is randomly assigned at the beginning of each lifetime**.

Within one lifetime that mapping remains stable.

The creature therefore has to discover through experience:

```text
sensory pattern
    +
experienced reward/pain
    ↓
persistent internal change
    ↓
different future behaviour
```

The evolved local rule is therefore effectively a mechanism for **within-lifetime learning**.

---

## Timescales

Keep the distinction explicit:

1. **Evolutionary time** — the shared update rule changes between candidate organisms.
2. **Lifetime dynamics** — cell and edge state change while the rule remains fixed.
3. **Persistent facts within a lifetime** — e.g. an object's valence may remain unchanged for the whole lifetime.

Do not encode these lifetime facts directly into the controller.

---

## Environment and curriculum

Do not begin by requiring a sophisticated solution.

Use an **on-ramp of increasingly difficult environments** so that useful partial mechanisms can evolve before harder capabilities are required.

Start with very simple tasks and progressively introduce requirements such as:

* recognising recurring sensory patterns;
* learning arbitrary valence;
* retaining associations;
* navigating efficiently;
* handling more object types;
* context-dependent behaviour.

The curriculum itself should be easy to modify.

---

## Simplicity

Simplicity is a first-class objective.

Avoid large hidden neural networks inside cells.

Prefer:

* small state vectors;
* small shared update rules;
* recurrent local dynamics;
* few parameters;
* interpretable mechanisms;
* explicit complexity penalties or multi-objective optimisation.

A slightly worse but comprehensible solution may be much more valuable than a better opaque one.

---

## Tentative implementation

Use:

* Python;
* JAX for fast compiled/vectorised simulation;
* WSL2/Linux as the normal development environment;
* many worlds/candidate organisms evaluated in parallel;
* rendering separated from training.

Training should normally run headless and much faster than real time.

A debugging mode should allow individual runs to be visualised, including both:

* creature behaviour in the world;
* internal cell and connection state.

---

## Design principle

For every proposed feature ask:

> Are we providing primitive physics, or quietly providing part of the solution?

Prefer the smallest substrate that leaves useful mechanisms possible without prescribing what they should be.

---

# Ideas and Thoughts — feel free to ignore

* Start with a regular 2D lattice and nearest-neighbour communication.
* Do not expose absolute coordinates.
* Give each cell a small vector of state variables; different variables may naturally become independent communication channels, memories or modulators.
* Keep neighbour directions distinguishable initially rather than summing them together.
* Mutable edge state may provide a natural route to emergent wiring.
* Evolutionary optimisation may be more suitable than conventional reinforcement learning if rule simplicity or rule structure itself is being searched.
* Later experiments could include multiple creatures, competition, cooperation, ecological niches and evolutionary arms races.
* Long-range connections, graph rewiring and growth can be explored later rather than assumed initially.

