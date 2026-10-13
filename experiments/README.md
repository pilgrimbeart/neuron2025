# Prototypes

Side experiments, kept as they were when each stopped; what they found is in `LESSONS.md`. Most import the top-level
modules (`paths.py`, `cell.py`) as they were at the time, so a prototype may need the matching commit
(`git log -- paths.py`) to run unchanged.

- `coords/`: coordinates from two shared numbers (corner flags; two-copy growth). See `coords/README.md`.
- `paths/`: path experiments before and around the breadcrumb rule: persistence, scale, cuts and walls (`suite.py`),
  shared channels, and the first one-bit learning on a grown path (`learn.py`). `cross/`: crossing mechanisms on a
  robot-like ring (`lab.py`): a channel per target, a hub, addresses, rays.
- `seed_design/`: designing a seed rule for the evolved cell by hand on a bare 48×48 sheet (`seed.py`, `excite.py`,
  `channels.py`, `canal.py`), with the copies of `cell.py`, `robot.py`, `evolve.py`, `mapelites.py` and `app.py`
  it ran against.
