# Coordinates from two shared numbers (prototypes)

Side project: can a square sheet form a coordinate frame if each cell shares only two numbers and sees only its side
neighbours' averages of them (no geometry, no compass, noise in every update)? Numpy prototypes, kept as they were
when the work stopped; the findings are in `LESSONS.md`, "Coordinates from two shared numbers". The brief's own
method (diffuse, remember, then anchor) is `coords.py` at the top level.

- `flags/`: the working solution. Corners raise flags (u = 10) at random moments and label themselves from the order
  of the flags and which rise looked smallest (the diagonal); then corners hold their coordinates, edges straighten,
  and everyone else averages. `python whole.py 16 0.02 20` (sizes, settle tolerance, wait factor): at 16×16 with
  noise 1e-2, 7 of 8 runs reach exact coordinates (8×8: 4 of 8; overlapping flags are the main open failure). `flags2.py` alone tests the corners' labels.
- `growth/`: two copies at two speeds (u fast, v slow, v grows v - u): grows the large pattern from noise, robust to
  noise of 1e-2 when the growth rate is right; the cells finding that rate themselves works only partly
  (`e2e.py`: 24×24 15 of 16, 16×16 poor).
