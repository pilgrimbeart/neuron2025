"""Test bench for crossing: a robot-like ring on the 48x48 sheet, and a pulse test that every mechanism is judged by.

Layout: G sensor groups round a ring of radius R about the centre; each has a red and a blue input on the ring (either
side of the group's angle) and a thruster RIN cells in. Crossed wiring: each input should drive the thruster of the
opposite group (group + G/2).

A mechanism provides grow(seed) -> state (after its wiring has grown, if any) and step(state, pulses) -> (pulses, hits),
where pulses is a list of pulse objects it understands and hits a list of thruster indices reached this step. The
test sends single pulses from each input in turn and records where they arrive."""
import numpy as np
SIZE, C, R, RIN, G = 48, 23.5, 21.0, 3.0, 6

def at(angle, radius):
    return (int(round(C + radius * np.sin(angle))), int(round(C + radius * np.cos(angle))))

def layout(groups=G):
    angles = np.arange(groups) * 2 * np.pi / groups
    inputs = [at(a + side * 0.15, R) for a in angles for side in (-1, 1)]      # input i belongs to group i // 2
    thrusters = [at(a, R - RIN) for a in angles]
    return inputs, thrusters

def disc():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    return (yy - C) ** 2 + (xx - C) ** 2 <= (R + 0.6) ** 2

def pulse_test(mech, state, inputs, thrusters, groups=G, max_steps=200, repeats=3):
    """For each input, send `repeats` single pulses (one at a time) and record which thrusters they reach. Returns
    (correct fraction of arrivals, crosstalk fraction, lost fraction) over all pulses, and per-input detail."""
    correct = wrong = lost = 0
    detail = []
    for i, cell in enumerate(inputs):
        target = (i // 2 + groups // 2) % groups
        got = []
        for _ in range(repeats):
            pulses, reached = mech.inject(state, cell), set()
            for _ in range(max_steps):
                pulses, hits = mech.step(state, pulses)
                reached.update(hits)
                if not pulses:
                    break
            got.append(sorted(reached))
            if not reached:
                lost += 1
            elif reached == {target}:
                correct += 1
            else:
                wrong += 1
        detail.append((i, target, got))
    total = correct + wrong + lost
    return correct / total, wrong / total, lost / total, detail
