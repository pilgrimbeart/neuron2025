# Lessons

Alternatives we considered and why we settled where we did, so we don't re-litigate them.

## One dynamic variable per cell is possible, but only as a phase

- A single *level* per cell (like energy or flame alone) can't make a pulse: with no input it just drifts to rest, so it can't rise, fall, and then be refractory. That's why the current model needs two (flame fires, energy recovers).
- A single *phase* on a circle can do it: at rest until light reaches threshold, then it advances at a fixed rate back round to rest. Brightness is a fixed "light curve" of phase; only a resting cell can ignite.
- Everything demonstrated so far survives (it depends only on light geometry and refractoriness). The stuck-on trap, the fuel rule, re-entry and fizzles disappear, and there are about five parameters instead of eight.
- Lost: the fuel metaphor (recovery becomes a clock) and graded partial recovery.
- The light curve needs a finite rise, otherwise propagation speed depends on frame rate. Its rise sets hop time, its peak relative to the threshold sets the diode and AND margins, its bright length sets the coincidence window, and the dark rest of the cycle is the refractory period.
- Independent of this: `COUPLING_GAIN` is redundant, because it only ever acts through the ratio `MIN_STRIKE / COUPLING_GAIN`.
- A per-cell phase is not a clock. It is a one-shot timer started by that cell's own ignition; there is nothing global or synchronised.

## No global clock

- Core thesis: local behaviour only, no global god. A global clock is exactly that god, so it is ruled out.
- With a clock, everything we've built would carry over (Wireworld, a synchronous cellular automaton, has diodes, logic, crossings and a computer). Our gates depend only on threshold excitation, refractoriness and local coupling, not on continuous time.
- A clock would remove our races and hazards (veto-lead windows, coincidence windows, crossing dead time and misroutes), because "at the same time" becomes "on the same tick". We accept those as the price of asynchronous, local timing.
- Continuous asynchronous time also keeps graded timing differences available to learning rules.

## Learning: where the change lives, and what triggers it

- A cell can tell three things locally: *was I involved* (light arriving, recent firing, low energy), *was it good* (nothing yet: needs a teaching signal), and *how much to change* (a new slow per-cell value).
- Teaching signal as a second kind of light (a "modulator"), rather than a special input on the ordinary light: it never excites, it only switches plasticity on. So a teacher can't itself leak through as a pulse. A modulator is just another cell's signal on a second channel.
- Eligibility by light or by energy are mirror images. Receiver learns to listen: light reaching me now, a near-miss, lowers my threshold. Sender learns to shout: my energy is low (I fired recently), raises my brightness. Energy can't mark a blocked junction (it never fires), but tolerates a teacher arriving seconds late.
- "Was I used?" is available for free: light from others *after* I ignite. Measured on a line: used cells get about twice the neighbour light over their burn of a dead-end cell, from about one hop after ignition. It says "used", not "useful", and correlation can fake it.
- Illumination counts other cells only. A cell's own light never affected ignition (only dark cells ignite), but it swamped any rule or measurement reading a burning cell's light. Light from others before I ignite is the *cause*; after, the *echo*.

## Learning experiments (T1 single junction, T2 dual input; learn.py)

- **Reciprocity.** With light and teaching signal sharing a reach, a transducer close enough to teach junction J is lit by J just as strongly. A diagonal transducer got ignited by J's own firing, so plain use of A re-taught A and switching to B never happened. Put the transducer two cells from J (J's light there ≈ 0.18 × threshold) and raise the learning rate to compensate.
- **Only "ready" cells should learn:** dark, with at least `STRIKE_LEVEL` energy. A weight only matters at ignition. Letting every cell learn strengthened the teacher's own line and shifted its timing. Dark-only still punished the transducer's feeder in the moment it was exhausted but the transducer was still burning, cutting off the teacher's supply.
- **Constants in light units are gain-sensitive.** A fixed offset or darkness cut-off shifts balance when all light scales by ±20%. Scale-free alternative: compare light with the teaching signal itself, `dw = M(η·I − κ·M)`. Taught junction I/M ≈ 4.4, untaught ≈ 0.55, so κ/η = 1.5.
- **With ready-only learning, a junction stops learning the instant it fires,** so its first success has zero margin. Margin grows with further pairings (each fires a little earlier on the rising light), so train for a fixed number of pairings, not until first success.
- **Learning speed scales with gain²** (light and teaching signal each scale with gain): −20% gain learns about 2.5× slower.
- Letting burning cells keep strengthening (to build margin) strengthened the teacher's own line too, so L's pulse arrived earlier and fell out of step with A.

## What is designed and what is learned (as of learn1/learn2)

- Designed by hand: junction geometry (input ends diagonally beside the junction, ≈ 0.7 × threshold), transducer placement (two cells from the junction), matched path lengths so input and teacher coincide, and a teacher that fires at exactly the right moment.
- Learned: only whether each designed junction conducts.
- So the designer still does most of the work. Next steps remove pieces of it: a teacher that arrives *after* the action (needs an eligibility trace), then junctions nobody designed, then teaching with no teacher at all.

## Reward timing: traces, not order

- Anticipating a reward and receiving it are the same signal (dopamine reward-prediction error: after learning, the burst moves to the earliest reliable predictor). So the rule shouldn't care whether the teacher comes before or after the activity, only whether they're close in time.
- Each cell keeps a fading trace of the brightest recent light and teaching signal it received (peak-hold, fading with `TRACE_TIME`); learning uses the traces. A near miss now leaves a wake, which energy alone can't (a cell that doesn't fire never uses energy).
- Traces broke the teacher again: the transducer's feeder recovers enough to ignite (`STRIKE_LEVEL`) about 0.8 s after burning out, while the teaching trace still lingers. Requiring full energy ("at rest") fixed that, but then a junction that fired on the input was refractory when a late teacher arrived, and couldn't be credited. Splitting eligibility fixed both: every cell strengthens for light it received; only cells at rest weaken.
- `TRACE_TIME` sets how close counts as "together". 2 s gives a robust window from 0.5 s early to 1.5 s late. 4 s widened it, but the trace was still about 17% strong when the feeder fully recovered, breaking learn1.
- Noise will matter (exploration: letting near-miss paths occasionally fire), but not yet: it makes training non-deterministic. Noise on received light would be targeted (only near misses fire), and would need its own timescale to stay frame-rate independent.
- Possible later: make ignition itself integrate light (like a membrane potential), so the trace and the firing mechanism are one; or hold a spread of activation levels per cell. Both change every gate's dynamics, so parked.
