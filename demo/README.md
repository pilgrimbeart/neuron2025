# A brain that builds itself from averages, then learns

Like the "lizard" of Growing Neural Cellular Automata, this is a sheet of identical cells, each born the same bar
random noise, each running one local rule. Each cell sees nothing of its neighbours but the **average** of each
number they share, as in an excitable medium. Unlike the lizard, the sheet doesn't grow a picture. It grows a nervous
system: it names its organs, wires its sensors to its thrusters, and then the creature it drives learns, within its
own lifetime, which of two colours is food.

```bash
python creature.py                 # watch a life (r: new life, d: damage the brain at the mouse, space: pause,
                                   # up/down: speed, Esc: quit)
python creature.py --measure 16    # 16 lives with learning and 16 without, headless (a few minutes)
python test_rules.py               # every rule keeps the golden rules
python watch.py organs             # one stage on its own (organs, wiring; --show picks the numbers drawn)
python organs_test.py              # organs: when the coordinates and names are right, life by life
python wiring_test.py              # wiring: pulses sent from R and B, how many arrive and how soon
```

Needs Python 3 with numpy, numba and pygame.

## The rules can't be broken by accident

- **`sheet.py` is the only code that knows there is a grid.** A rule (`rules/*.py`) declares its numbers, which of
  them its neighbours can see, and what every cell is born with, and one compiled function
  `update(me, mean, count, inp, out)`. It is given its own numbers, its neighbours' **mean** of each shared number,
  how many neighbours it has, and what the body delivered: nothing else. Not its position, the size, a clock, or any
  neighbour on its own.
- **The engine does the rest.** Each tick each cell updates with some chance, never all at once. Every number gets a
  little noise at every update. Every cell is born from one birth vector, identical bar declared noise.
- **The body touches the brain only at cells, by the names they gave themselves:** `cells()`, `give()`, `get()`,
  `take()`. (Watching uses `view()`; the experimenter's `damage()` replaces cells with new-born ones.) `test_rules.py` checks every rule's imports, signature and birth, and that no body code reaches inside.
- **A rule never holds a view into the sheet.** The engine copies a cell's numbers and its neighbours' means into the
  rule's own small arrays, and `test_rules.py` runs every rule with array bounds checked: an index past the end can't
  reach another cell's numbers.
- Numba-compiled: about a million cell-updates a second on one core for the whole brain (about 375 ticks a second at
  49 × 49).

## Averages are enough

- **Any and all.** A cell that shares a 0/1 flag tells its neighbours "any of you" (mean > 0) or "all" (mean = 1).
- **The smallest.** A cell that shares exp(-5 d) (the engine does this for a shared ("d", 5)) lets its neighbours find
  the smallest d among them from the mean, to within log 8 / 5 < 1/2: rounded, it is exact. So cells can count steps
  and elect.
- **Which way.** A cell that shows 1, 9 or 81 by its distance from somewhere, mod 3, lets each neighbour count, from
  the sum, how many of them are one step nearer, level, or one step farther (three base-9 digits).
- **Whole numbers.** With noise in every number at every update, a cell can only remember whole numbers, re-rounded at
  each update. So tags, counts, timers and even the balance (in twentieths) are whole numbers.

## What happens in a life (`rules/brain.py`, on `rules/wiring.py`, on `rules/organs.py`)

1. **Where am I.**
   - **Election.** The four corners know they are corners: they have three neighbours. They elect an origin: the
     smallest random tag wins.
   - **Handedness.** The origin's two edge neighbours pick a direction round the edge (the one with the smaller of two
     more random tags goes first).
   - **Ring counts.** Edge cells count steps round the edge both ways, so each knows its place round the perimeter,
     and so its side.
   - **Coordinates.** One side counts rows in, the next columns: every cell gets x and y (with 8 neighbours, steps from
     a straight edge are exactly the row). The live view shows them as a faint checkerboard.
   - **Organs.** The first side is the tail, A, bar its corner, V. Two cells in the middle of the far side, the head,
     6 apart, name themselves R and B. A cell takes a name only once it has held it for 80 updates in a row: at first
     every corner thinks it won, and the losers' counts take a while to rise away, giving places that drift. So
     nothing is wired until the organs stand still (about tick 490).
2. **Wiring.** A cell's distance to the tail is its row, y; to the corner V, the larger of x and y. R and B are on a
   path to each. The end of a path (a tip) asks its neighbours one step nearer for offers, each a random whole number;
   it shows the smallest offer it sees, and that cell joins. So a path grows one cell at a time, downhill, until it
   reaches the tail (or V). The tip also shows its x and y mod 3, so each cell below knows where it sits relative
   to it, and a cell on the straight way (straight down to the tail; straight at the corner for V) offers from a
   lower band: so paths run straight, and R's and B's can be told apart. Three refinements, each traced from a fault:
   - if two cells share the smallest offer, the tip sees it (the mean comes out a little off a whole number) and asks
     again;
   - a tip keeps its choice for a few updates, so it can't change its mind while the chosen cell joins, and get two;
   - a cell that would also touch another path offers a bigger number, so it is chosen only if no other cell offers
     (otherwise the other path's cell would get a second way on, and its pulses would split in two).

   A path cell with no path neighbour one step farther leaves, so a path cut off from its source fades and its source
   grows a new one. A pulse runs down a path, from a cell to its neighbour one step nearer, and the tail (or V) counts
   the pulses it takes: the body fires that thruster.
3. **Choosing.** When a sensor kicks R or B, that cell sends a pulse towards the tail (forward, at what it sees) with
   chance b, its balance; otherwise towards V (back off and turn). Born b = 0.9, so a new-born creature is drawn to
   anything it sees. R and B also send towards V now and then by themselves: a tumble, so it searches.
4. **Learning, in R and B only.**
   - **The trace.** A kicked cell keeps how many updates since its kick.
   - **News.** Every cell keeps the news of each sensor: how old the freshest kick of R it has heard of is, and of B.
     A cell's news never gets fresher unless there was a new kick, whatever order the cells update in. So when R's news
     of B gets fresher than R's own kick, B has been kicked since, and R forgets its own kick (and likewise B). Of R
     and B, only the more recently kicked keeps a trace. The news can be watched spreading, as a red glow.
   - **A meal** kicks the eaten colour's cell once (its flavour), and the creature chews with its eyes shut, so the
     news has time to cross.
   - **The taste** (delivered to R and B) then moves b by taste × exp(-trace / 30), towards 0 or 1. So the change
     needs a recent input (the trace, sharpened by the news) and the taste; it doesn't depend on which way the cell
     sent its pulse. Strictly that is two factors, a stimulus trace and a reward, not the three of a three-factor rule
     (pre, post, reward): the creature learns what each colour is worth, and its choices follow from that.
     A bad taste also makes R and B send towards V for a while: the creature recoils.
   - All other cells only build and carry: frame, paths, pulses, news.

No stage waits a fixed time: each starts when the one before has produced what it needs.

## Result

The table shows poison's share of meals in each third of life after the creature starts moving: 32 lives of 8,000
ticks, half with red as poison and half with blue.

|  | first third | second | last | balance at the end |
|---|---|---|---|---|
| learning on | 45% | 34% | 34% | poison 0.12, food 0.93 |
| learning off | 51% | 50% | 47% | both 0.90 |

The balances come out right; the behaviour is at the body's limit. With balances fixed perfect from birth, poison is
still about a third of meals (37%, 29%, 30%; 16 lives). Mostly that's because both colours are in view at once: food
drives the creature forward into the poison. It also acts on old news: a pulse takes about 55 ticks to cross the
brain, to the tail or to V.

## Damage

Press d in the live view to replace a disc of cells under the mouse with new-born ones, as with the lizard. The new
cells relearn their coordinates from their neighbours within about 50 to 100 ticks; a path cut off from its source
fades and regrows (about 500 ticks for a band across the middle), and every pulse arrives again. Damage R or B and the
new cell takes the name again, but what the old one had learned is gone. Damage the corner that won the election and
the frame dies (6 lives of 8): every cell remembers the smallest tag it has heard, so the dead winner's tag lives on
with no corner to own it, and no corner wins again. Not yet handled.

## The numbers a cell shares

16, as many as the lizard's channels, all seen only as averages over the neighbours:
- where am I: 6 (the election, handedness, two ring counts, x and y);
- wiring: 4 for each of the two channels (path, tip, pulse, offer);
- learning: 2 (news of R, news of B).

## Not yet

- It is tuned for a 49 × 49 sheet only.
- Once in about 16 lives a fork still doubles one input's pulses.
- Election ties (two corners with the same tag, about 2% of lives) would make two origins.
- The starvation trap: a creature that learns to avoid everything stops eating.
