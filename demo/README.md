# A brain that builds itself from averages, then learns

Like the "lizard" of Growing Neural Cellular Automata, this is a sheet of identical cells, each born the same bar
random noise, each running one local rule. Each cell sees nothing of its neighbours but the **average** of each
number they share, as in an excitable medium. Unlike the lizard, the sheet doesn't grow a picture. It grows a nervous
system: it names its organs, wires its sensors to its thrusters, and then the creature it drives learns, within its
own lifetime, which of two colours is food.

```bash
python creature.py                 # watch a life (r: new life, space: pause, up/down: speed, Esc: quit)
python creature.py --measure 16    # 16 lives with learning and 16 without, headless (a few minutes)
python test_rules.py               # every rule keeps the golden rules
python watch.py organs             # one stage on its own (organs, wiring; --show picks the numbers drawn)
python organs_test.py 49           # organs: when the names settle, life by life
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
  `take()`. `test_rules.py` checks every rule's imports, signature and birth, and that no body code reaches inside.
- Numba-compiled with the rule inlined: about 13 million cell-updates a second on one core.

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

1. **Organs.** The four corners know they are corners: they have three neighbours. They elect an origin: the smallest
   random tag wins. The origin's two edge neighbours pick a direction. Edge cells count steps round the edge both ways,
   so each knows its place round the perimeter. Four edge cells find themselves at an organ's place: R and B (6 apart, the
   middle of one side), A (the middle of the opposite side), V (the middle of a side between them). A cell takes the name only once it
   has been at the place for 80 updates in a row: at first every corner thinks it won, and the losers' counts take a
   while to rise away, giving places that drift along the edge. So nothing is wired until the organs stand still. Edge cells also know their side, so
   two sides count rows and columns in: every cell gets x and y (with 8 neighbours, steps from a straight edge are
   exactly the row). Nothing uses them yet; the live view shows them as a faint checkerboard.
2. **Wiring.** Every cell keeps its distance in steps from A, and from V. R and B are on a path to each. The end
   of a path (a tip) asks its neighbours one step nearer for offers, each a random whole number; it shows the smallest
   offer it sees; that cell joins. So a path grows one cell at a time, downhill, until it reaches A (or V). A path cell
   with no path neighbour one step farther leaves, so a path cut off from its source fades. A pulse runs down a path,
   from a cell to its neighbour one step nearer, and A (or V) counts the pulses it takes: the body fires that thruster.
3. **Choosing.** When a sensor kicks R or B, that cell sends a pulse towards A (forward, at what it sees) with chance
   b, its balance; otherwise towards V (back off and turn). Born b = 0.9, so a new-born creature is drawn to anything it
   sees. V also fires now and then by itself: a tumble, so it searches.
4. **Learning.**
   - **The trace.** A kicked cell keeps how many updates since its kick.
   - **News.** Every cell keeps the news of each sensor: how old the freshest kick of R it has heard of is, and of B.
     A cell's news never gets fresher unless there was a new kick, whatever order the cells update in. So when R's news
     of B gets fresher than R's own kick, B has been kicked since, and R forgets its own kick (and likewise B). Of R
     and B, only the more recently kicked keeps a trace. The news can be watched spreading, as a red glow.
   - **A meal** kicks the eaten colour's cell once (its flavour), and the creature chews with its eyes shut, so the
     news has time to cross.
   - **The taste** then moves b by taste × exp(-trace / 30), towards 0 or 1: a three-factor rule (just active, a meal,
     its taste). A bad taste also makes R and B send towards V for a while: the creature recoils.

No stage waits a fixed time: each starts when the one before has produced what it needs.

## Result

The table shows poison's share of meals in each third of life after the creature starts moving: 32 lives of 8,000
ticks, half with red as poison and half with blue.

|  | first third | second | last | balance at the end |
|---|---|---|---|---|
| learning on | 38% | 23% | 33% | poison 0.08, food 0.90 |
| learning off | 51% | 50% | 49% | both 0.90 |

The balances come out right; the behaviour is at the body's limit. With balances fixed perfect from birth, poison is
still about a quarter of meals (31%, 17%, 28%; 16 lives). Mostly that's because both colours are in view at once: food
drives the creature forward into the poison. It also acts on old news: a pulse takes about 55 ticks to cross the
brain.

## The numbers a cell shares

20, all seen only as averages over the neighbours:
- where am I: 8 (the election, the origin flag, two for handedness, two ring counts, x and y);
- wiring: 5 for each of the two channels (distance, path, tip, pulse, offer);
- learning: 2 (news of R, news of B).

## Not yet

- It is tuned for a 49 × 49 sheet only (the organs also work at 97).
- Two paths sometimes branch (a tie of offers, or a cell joining as its tip moves on); the branches carry pulses too.
- Election ties (two corners with the same tag, about 2% of lives) would make two origins.
- The starvation trap: a creature that learns to avoid everything stops eating.
