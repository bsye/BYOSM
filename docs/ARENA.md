# Poker arena: watch, play and measure the bot

Goal: a six-seat poker table on the owner's domain where the bot BYOSM trained sits with other bots, the owner can
simulate hands or take a seat, every seat's result is measured, and only at the end the table says which seat was
which bot. It is a baseline to look at before any money is spent on a bigger teacher. Play money only.

## Decisions so far (2026-11-24)

| Question | Decision |
|---|---|
| Where it lives | A separate project for the site; BYOSM only exports the bot for it |
| Opponents | An open-source poker bot, next to the bot we trained |
| Who can open it | Behind Cloudflare Access at first: the student stays private |
| Order | Now, as the baseline before the rented-machine step of [the Pluribus push](GAMES.md) |

## Open-source opponents found

| Bot | Licence | What it is | Fit |
|---|---|---|---|
| [SHARINGAN-1](https://github.com/markokraemer/poker-sharingan) with [its weights](https://huggingface.co/markokraemer/sharingan-1) | Apache-2.0, code and weights | A Pluribus reproduction: six seats, 100 big blinds, 200 card buckets, trained on 64 cores and 576 GB; ships a self-hosted arena where an outside bot plays it over stdin and stdout with duplicate deals | The strongest opponent we can get, and the measure that matters before the rented machine: it is what that run would try to build. Too big to host: its checkpoints are 28.3, 124.7, 141.7 and 186.9 GB, and a Cloudflare container has at most 20 GB of disk. The project is a month old and unvetted |
| [G5](https://github.com/Nemandza82/g5-poker-bot) | MIT | Winner of the six-player no-limit event of the 2018 Annual Computer Poker Competition, by its own README; opponent modelling and tree search, C++ and C# | Small enough to host, but built for Windows and its interface is undocumented: an unknown amount of work |
| Always-call, random, a tight and a loose rule bot | ours | Simple fixed players | Trivial to host; our bot already wins about 2,900 mbb/hand against callers, so they show the table works, not how strong the bot is |

## Two pieces, in this order

1. **Local arena (the baseline that answers the money question).** Our student and our teacher against SHARINGAN-1
   in its own arena, inside a sealed container like our teacher's. If SHARINGAN-1 is clearly stronger, it is also a
   candidate teacher (its licence allows it), and the rented machine may not be needed at all.
2. **Hosted table (to watch and to play).** A Cloudflare Worker serves the table page; a Durable Object holds each
   table's state and the hidden seat list; a Cloudflare Container runs our engine and the 5 MB student, so the model
   never reaches the browser; D1 keeps each session's results.

## Hosted table: acceptance

| # | Story | Done when |
|---|---|---|
| 1 | Simulate | GIVEN a new table of six bots WHEN the owner asks for the next 100 hands THEN the hands play out, each seat's stack and winnings update, and no seat's identity is shown |
| 2 | Play along | GIVEN the owner takes a seat WHEN it is his turn THEN he sees his cards and the legal moves with their amounts, and the bots act in turn around him |
| 3 | Measure | GIVEN any number of hands THEN every seat shows hands played and mbb/hand with a 95% interval, and the page says when the interval is too wide to tell seats apart |
| 4 | Batch | GIVEN a request for 100,000 hands THEN they run without the animation and the same table of results comes back |
| 5 | Reveal | GIVEN the owner ends the session THEN the table names each seat's bot and the session is stored with its results |
| 6 | Private | GIVEN any request from the browser THEN no model weights, no hidden cards and no seat identities are in the response before the reveal |

## Edge cases

| Case | Handling |
|---|---|
| 100 hands | Shown as watching, not measuring: the interval is printed next to every number |
| The owner leaves mid-hand | His seat folds when its clock runs out; the session can be resumed |
| A bot errors or times out | The seat checks or folds, the event is logged, the session continues |
| All-in with side pots | Settled by the engine, the same code that scores our table tests |
| The container is asleep | The page says it is waking the bots; the first hand waits for it |
| Unequal stacks after a hand | Stacks reset to 100 big blinds every hand, the format every bot here was trained for |

## What the engine still needs

- Settling a finished hand (showdown, side pots, payouts) through the stdin interface that already describes a
  decision, and dealing from a seed so a session can be replayed.
- The student read as an ONNX file by the engine's container, so the container needs no PyTorch.
- `byosm poker export`: the student, its card abstraction and the engine image reference as one bundle.

## Cost

Workers Paid is $5 a month and includes 25 GiB-hours of container memory and 375 vCPU-minutes; beyond that a
container costs $0.0000025 per GiB-second of memory and $0.000020 per vCPU-second, and it sleeps when idle
([pricing](https://developers.cloudflare.com/containers/platform/pricing/)). A 1 GiB student container awake for an
hour is under one cent of memory.

## Not in scope

Real money, public sign-up, hosting a 28 GB or larger opponent on Cloudflare, mobile apps.

## Needs the owner

1. Download SHARINGAN-1's smallest checkpoint for the local arena: `B-local-2e10` (28.3 GB) plus its bucket tables
   (0.28 GB) from Hugging Face, and its code from GitHub, built and run only in a sealed container. The full
   124.7 GB checkpoint needs disk space this machine does not have free today.
2. Which opponent the hosted table starts with: the simple bots now and G5 later, or wait for G5.
3. The deploy itself: the owner runs the `wrangler` commands; nothing is deployed from here.
