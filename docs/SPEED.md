# Speed plan

Measured on 2026-10-29 at 11:00, the machine at load 9 to 13 (two training jobs, plus about 300% CPU from other
projects' containers in OrbStack).

## What is slow

| Measure | Now |
|---|---|
| Specialists page, last API response | 31 s (production bundle), 28 s (Vite dev) |
| Main thread blocked on that page | 3.8 s production, 8.7 s dev (longest task 556 ms / 730 ms) |
| Requests fired by that page | 26 health, 6 identical `/api/specialists`, 1 `/api/next-steps` (19 s) |
| A light page opened right after (Settings) | `/api/config` 1.2 s, `/api/specialists` 9.6 s |
| One specialist's health plus next steps | 1.4 s of CPU |

## Why

1. **The server works through a queue.** Home and Specialists fire about 30 CPU-heavy requests at once. Python runs
   them one at a time under its interpreter lock, and leaving the page does not cancel them, so every page opened next
   waits behind them.
2. **Each health is mostly repeated work.**
   - Every board read decodes each test item's units, which are 95% of the stored bytes (915 KB of 965 KB on tool
     choice), and one health reads the board 11 to 22 times.
   - Every classifier first loads all forecaster reports with 318 published tables (0.31 s) to learn it is not a
     forecaster.
   - Next steps computes the health checks again.
3. **The browser re-renders everything on each response.** Each of the 26 health answers re-renders the whole page.

## Plan

| Phase | Change | Expected |
|---|---|---|
| 1 | Check one forecaster by name instead of loading all of them | -0.3 s per specialist |
| 1 | Board reads leave units in the database; only the paired comparison, overfit and test-set rows read them | board read 42 ms to about 8 ms |
| 1 | Health checks computed once per request and shared with next steps | about half the CPU per specialist |
| 2 | One shared `/api/specialists` read for the whole app | 6 requests to 1 |
| 2 | Health and next steps fetched a few at a time, dropped when the page is left | other pages stop waiting |
| 3 | Cards update one at a time instead of re-rendering the page | blocking under 1 s |

Each phase is measured before and after with the numbers above. Target: Specialists fully loaded under 5 s on a busy
machine, and any light page under 300 ms.

## Edge cases

- The near-miss vote runs from an older copy of the code and reads units from board reads. Stored data does not
  change, so it keeps working.
- A forecaster and a classifier with the same name: the forecaster lookup keeps its current precedence.
- Results without units: found while planning, `sure_lead` counts a missing paired comparison as a sure lead. That is
  right for a first model but fails open when both models have runs and the units are missing. Fixed in phase 1:
  with runs on both sides and no paired units, the lead is not sure.

## Results

Same machine, same load, production bundle.

| Measure | Before | After |
|---|---|---|
| Home: requests, all data loaded | 108, about 41 s | 51, 7.2 s |
| Home: main thread blocked, longest task | 6.5 s, 704 ms | 0.83 s, 241 ms |
| Specialists: all data loaded | 31 s | 6.4 s |
| Specialists: main thread blocked, longest task | 3.8 s, 556 ms | 0.26 s, 189 ms |
| Settings opened right after: `/api/config` | 1.2 s | 22 ms |
| One health request (tool choice) | about 0.7 s | 0.1 s |
| `/api/specialists` | 0.9 s | 0.66 s |

Specialists now asks for next steps per card, like Home, and each card re-renders only when its own data changes. Not
done yet: it finishes at 6.4 s, above the 5 s target, on a machine at load 9 to 13.
The Postgres form of the board read without units is written but not run here.

### Second pass

"All data loaded" above is the last API response, which can be the sidebar's memory poll that repeats every 15 s.
From here on it is the last of the page's own requests (the list, health and next steps). Machine at load 16 to 30:
a training job plus the ready loop scoring a 395M model on 8,797 items on the CPU.

| Measure | Before | After |
|---|---|---|
| Topic: health plus next steps (label index) | 1.3 s | 0.1 s |
| Every card's health and next steps, one after another | 5.5 s or more (load 12) | 3.3 s (load 21 to 24) |
| Specialists: page requests, list requests | 56, 2 | 47, 1 |
| Specialists: page data loaded, two runs each | 5.19 s, 4.76 s | 4.37 s, 7.27 s |

The timings swing with the load more than the two versions differ, so the browser runs do not separate them; the
requests they drop take about 0.8 s of server time when timed alone. The page meets the 5 s target in three of
four runs on a machine busier than the one measured above.
