# Hard cases plan

Written on 2026-10-29 for tool choice held, as a default path for every specialist. Two parts that only work
together: promote on the suites where the model is weak, then add hard cases round after round until those suites
stop moving.

## What we know

| Finding | Evidence |
|---|---|
| The primary suite has no headroom | Ten runs on builtin calib land at 0.93 to 0.96; the same recipe moves 0.913 to 0.942 by seed alone |
| Promotion leans the wrong way | v5-r1 and v6-r1 top builtin calib (0.962, 0.961) and are among the worst on When2Call (0.453, 0.416) and BFCL-live (0.667, 0.642) calib |
| Mined hard negatives help where they apply | Near-miss lifted "no tool fits" on When2Call from 0.66 to 0.75; "ask for info" stayed near 0.30 because the judge relabelled every missing-argument row as a fit (fixed, `near-miss-labels` is training on the corrected rows) |
| Rows aimed at dev errors cannot reach this gap | Dev has the training distribution and 8 errors at 0.99; refine rounds were neutral on Banking77 with two seeds |
| The public recipe is known | [When2Call](https://arxiv.org/abs/2504.18851) trains on 2,000 each of tool call, ask and cannot answer; on an 8B model "ask" went from 27% to 71% |
| The hard suites have calib splits big enough to select on | BFCL 240 items, BFCL-live 1,241, When2Call 2,198 (LLM-written has 8, too few) |

## Part 1: select on the hard suites

A spec names its selection suites (`selection_suites`, default the primary only, so nothing changes elsewhere). The
selection value is the mean over those suites of the calib score, on exact requests where the suite counts them. The
sure lead is a paired bootstrap that resamples each suite's calib items on its own and averages the per-suite deltas;
seed noise is the pooled spread of that mean across a recipe's seeds. Test stays report-only.

- GIVEN tool choice held with selection suites builtin, BFCL, BFCL-live and When2Call, WHEN a candidate leads on the
  mean by more than seed noise and 95% of resamples, THEN it is promoted and the reason names every suite's delta.
- GIVEN a candidate that wins builtin calib and loses the mean, THEN it is kept out, and the log says which suites it
  lost.
- GIVEN a suite with fewer than `MIN_CALIB_ITEMS` calib items, THEN it is left out of the mean and the reason says so.
- GIVEN a specialist without `selection_suites`, THEN promotion behaves exactly as today (a test pins it).

Edge cases: a suite scored for one side only (the candidate is scored on every selection suite before the vote, the
seed replicas too); a suite whose calib changed between runs (its fingerprint differs, so it is left out and logged).

## Part 2: add hard cases round after round

A round takes the promoted model, scores it on a mining pool that is never scored for selection or report, keeps the
cases it gets wrong or is least sure of, labels them and trains the next version on the previous data plus them.

| Step | What it does |
|---|---|
| Pool | Rows like the weak suites but never in them: xlam rows not used for training, When2Call's train split, other tool catalogues. Deduped against every calib and test split. |
| Mine | Score the pool with the current model; take the errors and the least sure rows, up to `HARD_ROWS` per round (3,000), balanced across the suite's kinds of request. |
| Label | The source's gold label where it has one; the judge with the shared definitions otherwise. A kind the judge flips more than half the time stops the round and names the definitions (the bug we just fixed). |
| Train | Warm start from the previous round's weights on the previous data plus the new rows. The data lineage is in the spec (`data.rounds`), so the key changes and old rows stay with their experiment. |
| Decide | Promotion by Part 1, with seeds. Stop when a round gains less than seed noise on the selection mean, and name the next lever. |

- GIVEN tool choice held after `near-miss-labels`, WHEN a round runs, THEN it adds rows from the pool only, the next
  version's spec records the round, and its health shows the mined rows per kind.
- GIVEN a round whose new rows include a calib or test item, THEN the round fails loudly before training.
- GIVEN two rounds in a row within seed noise, THEN the loop stops and the health check's next step names the lever.

It closes the loop the usual way: a health finding ("hard cases left") whose `args` start a round, a catalogue entry
that targets `robust` and `real`, and a next step the ready loop runs.

## What I need from you

1. A go for Part 1, then Part 2 in that order.
2. A go to download When2Call's train split (23.9 MB, CC-BY-4.0, [NVIDIA/When2Call](https://github.com/NVIDIA/When2Call)) into the pool.

## Measures

Before and after, three seeds each: When2Call calib and test on every kind of request, BFCL-live, builtin, and the
time per round. The target is "ask for info" above 0.6 exact with builtin within seed noise of today.
