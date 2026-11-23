# Games: a stronger poker bot on a generic game layer

## Goal

One sentence builds a game specialist.

- A self-play teacher solves the game.
- A small numeric student is distilled from it, using soft move probabilities on the states the student itself reaches.
- Strength is measured by play.

Poker (6-max no-limit hold'em) is the first real game. No module outside `byosm/games/poker/` knows about cards.

Today's poker specialist is a 22M MiniLM text classifier:

| Measure | Ours | References |
|---|---|---|
| PokerBench exact move | 51.0% | GPT-4 53.6, fine-tuned Llama-8B 78.3 |
| Pluribus agreement, 8 move classes | 58.3% | always-fold floor 52.3, open-source rebuild 66.8 (different protocol, see Evaluation) |

The target is a student that plays near a locally trained MCCFR blueprint, plus a server-side search mode on top whose gain is measured. "Approaches Pluribus" is not a claim this plan can earn (see [Expected strength](#expected-strength)).

## What exists and what is missing

| Piece | Exists | Missing |
|---|---|---|
| Soft-label training | `distill_loss` (model.py:74), `row_soft_targets` (train/teacher.py:76), `Trainer._loss` mix | `data/table.py:97` and `data/local.py:16` drop every column except input, label and split, so `meta['soft']` is never filled. `distill_loss` re-tempers the teacher (`teacher ** (1/T)`, default T=2), has no row weight and no legal mask. `task_loss` smooths labels by 0.02 over every class |
| Numeric student | `SpecialistModel` and `Head` do not depend on the backbone | No `vector` modality (spec/sections.py:9, modalities/base.py:14, export/frontends.py, rt.py:305, distill.py STUDENTS) |
| Legal moves | Suite-level `label_space` only | No per-row mask in training, scoring or rt.py. docs/POKER.md:45 claims masking that does not exist |
| Game engine | `phh.Table` and `lines.py`, two partial poker replays | No rules engine with legality, chance and payoffs. No game contract. `games/__init__.py` is empty |
| Choice among offered items | `options/situations.py` | Must be merged with the new `Decision`, not duplicated |
| Teacher | Sealed Rust rebuild ([noregrets](https://github.com/conorarmstrong/noregrets), pinned at 757f769): train and benchmark, with checkpoint resume | No verb returns move probabilities at states we supply. Runs in the foreground, not as a Job. Re-types the `HARDENED` flags. 16 CPUs / 28g set inline |
| Jobs | Runner with a count cap, resume and watchdog | No CPU or RAM claim per job. A sandboxed job on Linux cannot start Docker |
| Play evaluation | `resample.bootstrap_stats` / `bootstrap_runs`, `split_of` (calib and test only) | No play harness, AIVAT, LBR or best response. No signed payoff metric. No train seed namespace |
| Product loop | Health `check()`, techniques.yaml, Runner jobs | No game checks or techniques. Routing hard-codes `POKER` (routing.py:6-43) |
| Second game | None | Nothing proves the layer is generic |

## Architecture

```
            one sentence -> routing "game" kind -> GAMES registry
                                     |
 byosm/games/            (generic, never imports a game folder)
 +---------------------------------------------------------------+
 | game.py contract  registry.py  actions.py  decision.py         |
 | cfr_exact.py  mccfr.py  best_response.py  lbr.py  aivat.py     |
 | mcts.py  depth_limited.py  label.py  dagger.py  play.py        |
 | seeds.py  engine.py (host-side sealed runner)  create.py       |
 | suites.py  checks.py                                           |
 +---------------------------------------------------------------+
        |                  |                        |
 games/kuhn, leduc   games/tic_tac_toe       games/poker/
 (pure Python,       (pure Python, perfect   rules: pokerkit
  exact BR)           info, exact minimax)   encoder, menu, value fn
                                             teacher: sealed Rust image
                                               + label patch (stdin JSONL)
        |                  |                        |
        +------ vector student (MLP, legal mask, soft targets) -------+
                 -> ONNX -> rt.py "vector" frontend (numpy + onnxruntime)
                 -> optional server-side search mode (needs engine)
```

| Layer | Owns | Rule |
|---|---|---|
| `byosm/games/*.py` | Contract, records, solvers for small games, DAgger, play, AIVAT, LBR, exact best response, search, seeds, engine runner, suites, checks | No card, board or chip concept. A CI test fails if any of these modules imports a game subfolder |
| `byosm/games/<game>/` | Rules, encoder (state to float32 vector), action vocabulary, AIVAT/LBR value function, payoff unit and scale, optional `EngineSpec` | A fixture game defined in `tests/` registers and runs the whole pipeline without editing any other file (CI test) |
| Shared BYOSM | `vector` modality; `soft`, `legal` and `weight` columns; mask, weight and temperature in `task_loss`, `distill_loss`, scoring and rt.py; `payoff` metric; routing `game` kind; job resource claims | Extend the existing layers, never fork them |

The contract uses [OpenSpiel](https://github.com/google-deepmind/open_spiel) State API names: `new_initial_state`, `current_player`, `legal_actions`, `chance_outcomes`, `apply_action`, `information_state_key`, `returns`, `clone`. An adapter can then be added later without reshaping anything. OpenSpiel is not a dependency: `open_spiel.python.scale` needs JAX and Python 3.12+, and the venv is 3.11.15 with torch 2.14.1.

### Core modules (each under 200 lines)

| Module | Does |
|---|---|
| `game.py` | `Game` / `State` protocol, plus `zero_sum`, `enumerable`, `payoff_scale` (the game names its unit), `encoder`, `value`, `engine` |
| `registry.py` | `names()`, `get(name)`. Fails loudly on an unknown game |
| `actions.py` | Discretises actions relative to the offer, moved from `poker/moves.py:23-60` with one implementation for both paths. Off-tree sizes map to the nearest abstract size in log space |
| `decision.py` | `Decision(state_key, features, legal, soft, weight, episode_key, source)`. Merges `options/situations.py` and replaces `Spot`/`Raw` on the numeric path |
| `seeds.py` | Three hashed namespaces (train, calib, test) over deal or episode keys, built on `split_of`. Each DAgger round gets a fresh calib block |
| `engine.py` | Host-side runner, launched by the Runner as its own job kind, never from inside a sandboxed job. It imports `sandbox.containers.HARDENED` and adds `--network none`, `--pids-limit` and a nosuid/nodev tmpfs. It runs in batch mode or as a long-lived `-i` stream. CPU and RAM come from `byosm.yaml` and are declared as the job's resource claim |
| `label.py` | Chunked labelling keyed by a content hash of (round, seed range). A re-run skips finished chunks |
| `dagger.py` | One resumable round: beta mix, rollouts, label, append, warm-start retrain, keep or revert on that round's calib block |
| `play.py` | Lockstep harness: N tables step together and each policy is called once per step with a batch. Duplicate, seat-rotated deals; the statistical unit is one deal group |
| `aivat.py`, `lbr.py`, `best_response.py` | Generic [AIVAT](https://arxiv.org/abs/1612.06915), [local best response](https://arxiv.org/abs/1612.07547), and exact best response / NashConv for enumerable games |
| `cfr_exact.py`, `mccfr.py` | Vanilla CFR+ for enumerable games, and external-sampling linear MCCFR |
| `mcts.py`, `depth_limited.py` | MCTS for perfect-information games, and depth-limited solving with 4 biased continuations ([Brown et al. 2018](https://arxiv.org/abs/1805.08195)) for small imperfect-information games |

### Shared changes, precisely

- `table.py` and `local.py` carry `soft` (a JSON list in label order), `legal` (a JSON list of slot indices) and `weight`.
- In `task_loss`, illegal logits are set to -inf, and label smoothing is spread only over the legal slots.
- `distill_loss` takes the mask and per-row weights. A game spec sets `distill_temperature: 1.0` so stored strategies are not re-tempered, and leaves `distill_from` unset so `loop.py` does not override the stored soft targets.
- Scoring and rt.py apply the same mask and the same tie-break (slot order).
- `Runner._admit` gains a per-job CPU and RAM claim. Jobs queue while the claim does not fit, and no running job is ever cancelled.

## The improvement loop

Every step is a `byosm game <name> <verb>` Runner job. Each job is chunked and idempotent, streams progress, resumes after a restart, queues instead of being rejected, and is never cancelled. Every tunable named below lives in `byosm.yaml` under `games:`.

1. **new.** Routing picks `game` plus a name from `games.names()`. An unknown game fails loudly with "no rules engine registered". `create.py` writes a spec with modality `vector`, primary metric `payoff` on the calib play suite, and the agreement, play and best-response suites.
2. **teach.**
   - Kuhn and Leduc: `cfr_exact.py` / `mccfr.py` run in process.
   - Tic-tac-toe: exact minimax.
   - Poker:
     1. Cache `bucket-table`, keyed by its flags.
     2. Run memory sizing at 25M and 50M iterations, logging peak RSS from `docker stats` into the job log.
     3. Train with `--checkpoint/--resume` at the largest iteration count that fits.
3. **label (round 0).** Teacher self-play on train-namespace seeds produces Decisions with soft distributions and reach/visit weights. Calib and test state keys are excluded.
4. **distill.** The existing Trainer runs with `distill_alpha` (0 means pure soft targets), T = 1, masks and weights. The model is exported through ONNX. The student's action slots are the teacher's abstract menu, so self-play stays on the tree.
5. **play and best response on calib.** Duplicate play with AIVAT against the configured fields. Exact best response for enumerable games, LBR for poker.
6. **DAgger round k.**
   - The teacher acts with probability `beta_k` (from `byosm.yaml`, default 0.5^k), the student otherwise.
   - The teacher labels every state reached.
   - Rows are appended, never replaced, and retraining warm-starts from the last weights.
   - A round is kept only on a paired gain on its own calib block. Otherwise it is reverted and the result recorded.
   - Rounds stop after `dagger_max_misses` misses in a row.
7. **search relabel (gated).** Relabel the top `relabel_kl_quantile` of decisions by KL with the engine's `search_dist`. Keep it only on a paired calib gain. The rebuild measured expert iteration as a null result (crossplay -38 ±62, per its README).
8. **search gain.** The server-side mode "student preflop, engine search postflop" is measured as a paired gain against the student alone and against the blueprint alone. It is kept as a separate row only if the gain clears zero.
9. **health to next steps.** Each check's `args` is the step that fixes it. The ready loop, Measure everything and the proposer LLMs act on them through `targets` in techniques.yaml.

| Check | Measures | `args` (next step) |
|---|---|---|
| `game-coverage` | Share of states the student reaches that have a teacher infoset (not the check/call fallback) | `teach --iters <next>` |
| `game-agreement` | Top-1 and KL to the teacher on held-out calib states | `distill` with soft labels, or a bigger preset |
| `game-drift` | Agreement on teacher states minus agreement on student states | `dagger --rounds 1` |
| `game-play` | Paired calib delta against the previous champion, and the lower 95% CI bound against the reference field | `dagger` or `search-gain` |
| `game-best-response` | Exact exploitability (enumerable games) or LBR win rate (poker, never called exploitability) | `teach --iters <next>` or `search-gain` |

New techniques.yaml entries, each targeting the checks above: `soft-teacher-labels`, `dagger-rounds`, `structured-student`, `legal-move-mask`, `duplicate-aivat-play`, `play-time-search`, `teacher-iterations`. One later candidate, `self-play-rl-finetune`, is to be tried on Leduc first.

## Evaluation

Selection uses calib only; each DAgger round uses a fresh calib block. Test is reported once per promoted model, with a 95% bootstrap CI and a paired `p_better`. Every number comes from `byosm report`.

`payoff` is signed and unbounded. It gets its own board column and check, and never feeds `Objective.target` or the health `_target`/`_rival` checks.

| Measure | Game | What it proves | Protocol check before it counts |
|---|---|---|---|
| Exact exploitability | Kuhn, Leduc | Whether soft labels and DAgger move the student toward Nash | Kuhn value -1/18 within 1e-3; Kuhn closed-form equilibria have exploitability 0 within 1e-6; a best response to an equilibrium gains nothing; NashConv is at least 0 |
| Loss against minimax | Tic-tac-toe | The contract and the loop work on a perfect-information game | Minimax against itself draws every game; random against random is symmetric over seat swaps |
| Play, mbb/hand, duplicate with AIVAT | Poker | Strength, the primary measure. Fields: 5 blueprint copies, always-call, random, previous champion | Our harness's blueprint vs caller and vs random land inside the CI of the engine's own `eval --aivat`. Always-call vs always-call duplicate is exactly 0 |
| LBR win rate (heads-up, multiway seat) | Poker | A lower-bound probe | Our `lbr.py` on the blueprint agrees with the engine's `lbr --multiway` within CI. On Leduc, LBR is at most the exact best response (test) |
| Pluribus agreement, mapped action, covered decisions only | Poker | Comparable with the rebuild's 66.8% | The label verb's argmax reproduces the engine's `benchmark` exactly. Compare with 66.8 only after a Wide-menu, 12-bucket blueprint reproduces it |
| Pluribus agreement, 8 move classes | Poker | Continuity with 58.3% | Replay still gives exactly 15,169 decisions. A chip-exact check is new and has to be added to our replay |
| PokerBench exact move | Poker | Continuity with 51.0. Reported, not selected on | Data keys unchanged (pinned by test) |
| Teacher top-1 / KL (held out) | All | Imitation quality | Held-out set fixed by hashed key |
| Latency, bundle size, teacher RSS and wall time, harness hands/s | All | Cost | Read from job logs |

Variance:

- Before any DAgger round, measure the CI width at the configured hand count with AIVAT on and with it off. When the CI is too wide, raise the hand count instead of declaring a null.
- LBR and best-response probes run with `RAYON_NUM_THREADS=1` over `probe_seeds` paired seeds (default 8).

### Expected strength

These are not estimates we have measured. Each one will be replaced by a measured number.

| Bot | Expectation | Reference point |
|---|---|---|
| Leduc student | Close to its teacher's exact exploitability | Exact, so it is measured either way |
| Poker student, exported, no search | Below its teacher against the blueprint field, and clearly ahead of caller and random. No number is claimed | Blueprint: heads-up BR +116, multiway LBR +929, Slumbot -714 ±331 (rebuild README) |
| Mapped-action Pluribus agreement | A few points below our own blueprint | 66.8% (Wide menu, 200M iterations, no search) |
| PokerBench | May not rise | Fine-tuned Llama-8B 78.3 ([PokerBench](https://arxiv.org/abs/2501.08328)) |
| Server-side search mode | Positive in self-play, possibly negative against foreign opponents | Rebuild: search vs blueprint +125 ±40 in self-play; Slumbot -714 to -1771 with search on |
| Versus Pluribus | Not claimable | Pluribus: an 8-day, 64-core blueprint plus real-time search ([Brown & Sandholm 2019](https://www.science.org/doi/10.1126/science.aay2400)). Ours: hours on at most 18 CPUs and 31.35 GiB of Docker memory |

What this plan can honestly claim: one `byosm game <name>` pipeline reaches measured teacher-level play on Kuhn, Leduc and tic-tac-toe, and on 6-max NLHE it produces a millisecond CPU student that plays near an open-source MCCFR blueprint. Every number is measured by play with CIs, and the search row carries its own measured gain.

## Phases

| Phase | Builds | Done when |
|---|---|---|
| 0. Land and agree | When the owner asks: commit the in-flight teacher work as its own commit. Fix POKER.md:45. Move MCCFR and search out of "Not in scope". Owner approves this doc and questions 1-9 | GIVEN a clean tree, WHEN approved, THEN tests pass at their current baseline |
| 1. Contract and small games | `game.py`, `registry.py`, `actions.py`, `decision.py` (merging situations), `seeds.py`, `cfr_exact.py`, `mccfr.py`, `best_response.py`, `mcts.py`, `games/kuhn`, `games/leduc`, `games/tic_tac_toe`, the import-boundary test, and the fixture-game test | GIVEN Kuhn, WHEN solved, THEN its value is -1/18 within 1e-3 and the closed-form equilibria have exploitability 0. GIVEN Leduc MCCFR, WHEN iterations grow, THEN exploitability falls below `games.leduc.target`. GIVEN the fixture game, WHEN registered from tests, THEN the pipeline runs with no other edit. GIVEN the existing text specialists, WHEN `moves` moves to `actions.py`, THEN their data keys are unchanged |
| 2. Vector student and masks | `modalities/vector.py`; `soft`, `legal` and `weight` columns; mask, weight and T in both losses; scoring; the `vector` frontend and rt branch; STUDENTS entry; `label.py`; `create.py`; the `game-agreement`, `game-coverage` and `game-best-response` checks | GIVEN Leduc soft labels, WHEN distilled, THEN the ONNX student runs in rt.py without torch, and Health shows its exact exploitability next to the teacher's and a hard-label student's. GIVEN any suite, WHEN scored, THEN no prediction falls outside the row's legal set. GIVEN T=1, WHEN stored soft targets are loaded, THEN they reach the loss unchanged (test) |
| 3. Play, AIVAT, DAgger, payoff | `play.py`, `aivat.py`, `lbr.py`, `dagger.py`, `depth_limited.py`, the `payoff` metric with its own column and check, `game-play`, `game-drift`, techniques entries, Runner resource claims | GIVEN Leduc, WHEN DAgger rounds run, THEN the paired DAgger vs round-0 result on calib and test is recorded in design-notes, whichever way it comes out. GIVEN AIVAT, WHEN compared on the same deals, THEN its mean is within the raw mean's CI and its variance is lower. GIVEN tic-tac-toe, WHEN the same pipeline runs, THEN no code outside its folder changes. GIVEN two jobs whose claims exceed the limits, WHEN submitted, THEN the second queues |
| 4. One sentence to a game | Routing `game` kind from `games.names()`, `byosm game <name> ...` CLI, `app/allowed.py`, a Runner job per step, next steps from check args, card and board using existing patterns, the screenshot and critique cycle, Prettier | GIVEN "a Leduc poker bot" in Home, WHEN submitted, THEN it teaches, distils, plays and ranks with no settings asked. AND WHEN the app restarts mid-job, THEN the job resumes (checked in the browser and the database) |
| 5. Poker engine and teacher | `poker/game.py` on [pokerkit](https://github.com/uoftcprg/pokerkit), replacing `phh.Table` and the `lines.py` replay; one seats/depth constant; `menu.py`, `encoder.py` (equity cached by content hash), `value.py`; the generic `engine.py` as a host-side job kind; the `label.patch` Rust verb (relaxes the hero name and symmetric stacks); cached bucket-table; memory sizing; repo URL changed to noregrets at the same commit | GIVEN the Pluribus hands, WHEN replayed through pokerkit, THEN 15,169 decisions come out AND the new chip-exact check passes AND the PokerBench and Pluribus data keys are unchanged. GIVEN the label verb, WHEN run on the Pluribus hands, THEN its argmax reproduces the engine's `benchmark` exactly. GIVEN the sizing runs, THEN peak RSS, iterations and wall time are in the job log, and harness hands/s and per-decision engine latency are recorded against `games.poker.eval_budget` |
| 6. Poker student by play | Round-0 table (`games.poker.round0_decisions`, starting at 2M), vector student, DAgger rounds on fresh calib blocks, play fields, LBR, promotion switched to calib play | GIVEN the harness, WHEN blueprint vs caller and vs random are played, THEN results fall inside the CI of the engine's `eval --aivat`. GIVEN each round, THEN its paired calib delta and coverage are recorded. GIVEN the promoted student, THEN test mbb/hand ± CI, both agreement protocols and the LBR rows are in docs/results.md via `byosm report` |
| 7. Search mode | Poker "student preflop + engine `search_dist` postflop" through the stream engine; the gated KL relabel; `play-time-search` technique; Leduc search through `depth_limited.py` | GIVEN the search mode, WHEN played paired against the student alone, the blueprint alone and caller, THEN each gain has a CI, AND the row is kept only if the gain clears zero and is labelled "needs engine at play time" |

## Edge cases

| Case | Handling |
|---|---|
| Off-tree bet (Pluribus, PokerBench rows) | Map to the nearest abstract size in log space. An unseen infoset makes the teacher fall back to check/call: counted in `game-coverage`, with row weight 0 |
| Teacher menu differs by street and node | Fixed slot set. The per-row legal mask removes slots the node does not offer, in losses, scoring and rt.py |
| PokerBench spots with unequal stacks | The label patch accepts them. Otherwise they are skipped and the count is logged |
| Rarely visited infosets | Soft targets weighted by reach/visits, and the weight reaches `distill_loss` |
| Teacher near the memory limit | Sizing decides the iteration count first. Accept a smaller blueprint rather than an OOM kill. Never cancel a running teacher |
| Teacher competing with other jobs | Declared resource claim; other jobs queue |
| Engine launched from a sandboxed job | Not allowed. Engine runs are host-side Runner jobs; the Docker socket is never mounted into a sandbox |
| App restart mid-DAgger | Label chunks are content-hashed; the round resumes at the first unfinished chunk |
| Calib/test leakage | Three seed namespaces. Pluribus hands are split by whole episode. Each round gets a fresh calib block |
| Tied argmax | Tie-break by slot order, identical in trainer, scoring and rt.py |
| Negative or unbounded payoff | Own check and column, never the [0,1] objective |
| Non-zero-sum or n-player game | No exact exploitability. Report play and LBR win rate only |
| Perfect-information game | `information_state_key` is the full state, and the same pipeline runs |
| Non-reproducible LBR/BR | `RAYON_NUM_THREADS=1`, paired seeds |
| Image uid 10001 vs bind mount on Linux | The work dir is created writable for that uid before the run, and the check is tested |

## Not in scope

- Beating or claiming parity with Pluribus, or calling anything "unexploitable".
- Search inside the exported bundle (rt.py stays a single file with numpy and onnxruntime).
- Neural CFR ([Deep CFR](https://arxiv.org/abs/1811.00164), [ESCHER](https://arxiv.org/abs/2206.04122)), [ReBeL](https://arxiv.org/abs/2007.13544)-style value nets, and RL fine-tuning. These are a later technique candidate, to try on Leduc first.
- External opponents (Slumbot, GTO Wizard) until their terms are checked. Both are heads-up only.
- Publishing blueprints.
- An OpenSpiel runtime dependency, or `open_spiel.python.scale`.
- Connect Four and other larger perfect-information games, until tic-tac-toe passes.
- Single-player sequential tasks.

## Decisions

The owner answered the open questions on 2026-11-23.

| # | Question | Decision |
|---|---|---|
| 1 | Rust `label` verb, patched into the teacher image | Yes |
| 2 | pokerkit as the one poker rules engine | Yes |
| 3 | OpenSpiel | No dependency now; an adapter later |
| 4 | Raise Docker memory if the teacher needs it | Yes |
| 5 | Encoder in the bundle: numpy features plus an equity lookup table | Yes |
| 6 | Server-side search mode, its own row, kept only with a measured gain | Yes |
| 7 | Poker promotes on calib play against the blueprint field | Yes |
| 8 | Publish the distilled student | No: it stays private, never published |
| 9 | The text specialists | Removed when not good: `poker-move` and `poker-facts` are archived (hidden, recoverable), since `poker-mix` beats or ties both on every suite |
