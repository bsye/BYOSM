# Poker specialist plan

Goal: one sentence in Home ("Given a 6-max no-limit hold'em spot, pick the optimal move") builds a small CPU specialist
that beats LLMs on [PokerBench](https://huggingface.co/datasets/RZ412/PokerBench) and reports its agreement with
Pluribus's own decisions. It imitates solver and Pluribus decisions; it is not a search bot and never plays for money.

## What exists and what is missing

| Need | In BYOSM today | Missing |
|---|---|---|
| Text classifier, LLM rivals, calib/test, CI, seeds, vote | yes | |
| Public benchmark suite with a protocol check | `byosm/benchmarks/` (BFCL, When2Call, eot-bench) | a PokerBench suite |
| Offered choices that change per row | tool `options` (pairs per offered tool) | a move label that maps back to the offered move |
| Card maths (hand class, equity, draws, pot odds) | none | deterministic features, the part a 100M encoder cannot learn from text |
| Hand history parsing | none | [pokerkit](https://pypi.org/project/pokerkit/) (MIT, reads PHH) |
| Unlabeled pretraining at scale | LeJEPA on unlabeled inputs | rendered spots from human hands |

## Data

| Source | Rows | Role |
|---|---|---|
| [PokerBench](https://huggingface.co/datasets/RZ412/PokerBench) train (Apache 2.0) | 60k preflop + 500k postflop, solver labels | training |
| PokerBench test | 1k preflop + 10k postflop | primary suite, split into calib and test |
| [gabbler/PokerData](https://huggingface.co/datasets/gabbler/PokerData) `pluribus/` (CC BY 4.0) | 10,000 hands, about 30k Pluribus decisions | second suite: agreement with Pluribus |
| gabbler/PokerData `handhq/` | 21.6M human hands (2009, 25NL to 1000NL) | unlabeled pretraining only, never labels |

Each PokerBench spot offers 2 to 4 moves (`['Check', 'Bet 24']`, `['5.0bb', '3.0bb', 'check', 'fold']`) and the label
is one of them.

## Published references (EM = exact move, AA = action type only)

| Model | EM | AA |
|---|---|---|
| GPT-4, few-shot | 53.55 | 65.54 |
| Llama-3 70B, few-shot | 39.16 | 49.78 |
| Llama-3 8B, fine-tuned on the train set | 78.26 | 80.64 |

From Table 2 of [the PokerBench paper](https://arxiv.org/abs/2501.08328). Our own LLM rival (Claude Haiku, plus
qwen3.8 locally) is scored by our code on the same 11k spots; the published rows are shown only as references.

## Design

1. **Labels are moves relative to the offer:** `fold`, `check`, `call`, `allin`, `bet small`, `bet large`, `raise small`,
   `raise large`, where small and large are the smaller and larger offered size of that kind. A prediction maps back to
   exactly one offered move, so EM and AA are both computed from it. Moves not on offer are masked out at prediction.
2. **The spot is rendered once, in one place,** from the structured CSV row (or a Pluribus decision point), with
   computed facts appended: made hand, equity against a random hand, draws, pot odds, stack to pot ratio, position,
   street. Rules compute these; the model only weighs them.
3. **Poker code lives in `byosm/games/poker/`** (render, features, labels, Pluribus extraction); the suite registers in
   `byosm/benchmarks/catalog.py` like the others.
4. **Pluribus suite scores AA only:** Pluribus played 100bb stacks with its own sizes, which do not match PokerBench's
   offered sizes. Pluribus mixes its actions, so 100% agreement is not the ceiling; we report the rate with a 95% CI.

## Phases

| Phase | Builds | Done when |
|---|---|---|
| 1 | PokerBench suite, renderer, move labels, EM and AA metrics, LLM rival | GIVEN the sentence in Home WHEN the build ends THEN the board shows our EM and AA on 11k spots next to Haiku's, with CIs |
| 2 | Card-maths features in the rendered spot | the feature run beats the plain run on calib in the four-seed vote |
| 3 | Pluribus agreement suite from the PHH files | GIVEN the 10k hands WHEN scored THEN the board shows AA agreement per street with a CI |
| 4 | Human hands as unlabeled pretraining | a LeJEPA run on them beats no pretraining on calib, or is dropped |
| 5 | Half of Pluribus's hands in training, exact-copy dedupe only | done: the suite scores only hands the model never saw |
| 6 | A solver teacher: train the [open-source Pluribus rebuild](https://github.com/conorarmstrong/pluribus) (MIT, Rust) blueprint here, label millions of 6-max spots with its move probabilities, distill them | GIVEN its blueprint WHEN it labels spots THEN its own Pluribus agreement reproduces near 66.8%, and the distilled student beats the PokerBench-only student on the Pluribus suite |
| 7 | Soft labels: train on a move distribution, not one move | a student trained on teacher probabilities beats one trained on the teacher's top move, on calib |
| 8 | A structured model: cards, seats and amounts as numbers instead of text | it beats the text model on the Pluribus suite, or is dropped |
| 9 | Strength by play: mbb/hand against the rebuild's blueprint and simple bots, with AIVAT-style variance reduction | GIVEN 10k paired hands WHEN played THEN the board shows mbb/hand with a 95% CI |

The rebuild's trained blueprints are never published (commercial licence), so phase 6 trains one here: 200M
iterations take about an hour on 16 cores and the 400M one needed 16 GB. Agreement with Pluribus is a proxy; phase 9
is the measure of strength.

## Edge cases

- A spot offering three sizes of one kind: counted at import; if any exist, the label set grows, never silently merged.
- A label not among the offered moves: the row fails import loudly.
- Pluribus decisions at a size PokerBench never offers: kept, scored on action type.
- Duplicate spots between train and test: removed from train by the existing near-copy check.

## Not in scope

Playing on real-money sites (their terms ban bots), opponent modelling. The MCCFR teacher and search are now in
scope: see [GAMES.md](GAMES.md).

## Open questions

1. Install pokerkit (MIT) for PHH parsing and hand evaluation, or write our own evaluator? Recommendation: pokerkit.
2. Download size: PokerBench is 724 MB of JSON plus CSV and the Pluribus folder is 5.5 MB; HandHQ is only needed in
   phase 4. OK to download the first two now?
3. Train on all 560k PokerBench rows, or start from a balanced 60k and let Keep training add more? Recommendation:
   start at 60k, then grow.
