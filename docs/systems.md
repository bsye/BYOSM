# Systems: teams of specialists

Status: the router, the tagger, tools, the system spec, its bundle, its end-to-end board, prefix training and the
text gate are built. Normalisers and the planner are not.

## Goal

One sentence describes a whole task, not a single model. BYOSM plans a team of small specialists, builds and proves
each one, wires them to your code, and ships one bundle that does the task with **no LLM at runtime**.

```bash
byosm system new assistant "while I type, notice when I'm scheduling something and offer to book it"
```

The LLM works only at build time: it plans the team, writes data, judges. At runtime it's small models and your code.

## The pieces

```
input stream ─▶ ① router ──────────────▶ ④ gate ─▶ ⑤ action (your code)
(keys, audio,  ─▶ ② tagger: arguments ──┘   │
 pages)        ─▶ ③ normaliser: values ─┘   └─ fires once, when stable
```

| piece | what it is | built? |
|---|---|---|
| ① router | classifier: which action, or none | ✅ `classify` |
| ② tagger | a label per word: which words are the arguments | ✅ `frame` on text, `span_f1` |
| ③ normaliser | text to value: "next fri 3pm", "€1.2k" | ❌ rules first, small models where rules fail |
| ④ gate | "sure enough, for long enough": hysteresis over a stream | ✅ audio and text (`TextGate`, tuned on calib) |
| ⑤ action | a typed function you write; the team only picks it and fills its arguments | ✅ `tools:` in the system spec |
| the team | one spec wiring the pieces, one bundle running them | ✅ `byosm system build` |

Audio clips (a wake word, a voice command) and documents (a page type, its fields) are more input kinds for ① and ②.

## The system spec

```yaml
name: assistant
input: {kind: text, stream: keystrokes}
nodes:
  intent:  {specialist: assistant-intent}                 # ① router
  slots:   {specialist: assistant-slots, when: intent != none}  # ② tagger
gate: {on: intent, min_confidence: 0.9, stable_for: 3}    # ④ fires after 3 agreeing keystrokes
tools:                                                    # ⑤ your code
  book_meeting: {when: intent == schedule, args: {who: PER, when: DATE}, call: myapp.calendar:book}
evals:
  - {name: massive, kind: hf, dataset: AmazonScience/massive, metric: exact_frame}
```

Each `specialist:` is an ordinary specialist with its own spec, board, benchmark and rivals. The system has its own
board too, scored end to end.

## How a system is built

1. **Plan.** The LLM splits the sentence into nodes and tools, and writes one specialist prompt per node.
2. **Build the nodes.** `byosm auto` runs per node, each against its own public benchmark when one exists.
3. **Wire.** Tool arguments are matched to tag types; the gate is tuned on calib.
4. **Prove.** End-to-end eval on a public benchmark with the same honesty rules (choose on calib, report test once,
   paired bootstrap against the LLM doing the whole task alone).
5. **Ship.** One bundle: every node's ONNX, the wiring, and one runtime with `step(input)` that emits events.

## How we prove it

| | measure | rival |
|---|---|---|
| right action and arguments | exact frame match (action + every argument right) | Qwen 3.8, given the tool schema |
| early, while typing | characters typed before the right action fires | (an LLM needs the whole sentence) |
| never wrong | wrong fires per 100 inputs | |
| cheap | ms per keystroke for the whole team, on CPU | seconds per call |

First benchmark: **MASSIVE** (60 intents, 55 slot types, 51 languages). SNIPS and ATIS for quick English checks.

## Order of work

1. ✅ Tagging (this commit): the ② piece, proved alone on CoNLL-2003.
2. ✅ The system spec and runtime, with ① + ② + ⑤ on MASSIVE, scored end to end against Qwen (`systems/assistant`).
3. ✅ Streaming text: train on prefixes (`train.prefix_rate`), the text gate, share typed before the action fires.
4. Normalisers for dates, times, amounts, durations.
5. Layout documents (LayoutLMv3 or LiLT) as another tagger input; audio clips as another router input.
6. `byosm system new`: the planner that does steps 1 to 5 from one sentence.

## Out of reach, on purpose

Writing new text (an email body) and multi-step reasoning. A team can pick an action and copy its arguments from the
input; it cannot compose. When a task needs that, the planner says so instead of pretending.
