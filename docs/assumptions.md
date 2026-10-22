<p align="center"><a href="../README.md">Tour</a> · <b>Goal and bet</b> · <a href="methodology.md">Methodology</a> · <a href="results.md">Results</a> · <a href="details.md">Details</a></p>

# Goal and bet

## Goal

Describe a task in one sentence. Get back a small model that does that one job better than general models, LLMs
included, proven on a public benchmark, and fast enough to run on any CPU.

```bash
byosm "classify the emotion of a short English text: joy, love, sadness, anger, fear or surprise"
```

That one command names the task, works out what kind of model it needs (text, audio, image or document; one label
per input or one per word or frame), writes the spec, finds the benchmark and the models to beat, trains, keeps going until it wins or
runs out of time, and exports an ONNX bundle.

## The bet

| | we bet that | where it stands |
|---|---|---|
| 1 | A small model trained for one job beats a general model at that job, LLMs included. | **Holds on every public benchmark so far.** Ahead of the best outside model on test everywhere (on names, only just), and far ahead of Qwen 3.8 27B on every task an LLM can do. On documents, DiT-base still edges us on calib. |
| 2 | We can stay honest: choose on calib, report test once, compare with a paired bootstrap. | **Built in** to promotion, the autopilot and export. Broken once by hand: the training-recipe default was picked on test ([recipe](results.md#the-training-recipe-measured)). |
| 3 | Specialists are cheap: they run side by side on a laptop CPU, no GPU, no PyTorch. | **Yes.** Size and CPU time are in every table. |
| 4 | A team of small specialists does a whole task without an LLM, and does it better than the LLM alone. | **Holds on MASSIVE**, the first system: a router and a tagger against Qwen doing the whole task. Streaming while you type is next. |

Still open:

- **Specialists vs one shared generalist** (`byosm bench`). Not run yet: it grades on a split the promoted models
  trained on, so it has to move to the public benchmarks first.
- **The autopilot on its own.** Two full runs from one sentence: emotion won inside its budget; topic needed two more
  data steps by hand. VAD, documents and intent were built step by step.

<p align="center"><a href="../README.md">Tour</a> · <b>Goal and bet</b> · <a href="methodology.md">Methodology</a> · <a href="results.md">Results</a> · <a href="details.md">Details</a></p>
