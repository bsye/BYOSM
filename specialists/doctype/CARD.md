**What this model is.** A two-view document classifier: the page image goes through the visual encoder of
[`microsoft/dit-base-finetuned-rvlcdip`](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip) (read through its
own pooler, exactly as its classifier did), the page text (PDF text layer, or RapidOCR for scans) goes through MiniLM,
and a learned gate fuses the two. It was then trained on 34,935 RVL-CDIP documents (the `val` split plus a builtin
held-out split).

**How to read the comparison with DiT-base.** On 3,200 documents from RVL-CDIP's original `test` split this model scores
**94.8% vs DiT-base's 94.0%** (+0.8 pt, paired-bootstrap 95% CI +0.3 to +1.3, P(better) = 1.00; it fixes 50 of DiT's
errors and introduces 24). The gain comes from reading the text, which DiT does not. This is "DiT-base + the text view
beats DiT-base", not a model that reached DiT's accuracy from less data: it starts from DiT's checkpoint, which was
fine-tuned on RVL-CDIP's 320k-document `train` split. That split is disjoint from the `test` documents used here. On the
800-document calib split the two tie (93.5% vs 93.6%).

**Label noise sets the ceiling.** On 69 of the 3,200 test documents (2.2%) this model, DiT-base and an independent BYOSM
model all agree on the same label, and it is not the given one. In a manual look at a random sample, about half were
clearly mislabelled (for example an email printout labelled `letter`) and others were genuinely ambiguous
(`scientific_report` vs `presentation`). Published estimates put RVL-CDIP's label error rate at several percent, so
accuracy far above ~96-97% on this benchmark would mean fitting the noise.

**Confidence-gated use.** If you need very high precision, act only on confident predictions and route the rest to a
person. With a confidence threshold fitted on calib, a DiT + BYOSM ensemble answered 59.6% of test documents at 99.6%
accuracy. Measure this on your own documents before relying on it.

**Domain.** RVL-CDIP is 1990s scanned tobacco-industry correspondence at low resolution. Modern born-digital PDFs are
out of distribution; add an eval suite of your own documents to see how it does there.
