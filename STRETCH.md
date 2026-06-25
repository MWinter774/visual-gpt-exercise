# Stretch exercises — push the image-GPT further

> Do these **after** the core exercise ([EXERCISE.md](EXERCISE.md)) works. Each is a *bonus*: a
> small change to the **data** or the **prompt** that teaches a new idea. `model.py` still never
> changes — that's the whole point.

Each task lists: the goal · the idea · what to change · hints (click to reveal) · how to know it
worked. Try it yourself before peeking.

---

## 1. Complete a half-drawn digit (image in → image out)

- **Goal.** Give the model the **top half** of a real digit and have it draw the bottom half.
- **The idea.** Generation already takes a prompt — you seeded it with one pixel. The top half of
  an image is just a *longer* prompt: the first `seq_len // 2` tokens. So "completion" is the
  **same `generate()` call** with more seed tokens. No new model, no encoder.
- **What to change.** In your sampler, instead of seeding `[0]`, seed with the first half of a
  real validation image's tokens and generate the rest.
- <details><summary>Hint</summary>Take a val image's token row, keep `tokens[:seq_len//2]` as the
  prompt, `generate` the remaining tokens, then concatenate given+generated and reshape to H×W.</details>
- **Worked?** The completion should **continue the given strokes** (a 7's diagonal reaches the
  corner; a loop closes), not draw an unrelated digit. Pick prompts whose bottom half is non-trivial.
- **Insight.** "Input" and "output" are the same operation — generation fills in the tokens you
  don't provide. Conditioning is just a prefix.

## 2. Draw me a *specific* digit (class-conditional)

- **Goal.** Ask for a chosen digit ("draw me a 7"), not a random one.
- **The idea.** Prepend a **class token** to every sequence: `token = num_pixel_values + label`.
  Vocab grows 2 → 12; each sequence becomes `[class, pixel_0, …]`. The autoregressive shift wires
  the conditioning in for free — the model learns "given class 7, here's how a 7 starts."
- **What to change.** `prepare.py`: read the MNIST **labels** (the core prepare dropped them) and
  prepend `2 + label`. Sampler: seed with `[2 + d]` for the digit `d` you want; drop the class
  token before reshaping.
- <details><summary>Hint — labels</summary>The IDX label file is magic 2049, an 8-byte header,
  then one uint8 per image. Same "parse the bytes" approach as the images.</details>
- <details><summary>Hint — alignment</summary>Keep the same image-aligned batching and
  `x=seq[:-1], y=seq[1:]`. Position 0 is now the class token; nothing else changes.</details>
- **Worked?** Seeding `[2+7]` mostly draws 7s, `[2+0]` mostly 0s. A 10-row grid (one row per digit)
  makes it obvious.

## 3. Smoother strokes (gray levels)

- **Goal.** Less blocky digits — soft, anti-aliased edges.
- **The idea.** Binary forces every pixel to pure black/white, so edges are a hard staircase. After
  the 2×2 pool a half-covered pixel is genuinely gray (~128). Let the tokens carry that: quantize
  each pixel to **K gray levels** (e.g. K=4) → vocab K.
- **What to change.** `prepare.py`: `token = pixel*K//256` (clamp `0..K-1`), `vocab_size=K`, and
  store the K decode shades in `meta`. **Quantize the pooled gray values directly. Don't binarize
  first, or you'll be left with only 2 levels.** Sampler: map each token back to its gray value and write a
  grayscale image.
- <details><summary>Hint</summary>Decode shade for level `i` ≈ `round(i*255/(K-1))`; K=4 →
  `[0, 85, 170, 255]`.</details>
- **Worked?** Strokes look softer/rounder than binary at the *same* resolution and cost. Note: the
  loss is **not** comparable to binary (vocab-4 baseline is ln 4 ≈ 1.386 vs ln 2 ≈ 0.69) — judge by
  the picture. **Dial to try:** K = 4 → 6 → 8 (smoother shading, harder next-token problem).

## 4. Crisper digits (full resolution, 28×28)

- **Goal.** More detail than 14×14 allows.
- **The idea.** Skip the downscale — use the full **28×28 = 784 tokens** (still binary). Same
  everything, just a longer sequence.
- **What to change.** `prepare.py`: downscale factor 1 (no pooling). That's it — train/sample read
  `seq_len` from `meta`.
- **The cost-knob lesson.** Self-attention is O(seq_len²) and sampling ~O(seq_len³); 784 vs 196
  makes training ~4× slower and **sampling tens of seconds per image** on CPU. Keep `num_samples`
  tiny.
- **Pitfall, don't skip this.** At 28×28 the image is ~84% background. If you sample at a *low*
  temperature (like the 0.8 that worked for 14×14), the model over-commits to "background" and you
  get a **blank** image — while the loss looks great (low, because background is easy to predict).
  Lesson: **look at the output, not just the loss**, and the right temperature depends on the
  representation — 28×28 wants temperature ≈ **1.0**.
- **Worked?** Crisper strokes than 14×14 (compare side by side), with non-trivial ink (~15%).

## 5. Color (CIFAR-10) — and why it's hard

- **Goal.** Generate color images, and *feel* why color on a CPU is a different league.
- **The idea.** An RGB pixel can't be one of 256³ tokens. Run **k-means** over the RGB pixels to
  learn a **palette** of K colors; each pixel becomes its nearest palette index (vocab K) — the real
  iGPT trick. Downscale (e.g. 32→16) to keep the sequence affordable.
- **What to change.** `prepare.py`: load CIFAR-10, k-means a K-color palette, map each pixel to its
  index, store the palette in `meta`. Sampler: write an **RGB** PNG by decoding indices through the
  palette.
- <details><summary>Hint</summary>Sample ~50k pixels for k-means (plain Lloyd's, ~15 iters, no
  sklearn needed); save the K RGB centroids in `meta` for decoding.</details>
- **Expect mush; that's the result.** A tiny model on a CPU, longer sequences, and CIFAR's huge
  variety → coarse color blobs, not recognizable objects. The honest lesson: *"the transformer is
  generic" does not mean "a tiny transformer on a CPU solves everything."* Compute and
  representation matter. (Tip: decode some *real* images through your palette too — that separates
  the palette's ceiling from the model's limit.)

---

**The throughline:** in every one of these, `model.py` is untouched. You changed the **tokens**
(gray, color, resolution) or the **prompt** (half-image, class token). The transformer is a generic
sequence model; the representation is the design.
