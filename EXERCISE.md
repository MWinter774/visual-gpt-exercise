# Exercise — make a GPT *see* and *draw*

> **Prerequisite:** you've already built a decoder-only GPT (Karpathy's *Let's build GPT* /
> nanoGPT) and trained the char-level Shakespeare model. You understand self-attention, the
> causal mask, next-token prediction, and sampling.

## The point

You think your GPT is a *language* model. It isn't. It's a generic **next-token predictor**
that knows nothing about language. To prove it, you'll feed it images instead of text — and
the **exact same `model.py`** will learn to generate handwritten digits.

You will change the **tokenizer**. You will **not** change the transformer.

When it works, you'll have a model that draws MNIST digits, pixel by pixel, on a CPU.

## The one rule

**Do not edit `model.py`. Not one line.** (It's Karpathy's GPT, provided in this folder.) If you ever feel you need to — stop. It
means your *data* is wrong, not the model. That realization *is* the lesson.

---

## The big idea (read this before coding)

- A 14×14 image is 196 numbers. Flatten it row by row → a **sequence of 196 tokens**.
- A token used to be "which character (of 65)". Now it's "which pixel value". If you
  binarize the image (black/white), the vocabulary is just `{0, 1}` — **size 2**.
- Next-token prediction becomes next-**pixel** prediction: given the pixels seen so far
  (scanning left→right, top→bottom), predict the next pixel.
- **Generating an image** = sampling that sequence, then folding it back into a 2-D grid.

Everything below just makes this concrete.

## Setup

Python 3.11 with PyTorch (CPU is fine), numpy, and requests. On macOS or Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/mnist/prepare.py
```

This downloads MNIST, turns each image into a token sequence, and writes
`train.bin` / `val.bin` / `meta.pkl`. The boring part — downloading and parsing MNIST's raw
byte format — is **given to you** (it's plumbing, not the lesson). The script ends by
decoding the first image back out of `train.bin` and printing it; **look at it.**

We default to **14×14** (a shrunk MNIST), not 28×28. You'll discover why in Part 3.

## What you'll build

You're given **`model.py`** (Karpathy's GPT, unchanged — never edit it) and
**`data/mnist/prepare.py`** (the tokenizer). You write **`train_image.py`** and
**`sample_image.py`** in this folder.

1. **Tokenize** images (mostly given — but *you* decide the vocabulary).
2. **Train** the GPT on image-sequences. ← the interesting trap lives here.
3. **Generate** images.
4. *(stretch)* five guided bonuses: complete a half-drawn digit, "draw me a 7", gray levels,
   full 28×28, and color. They live in [STRETCH.md](STRETCH.md).

---

## Part 1 — Tokenize (understand the representation)

`prepare.py` already builds the tokens. Open it and answer:

- **Q1.** What is `vocab_size`, and why that number?
- **Q2.** We binarize at threshold 128. What information is thrown away? When would that
  hurt (think ahead to color)?
- **Q3.** Images are flattened row-major and concatenated into one long stream; decoding does
  `.reshape(H, W)`. Why must the model's notion of *position* line up with this exact order?

**Do:** change `--threshold` or `--downscale`, re-run, and watch the printed digit change.
Build a feel for the representation *before* you train on it.

## Part 2 — Train (and the trap)

You'll write a short `train_image.py`. **Most of it is identical** to the char-GPT trainer
you already wrote — import the GPT, build the optimizer, loop, eval, save. The skeleton:

```python
import numpy as np, torch
from model import GPTConfig, GPT          # model.py is in this folder -- the SAME transformer, unchanged

# meta.pkl gives vocab_size and seq_len
block_size = seq_len - 1                  # predict pixel t+1 from pixels 0..t
model = GPT(GPTConfig(block_size=block_size, vocab_size=vocab_size,
                      n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False))
# ... optimizer + training loop: same shape as your char-GPT ...

def get_batch(split):
    # TODO: return x, y  (each shape (batch_size, block_size), dtype long)
    ...
```

**Your job is `get_batch`. And here's the assignment: first, do the obvious thing.**

You already wrote `get_batch` for text — pick a *random offset* into the token stream, take
`block_size` tokens as `x` and the next `block_size` as `y`. Do exactly that here. Train a
few hundred iterations, then generate (Part 3) and look at the result.

**It will look like noise. Stop and figure out why before reading the hints.**

<details>
<summary>Hint 1</summary>

What does sequence *position 0* mean for text? What does it mean for an image?
</details>

<details>
<summary>Hint 2</summary>

The GPT adds a **positional embedding indexed by where a token sits in the window**. If your
window starts at a random pixel, what does the model believe "position 0" is — and is that
consistent from one batch to the next?
</details>

<details>
<summary>The fix (only after you've tried)</summary>

For an image, **position = spatial location**: position 0 is the top-left pixel, *always*.
A random offset destroys that. So each training example must be **one whole image, aligned
to position 0**:

- reshape the stream into `(num_images, seq_len)`,
- sample whole **rows** (whole images),
- within an image, `x = pixels[:-1]`, `y = pixels[1:]`.

Notice the fix is entirely in the **data layer**. `model.py` never changes.
</details>

> **Pitfall.** `model.py` runs `targets.view(-1)`. If you build `y` by slicing an array, the
> result may be non-contiguous and crash. Make your batch contiguous. (Ask yourself: the
> traceback points at `model.py` — but where is the *bug*?)

**Expected once fixed:** val loss falls to ~**0.15** on 14×14; ~1500 iters, ~10–15 min on CPU.

## Part 3 — Generate (images *out*)

Write `sample_image.py`. **Reuse the model's own `generate()`** — the same call you used for
text. Then `reshape(H, W)` and print as ASCII (or save a PNG). Pick a sampling **temperature** —
~0.8 looks clean at 14×14 (lower trims speckle; go *too* low and a background-heavy image collapses
to all-blank — you'll hit exactly that at 28×28 in the stretch).

- **Q.** An image is `seq_len` pixels. How many tokens do you ask `generate()` to produce,
  and what do you **seed** it with? (Hint: what's almost always in MNIST's top-left corner?)
- **Then** run `prepare.py --downscale=1` to rebuild at full 28×28, retrain, and **time the
  generation.** Why is it so much slower? Attention is O(T²) and sampling is sequential
  (~O(T³)). **This is why we default to 14×14** — sequence length is the cost knob.

You should now be generating recognizable digits. That's the whole claim, proven by your own
hands: the Shakespeare transformer just drew a number.

---

## Stretch goals

Five guided bonus exercises live in **[STRETCH.md](STRETCH.md)**: complete a half-drawn digit
(image in → out), "draw me a *specific* digit" (conditioning), smoother strokes (gray levels),
crisper digits (full 28×28), and color (CIFAR-10). Each changes only the **tokens** or the
**prompt** — `model.py` never changes.

---

## You should be able to answer

1. Why does the *same* `model.py` work for both text and images?
2. Why can't you batch images the way you batch text? *(the deep one)*
3. How are "image in" and "image out" the same operation?
4. What makes sequence length the dominant cost?

## Working rules

- `model.py` stays unchanged.
- Keep your code small and readable — if a file gets clever, simplify it.
- **Verify by looking at the picture**, not just the loss number.
