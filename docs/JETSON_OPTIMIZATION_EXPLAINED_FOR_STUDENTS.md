# Jetson Optimization, Explained From Scratch

A companion to [JETSON_ORIN_NANO_OPTIMIZATION_AUDIT.md](JETSON_ORIN_NANO_OPTIMIZATION_AUDIT.md).

The audit is the "what to do" list. **This document is the "why."** It assumes almost no
background. Every optimization in the audit is re-explained here as a small lesson:
*what the concept is → why the current code is slow → the fix → what changes and why it
helps.* If a word looks scary (synchronization, stream, FP16, throttling), it gets
defined the first time it appears.

> The golden rule of all of this: **the tracker must produce the exact same boxes after
> the change as before.** We are only making it *faster*, never *different*. Speed that
> changes the answer is a different kind of work (and the audit keeps those separate).

---

## 0. The cast of characters: CPU, GPU, and the Jetson

Think of the program as a **restaurant kitchen**.

- The **CPU** is the **head chef**. Very smart, makes decisions, but does one thing at a
  time. In our code the CPU runs the Python loop, reads files, does the camera-motion
  math, runs the bookkeeping.
- The **GPU** is a **huge brigade of line cooks** — hundreds of them. Each one is dumb on
  its own, but if you need the *same* operation done to a thousand pieces of data at once
  (like "multiply these millions of numbers"), the brigade finishes in a flash. In our
  code the GPU runs the neural networks (SiamABC, OSNet, YOLO).
- The **Jetson Orin Nano** is a *small* kitchen: a modest chef (a 6-core CPU that gets
  hot and slows down) and a small-but-real brigade (an Ampere GPU). It is not a giant
  workstation. So wasted effort hurts much more here than on a big PC.

A neural-network tracker is a back-and-forth dance: the chef preps an order (crop the
image, normalize it), hands it to the brigade (run the network), gets the result back,
makes a decision, preps the next order. **Most of our speed problems are about this
hand-off being clumsy.**

---

## 0.5 A report card: what this kitchen already does well

Before listing problems, it's only fair to say the kitchen is **already pretty well run**.
A beginner should learn to spot the *good* patterns too, because the goal is to keep them.
Here's the report card, in plain language (each "✓" links to the lesson that explains the
idea):

**Already good — don't break these:**

- ✓ **It pre-compiles the recipes.** The big networks are turned into fast TensorRT
  "engines" ahead of time and saved to disk, so they don't get re-figured-out every run
  (→ Lesson 4).
- ✓ **It uses fast math where it's safe and careful math where it matters.** The bulk of
  the network runs in quick low-precision FP16, but the final *scoring* stays in precise
  FP32 so the tracker's yes/no decisions don't wobble (→ Lesson 4).
- ✓ **It loads the appearance network only once** and reuses it everywhere (→ Lesson 7 —
  this is exactly the good habit that the *YOLO* part forgot to follow).
- ✓ **It already overlaps two GPU jobs** using a second "stream" so the appearance
  network and the main tracker can run at the same time (→ Lesson 3).
- ✓ **It does the cheap version of expensive things:** shrinks big frames before working
  on them, only computes appearance descriptors every other frame, and uses the cheaper
  camera-motion method by default.
- ✓ **It avoids obvious waste:** the appearance crops are prepared in one batched pass
  instead of a slow loop, and the "no video output" mode skips all the drawing.

**Where it wastes effort (the rest of this document):**

- ✗ The chef **stands around waiting for the disk** instead of reading the next frame
  early (→ Lesson 2).
- ✗ The chef **freezes the whole kitchen ~6 times per frame** to grab single numbers off
  the GPU (→ Lesson 1).
- ✗ It **rebuilds the YOLO detector for every video clip** instead of once (→ Lesson 7).
- ✗ It **leaves a free speed switch turned off** (`cudnn.benchmark`) and **carries a heavy
  toolbox it never opens** (`albumentations`) (→ Lessons 5 and 6).
- ✗ Its **packing list is written for a big PC, not the Jetson** (→ Lesson 8).

Notice the shape of the list: the good things are *clever algorithm choices*, and the
wasteful things are *clumsy hand-offs around those algorithms*. That's the whole theme —
**we fix the plumbing, not the recipes.**

---

## 1. Synchronization — the #1 villain

### What is it?

The chef and the brigade work **at the same time** on purpose. When the chef says "run
this network," they don't stand there watching — they hand over the ticket and *keep
working* on the next thing. The GPU works in the background. This is called
**asynchronous** ("async") execution, and it's what makes things fast.

**Synchronization** is the moment the chef stops and says: *"Brigade — STOP. I need the
exact result of that last dish, right now, in my hand, before I take one more step."*
The chef now **stands still and waits** until the GPU is completely finished and the
number is copied back.

In code, you cause a synchronization every time you pull a single value or array *out of*
the GPU and into normal Python/CPU land. The usual culprits:

```python
x.item()          # "give me this ONE number from the GPU, now"
x.cpu()           # "copy this whole tensor back to the CPU, now"
x.numpy()         # same idea
float(x)          # forces .item() under the hood
```

### Why is it bad?

Two reasons, and the second one is the real killer:

1. You wait for the GPU to finish (some idle time).
2. **Worse:** you destroy the overlap. While the chef is frozen waiting, they *cannot*
   start prepping the next order. So the next network call starts later than it had to.
   The whole pipeline goes "lockstep": CPU works, *everything stops*, GPU works,
   *everything stops*, CPU works… instead of the two overlapping.

If you do this 6 times per frame, you've put 6 little "everybody freeze" moments into
every single frame. At 30 frames per second that's 180 freezes per second of video.

> **Jetson twist:** on a normal PC, the GPU is a separate card and copying data back also
> costs time crossing a cable. The Orin Nano has **unified memory** (CPU and GPU literally
> share the same RAM — one fridge, not two), so the *copy* is cheap. But the *freeze* is
> still expensive, because it still forces the chef and brigade into lockstep. So on
> Jetson we remove syncs to kill the **stalls**, not to save copy time.

### Where it happens in our code

In the audit this is the cluster of "~5–6 GPU→CPU syncs per frame." A couple of examples:

**Example A — the TTA switch** ([siamabc.py:489](../models/SiamABC/tracker/trt_engine/siamabc.py#L489)):

```python
# BEFORE — every frame, freeze the pipeline to read one number off the GPU
bbox_pred, cls_pred = _dispatch_connect(
    self._connect_engines,
    lam_val=float(lam.item()),   # <-- synchronization!
    ...
)
```

Here `lam` is a tiny on/off knob ("is test-time augmentation on?"). The code reads it
*back from the GPU* every frame just to decide which engine to call. But we *already know*
the answer in plain Python — it was set by `set_tta()`. We're asking the GPU a question
we already know the answer to, and freezing to hear it.

```python
# AFTER — remember the value in plain Python when it's set; never ask the GPU
def set_tta(self, enabled):
    self._tta_lam.fill_(self._norm_lambda_tta if enabled else 0.0)
    self._tta_on = enabled            # <-- cache it on the CPU side

# ...in track():
lam_val = self._norm_lambda_tta if self._tta_on else 0.0   # no GPU read, no freeze
```

**Example B — finding the best grid cell** ([box_coder.py:397-398](../utils/box_coder.py#L397-L398)):

```python
# BEFORE — TWO freezes to turn one index into a (row, col)
flat_idx = torch.argmax(cls_map)
r_max = (flat_idx // W).item()   # freeze 1
c_max = (flat_idx % W).item()    # freeze 2
```

```python
# AFTER — ONE freeze, then do the cheap math in Python
flat = int(flat_idx.item())      # one freeze
r_max, c_max = divmod(flat, W)   # CPU arithmetic, free
```

Same numbers come out. We just stopped freezing twice when once would do.

### The bigger version of the same idea

The strongest version (audit item #4) is to compute the winning cell, its four box edges,
and its score **all on the GPU**, then copy that tiny result to the CPU **one time** — instead
of copying three or four separate maps back individually. Picture it as: instead of the
chef walking to the brigade six times to grab six things, the brigade puts all six things
on one plate and the chef makes a single trip.

**Before:** ~6 round trips per frame. **After:** ~1. The math is identical; only the number
of "everybody freeze" moments drops.

---

## 2. Overlapping the file reading (the biggest *free* win)

### The concept: don't wait for things you could prepare in advance

Right now, every frame, the program does this ([vis/test_model.py:376](../vis/test_model.py#L376)):

```
read frame from disk  →  track it on the GPU  →  read next frame  →  track it  → ...
   (CPU, ~5-15ms)          (GPU pipeline)           (CPU again)
```

Reading a frame means **decoding a JPEG** — turning a compressed file on disk into a grid
of pixels. That's real CPU work. And notice: while the CPU is decoding, **the GPU is
sitting idle.** While the GPU is tracking, **the disk/decoder is idle.** They take turns
when they could be working at the same time.

This is a waiter who takes one order, walks to the kitchen, **stands there until the food
is cooked**, serves it, and only *then* walks back to take the next table's order. The
kitchen and the dining room are never busy at the same time.

### The fix: a "prefetch" / "double-buffering" helper

Hire a second person whose only job is to **read the next frame while the current one is
being tracked**. By the time the GPU finishes frame N, frame N+1 is already decoded and
waiting.

```
reader thread:   decode N    decode N+1    decode N+2   ...   (runs continuously)
main thread:               track N        track N+1    track N+2
                           (GPU busy)     (GPU busy)
```

In code this is a tiny **producer/consumer**: a background thread (the *producer*) puts
decoded frames into a small queue; the main loop (the *consumer*) pulls them out. The
order of frames is unchanged, so the tracker sees exactly the same input → **identical
output.** You just stopped making the GPU wait for the disk.

### Why this is *the* win on Jetson specifically

The Orin Nano's CPU is the weak link, and JPEG decode is pure CPU. On a big workstation
the CPU is so fast you never notice this. On the Nano, decode can be a big slice of each
frame's time — and it's completely hidden by overlapping it with the GPU work you're
already doing. Best kind of optimization: **free real estate you already paid for.**

> This is also why the audit insists the profiler must *measure decode time separately*.
> If you don't, you can spend a week "speeding up the GPU" while a hidden `imdecode`
> quietly eats half your frame budget.

---

## 3. Streams — letting two GPU jobs overlap

### What is a CUDA "stream"?

A **stream** is a *queue of work for the GPU*. By default everything goes into one queue,
so GPU jobs run one after another even if they don't depend on each other.

If you make a **second stream**, two independent GPU jobs can run at the same time (the
brigade is big enough to split into two teams). This project already uses this trick:
OSNet (the appearance-descriptor network) runs on a *side* stream so that while it's
computing the descriptor for frame N, the main SiamABC network can already be working on
frame N+1 ([the `osnet_async_overlap` setting](../config/inference_config_experimental.yaml)).

You don't need to *add* this — it's a good example of the pattern done right, and it's why
removing the `lam.item()` sync (Section 1) matters even more: a stray synchronization
**cancels** the overlap the streams were giving you, because a sync waits for *all* GPU
work to drain.

---

## 4. TensorRT, FP16, FP32 — "compiling" the network and "how many decimals"

### TensorRT = optimizing the recipe ahead of time

Normally PyTorch runs a network by interpreting it step by step, deciding what to do as it
goes — like a cook improvising from a recipe each time. **TensorRT** takes the network
once, ahead of time, and **compiles** it into a single highly-optimized program tailored
to your exact GPU: it fuses steps together, picks the fastest math kernels, and removes
overhead. The result is a fixed, fast "engine."

That's why the first run is slow ("compiling… 1–3 min") and later runs are fast — and why
the engine is saved to a cache file (`.ts`) so you only compile once. **Important:** an
engine is built for a *specific* GPU. An engine compiled on your PC will not work on the
Jetson; the Jetson must compile its own. (The code already fingerprints the GPU so it
rebuilds correctly — see the audit.)

### FP16 vs FP32 = precision vs speed

Computers store decimal numbers with a fixed number of bits:

- **FP32** ("single precision") = 32 bits per number ≈ ~7 significant digits. Accurate.
- **FP16** ("half precision") = 16 bits ≈ ~3 digits. Less accurate, but **half the memory
  and often 2× the speed** on GPUs with Tensor Cores (the Orin has them).

This project runs the heavy backbone in **FP16** (fast) but deliberately keeps the final
*scoring* head in **FP32**. Why the split? The score head produces confidence values that
get compared against hard thresholds (e.g. "is the score below 0.55? → the target is
occluded"). In FP16, a true score of 0.561 might round to 0.557 or 0.564 — and a borderline
case could flip to the wrong side of the threshold, making the tracker hallucinate an
occlusion. FP32 keeps those decisions stable. This is a great example of **"use the fast
low-precision math everywhere it's safe, and the slow precise math only where it matters."**

> This is also why "just make everything FP16" is in the audit's *behavior-changing*
> bucket, not the free-wins bucket. It can change the boxes.

---

## 5. `cudnn.benchmark` — let the GPU find its own fastest moves

### The concept

For a given operation (say, a convolution on a 320×320 image), there are *many* algorithms
that all give the same answer but run at different speeds depending on the exact shapes and
the exact GPU. cuDNN (NVIDIA's neural-net math library) can either guess a decent one, or —
if you let it — **try several on the first few runs, time them, and then always use the
fastest.** That "try and remember the winner" mode is `torch.backends.cudnn.benchmark = True`.

It's like a new cook trying three ways to dice an onion on day one, timing each, then using
the fastest method for the rest of the year.

### Why it's safe and why we want it here

- **Safe:** every candidate algorithm computes the *same result*. Only the speed differs.
  The boxes don't change. (One subtlety: it only helps when the input **shapes stay the
  same** frame to frame — which is exactly our case: YOLO always sees 320×320, the
  attention always sees two fixed sizes.)
- **We want it** because the code never turns it on (the audit checked). It's a single line
  at startup. Free speed on the parts of the pipeline that aren't already TensorRT engines
  (notably the eager YOLO).

```python
import torch
torch.backends.cudnn.benchmark = True   # one line, at program start
```

The first couple of frames are a hair slower (it's timing options); everything after is
faster. That's why you always **measure with a warmup** and ignore the first few frames.

---

## 6. Dead code on the hot path — the `albumentations` toolbox you never open

### The concept

Importing a Python library has a cost: the moment you write `import albumentations`, Python
loads that library and everything it depends on. Some libraries are huge. If you import a
big one **but never actually use it**, you pay the loading cost (slower startup, more
memory, one more package that must install correctly on the Jetson) for nothing.

### What's happening here

The base tracker imports `albumentations` and builds three image-normalization "transforms"
at startup. But the function that's supposed to use them
([_preprocess_image](../models/SiamABC/tracker/base_tracker.py#L214-L222)) **ignores them**
and does the normalization itself on the GPU instead:

```python
def _preprocess_image(self, image, transform=None):   # 'transform' is handed in...
    x = torch.from_numpy(image[:, :, :3]).permute(2,0,1).unsqueeze(0).float()
    x = x.to(self.cuda_id).div_(255.0)
    return x.sub_(self._norm_mean).div_(self._norm_std)   # ...but never used; GPU does it
```

So the transforms — and the whole `albumentations` import — are **dead weight**. They run
once at init, produce objects nobody calls, and sit there.

### The fix and why it's risk-free

Delete the unused import and the unused transform objects. Because the code already does
the normalization a different way, **removing the dead path cannot change any output** — it
was never on the path that produces the boxes. The payoff: faster cold start and one fewer
heavy dependency to wrestle onto the Jetson. (This is the safest kind of change: deleting
something that was already doing nothing.)

---

## 7. Don't rebuild the same thing over and over — the YOLO reload

### The concept: building a model is expensive; do it once

Loading a neural network (reading weights from disk, putting them on the GPU, warming it
up) takes real time. If you process 100 video clips and you **rebuild the detector for
every clip**, you pay that cost 100 times — even though it's the same detector every time.

### What's happening here

Unless you pass `--reuse_tracker`, the program makes a *fresh* tracker wrapper for each
clip ([run_inference.py:1570](../run_inference.py#L1570)), and each fresh wrapper loads
YOLO again in its constructor ([tracker.py:390](../models/siamram/tracker.py#L390)).

Why does it make a fresh wrapper at all? For **safety** — so that leftover state from clip
A (where the target was, what it looked like) can't leak into clip B and corrupt it. That's
a good instinct. The problem is it throws away the *expensive, stateless* YOLO model along
with the *cheap, stateful* tracking memory.

### The fix: separate "expensive and shared" from "cheap and per-clip"

Load YOLO **once**, keep it in a cache, and **hand the same YOLO object** to each fresh
wrapper. The per-clip tracking state still gets reset (safety preserved); only the heavy,
identical-every-time model stops being rebuilt.

> Note the audit's nuance: the *other* heavy model, OSNet, is **already** loaded once and
> shared (it's a global singleton). So only YOLO has this problem. Good code-reading
> practice: verify which things actually reload before "fixing" all of them.

This is a general principle worth remembering: **construct expensive, reusable things once;
reset only the small things that must be fresh.**

---

## 8. Hardware facts that change the plan (Jetson-specific)

### Unified memory (already met in Section 1)

CPU and GPU share one pool of RAM. Consequence: copying data between them is cheap, so
don't bother with advanced "pinned memory / zero-copy" tricks here — they solve a problem
(slow CPU↔GPU transfer over a cable) that the Jetson doesn't have. Spend your effort on
removing **syncs** (stalls), not copies.

### The Orin Nano has **no DLA**

Some Jetsons have a **DLA** (Deep Learning Accelerator) — a separate little chip that can
run neural networks to offload the GPU. The bigger Orins (NX, AGX) have it. **The Orin
Nano does not.** So any blog post telling you to "put your engine on the DLA to go faster"
does not apply — there's no DLA to use. Knowing this saves you from a dead-end rabbit hole.

### It gets hot, and hot means slow (thermal throttling)

When a chip gets too hot it **protects itself by slowing down** — this is *thermal
throttling*. A Jetson running flat-out with poor cooling will quietly drop its clock speed,
and your "benchmark" becomes meaningless because the hardware was sandbagging.

Two consequences:
- **Always set max performance before measuring:** `sudo nvpmodel -m 0` (max power mode)
  and `sudo jetson_clocks` (lock clocks high), and put a fan on it.
- **Watch the temperature while you profile** (`tegrastats`/`jtop`). If your numbers wobble,
  check whether the clocks drooped before you blame your code.

### 4 GB vs 8 GB

The Nano comes in 4 GB and 8 GB versions, and that RAM is shared by *everything* (CPU
program + GPU models, remember — unified memory). On a 4 GB board you're tight. The part of
our code most likely to blow the budget is building the **entire results table in memory**
with pandas at the end of a long run ([run_inference.py:1688](../run_inference.py#L1688));
writing rows out incrementally with the standard `csv` module avoids a big memory spike.
Adding some swap/zram is also cheap insurance.

---

## 9. How to know your optimization actually worked (measurement discipline)

This matters as much as the changes themselves. A beginner mistake is to "optimize" and
*assume* it got faster.

1. **Measure on the Jetson, not your PC.** Different CPU, different GPU, different RAM. PC
   numbers do not transfer.
2. **Warm up first.** The first few frames pay one-time costs (TensorRT lazy init,
   `cudnn.benchmark` trying algorithms). Throw them away; measure the steady state.
3. **Build the TensorRT cache once, then measure the warm run.** Otherwise you're timing a
   3-minute compile, not the tracker.
4. **Measure each stage separately** — decode, camera motion, SiamABC, OSNet, YOLO. If you
   lump them together you can't see that (for example) decode is the real bottleneck. This
   is *exactly* how the "decode is serialized" problem stays invisible.
5. **Prove it's still correct.** Keep a saved "golden" set of output boxes from before your
   change, and diff against it after. "Behavior-preserving" is a claim you must **verify**,
   not hope for. If the boxes match and the clock is lower, you win. If the boxes changed,
   it's not a free optimization anymore — it's a different thing that needs accuracy testing.

---

## 10. One-paragraph summary you could explain to a friend

The tracker is a chef (CPU) and a brigade of cooks (GPU) working together. Most of our
easy speed-ups are about **stopping the chef from standing around waiting**: don't freeze
the whole kitchen to read one number off the GPU (*synchronization*); read the next frame
from disk *while* the GPU is busy instead of after (*prefetch*); don't rebuild the same
detector for every video (*caching*); and don't carry a giant toolbox you never open
(*dead imports*). Let the GPU pick its own fastest moves (`cudnn.benchmark`). None of these
change the *answer* — they just remove waiting. The riskier knobs that trade accuracy for
speed (FP16 everywhere, INT8, shrinking the model) are kept separate, because on a tracker,
a slightly different number can flip a decision and change what it does. And before trusting
any result: max out the Jetson's clocks, warm it up, measure each stage, and diff the boxes
to prove you only changed the speed — not the behavior.
