# 02 — Bounding Boxes & Image Geometry

> Prerequisites: doc 01 (norms, the exponential, clamping). This document makes "a box on an image" mathematically precise, defines the overlap measure (IoU) used pervasively, explains how the tracker turns a box into a fixed‑size network input (the *crop* and its *context*), and lays out the **coordinate spaces** that every later subsystem implicitly lives in. Get this right and the rest of the system stops being mysterious — most of SiamRAM is just careful bookkeeping in these coordinate spaces.

---

## 1. The image as a coordinate grid

A digital color image is a 3‑D array of numbers with shape $H\times W\times 3$: $H$ rows (height), $W$ columns (width), and $3$ color channels. Channel order in this codebase is **BGR** (blue, green, red) because it uses OpenCV; that detail rarely matters for the geometry.

We address a pixel by **(x, y)** where $x\in\{0,\dots,W-1\}$ is the column (horizontal) and $y\in\{0,\dots,H-1\}$ is the row (vertical). Crucially, in images **$y$ increases downward** and the origin $(0,0)$ is the **top‑left** corner:

```
 (0,0) ───────────────► x  (columns, →)
   │  ┌───────────────────────────┐
   │  │                           │
   │  │      • (x, y)             │
   ▼  │                           │
   y  │                           │
(rows,│                           │
  ↓)  └───────────────────────────┘
                              (W-1, H-1)
```

This "down‑is‑positive‑$y$" convention is just a labeling choice, but it must be consistent everywhere, and it is why a box's bottom edge has a *larger* $y$ than its top edge.

---

## 2. Two ways to write a box

An **axis‑aligned bounding box** is a rectangle with horizontal and vertical sides. SiamRAM uses two equivalent encodings, and converting between them is a constant chore worth internalizing.

**(A) Corner‑plus‑size, `xywh`** — the dominant format in this codebase:

$$
b = (x,\, y,\, w,\, h),
$$

where $(x, y)$ is the **top‑left corner** and $w, h$ are the width and height (both $> 0$). The box covers columns $[x,\, x+w)$ and rows $[y,\, y+h)$.

**(B) Two‑corner, `xyxy`:**

$$
b = (x_1,\, y_1,\, x_2,\, y_2),
$$

the top‑left $(x_1,y_1)$ and bottom‑right $(x_2,y_2)$ corners.

**Conversions** (memorize these — bugs hide in getting them wrong):

$$
\begin{aligned}
\text{xywh}\to\text{xyxy}:&\quad x_1=x,\ \ y_1=y,\ \ x_2=x+w,\ \ y_2=y+h,\\[2pt]
\text{xyxy}\to\text{xywh}:&\quad x=x_1,\ \ y=y_1,\ \ w=x_2-x_1,\ \ h=y_2-y_1.
\end{aligned}
$$

**Derived quantities** used everywhere:

$$
\text{center } (c_x, c_y) = \Big(x + \tfrac{w}{2},\ y + \tfrac{h}{2}\Big),
\qquad
\text{area } = w\,h,
\qquad
\text{diagonal } d = \sqrt{w^2 + h^2}.
$$

The **diagonal** $d$ is especially important: it is the natural "size unit" of the box. SiamRAM repeatedly measures displacements *in units of the box diagonal* so that the same physical event (e.g. "the center moved by half a box") means the same thing for a tiny target and a huge one. This is called a **scale‑normalized** measure, and you'll see $\big/\, d$ all over docs 06–09.

```
        w
   ┌──────────┐
   │          │
 h │    •─────┼── center (cx, cy) = (x+w/2, y+h/2)
   │  (cx,cy) │            diagonal d = √(w²+h²)  ◄── the "size unit"
   └──────────┘
 (x,y) top-left
```

---

## 3. Intersection over Union (IoU): the overlap measure

How much do two boxes overlap? The standard answer is **Intersection over Union (IoU)**, also called the *Jaccard index*. It is used as a gate in many places: admitting a frame to memory (doc 07), checking a re‑detection candidate (doc 08), and scoring motion consistency (doc 06).

Take two boxes $A$ and $B$. Their **intersection** is the overlapping rectangle; its area is

$$
\text{inter} = \max(0,\ \min(x_2^A, x_2^B) - \max(x_1^A, x_1^B))\ \times\ \max(0,\ \min(y_2^A, y_2^B) - \max(y_1^A, y_1^B)).
$$

The two $\max(0,\cdot)$ guards are essential: if the boxes don't overlap, one of the differences is negative, and clamping it to $0$ gives zero intersection (rather than a spurious negative "area"). The **union** is the total area both cover, counting the overlap once:

$$
\text{union} = \text{area}(A) + \text{area}(B) - \text{inter}.
$$

Then

$$
\boxed{\ \operatorname{IoU}(A,B) = \dfrac{\text{inter}}{\text{union}}\ \in [0,1].\ }
$$

$\operatorname{IoU}=1$ means the boxes are identical; $\operatorname{IoU}=0$ means they are disjoint. As a guard against the degenerate case where both boxes have zero area (union $=0$), the code returns $0$.

```
   A ┌──────────┐
     │      ┌───┼──────┐ B          inter = area of the shaded overlap
     │   ▓▓▓│▓▓▓│      │            union = area(A) + area(B) − inter
     │   ▓▓▓│▓▓▓│      │            IoU   = inter / union
     └──────┼───┘      │
            └──────────┘
```

**Why IoU and not center distance?** IoU jointly captures *position* and *size* agreement: two boxes can have the same center but very different sizes and still get a low IoU. It is also dimensionless (a ratio), so it needs no scale normalization. Its weakness — it is exactly $0$ for any pair of non‑overlapping boxes, giving no gradient toward "close but not touching" — is why SiamRAM *also* uses center‑distance‑in‑diagonals in places where it needs to reason about boxes that don't overlap (e.g. a target that jumped far away).

---

## 4. From a box to a network input: crops and context

SiamABC (doc 03) is a neural network. It does not consume the whole frame; it consumes **fixed‑size square crops**. Turning a variable‑size, variable‑shape box into a fixed square crop is a small but important geometric pipeline. Three ideas: *context expansion*, the *equal‑context square*, and *resize + map‑back*.

### 4.1 Why context, not a tight crop

If you cropped *exactly* the target box and showed it to the network, the network would lose all information about the object's *boundary* against its surroundings — and that boundary is a strong cue for where the object ends. So the crop always includes a margin of surrounding pixels, called **context**.

The simplest form is a multiplicative expansion. With a context factor $\rho$ (e.g. $\rho \approx 0.5$ means "add 50% of the box size as margin"), `extend_bbox` grows the box outward:

$$
\text{expanded box} \approx \big(x - \rho\,\tfrac{w}{2},\ \ y - \rho\,\tfrac{h}{2},\ \ (1+\rho)\,w,\ \ (1+\rho)\,h\big),
$$

(the exact per‑side arithmetic can be configured, but this is the idea). The expanded box can poke outside the frame; that's fine — the cropper pads the missing region (§4.4).

### 4.2 The equal‑context square `squared_size`

Real objects are rectangles, but CNN backbones want **squares** (so the output grid is square and scale is isotropic). We must *not* stretch a rectangle into a square — that distorts appearance and would change between frames. Instead we compute the side of a square that contains the object *plus a consistent amount of context*. SiamRAM's `squared_size` uses the classic SiamFC/SiamRPN rule: pad each dimension by the average half‑perimeter and take the geometric mean.

$$
\text{pad} = \tfrac12 (w + h),\qquad
\text{side} = \sqrt{(w+\text{pad})\,(h+\text{pad})}.
$$

Let's sanity‑check the design. Substituting $\text{pad}=\tfrac12(w+h)$:

$$
\text{side} = \sqrt{\Big(w + \tfrac{w+h}{2}\Big)\Big(h + \tfrac{w+h}{2}\Big)} = \sqrt{\Big(\tfrac{3w+h}{2}\Big)\Big(\tfrac{w+3h}{2}\Big)}.
$$

For a square object ($w=h$) this gives $\text{side}=2w$ — the crop is twice the object, i.e. the object occupies the central quarter of the area, with a uniform context ring. For a long thin object the formula still produces a single square side that contains it with proportional context. The key property: **the fraction of the crop occupied by the object is roughly constant** regardless of the object's aspect ratio, so the network sees a consistent "object‑to‑context" ratio every frame. That consistency is what lets one trained network handle wildly different object shapes.

### 4.3 Template vs. search crops

SiamABC takes *two kinds* of crops, with *different* context factors (doc 03 uses them):

- **Template crop** — centered on the target, with a *small* context (a `template_bbox_offset`). It answers "what does the target look like?" Resized to a small square (e.g. $64\times64$).
- **Search crop** — centered on the *predicted* location, with a *larger* context (`search_context`). It answers "where, in this neighborhood, is the target now?" Resized to a larger square (the `instance_size`, e.g. $256\times256$).

The search crop is bigger (in context) so the target can *move* between frames and still land inside it. This is the single most important geometric constraint in the whole tracker:

> **If the target moves outside the search crop, the matcher physically cannot find it.** Every motion prior (GMC in doc 06, the Kalman‑centered search in doc 06, the EKF‑guided occlusion search in doc 08) exists to keep the crop centered where the target actually is.

### 4.4 Resize, padding, and mapping back

Once the equal‑context square is chosen, the cropper (`get_extended_crop`) does three things:

1. **Crop** the square region from the frame. If part of it lies outside the frame (common near edges or during occlusion when the predicted center is near a border), the missing pixels are **padded** with a constant — SiamRAM uses the frame's mean color, so the padding is visually neutral and doesn't create a fake edge.
2. **Resize** the (possibly padded) square to the fixed crop size $S\times S$ (e.g. $256\times256$) the network expects, using interpolation.
3. **Record the mapping** needed to invert step 2 later. The network's predicted box comes out in *crop pixels* $[0,S)$; we must convert it back to *frame pixels*. If the crop came from a region of frame‑width $W_{\text{ctx}}$ and frame‑top‑left $(x_0, y_0)$, then a crop‑coordinate $x_{\text{crop}}$ maps back to the frame by

$$
x_{\text{frame}} = x_0 + x_{\text{crop}}\cdot\frac{W_{\text{ctx}}}{S},
\qquad
y_{\text{frame}} = y_0 + y_{\text{crop}}\cdot\frac{H_{\text{ctx}}}{S},
$$

and a predicted *width* scales by the same factor $W_{\text{ctx}}/S$. This is exactly what `_rescale_bbox` computes: the scale factor is `padded_box[2] / instance_size` for $x$/width and `padded_box[3] / instance_size` for $y$/height, then it shifts by the crop's top‑left. Forgetting this map‑back, or getting the scale wrong, is the classic tracker bug where the box drifts or shrinks for no visible reason.

### 4.5 Preprocessing the pixels

Before the network sees a crop, its pixel values are **normalized**. Each channel is divided by $255$ (to bring it into $[0,1]$) and then standardized with the ImageNet statistics

$$
x' = \frac{x/255 - \mu_c}{\sigma_c},\qquad \boldsymbol\mu = (0.485, 0.456, 0.406),\ \ \boldsymbol\sigma = (0.229, 0.224, 0.225),
$$

one $(\mu_c,\sigma_c)$ per RGB channel. This is the input distribution the backbone was trained on; matching it at inference is required for the features to be meaningful. It is bookkeeping, but it is *load‑bearing* bookkeeping.

---

## 5. The score grid: where the network "looks"

SiamABC's head does not output a single box; it outputs **dense maps** over a coarse grid (doc 03 explains the maps). If the search crop is $S\times S = 256\times256$ pixels and the backbone has total **stride** $r = 16$ (it downsamples by $16\times$), the output grid is

$$
G = \frac{S}{r} = \frac{256}{16} = 16,
$$

so a $16\times16$ grid of cells. Each cell corresponds to a $16\times16$‑pixel patch of the crop. We need a fixed lookup: *which crop pixel does grid cell $(i,j)$ correspond to?* That is `make_grid`. With cell indices $i,j \in \{0,\dots,G-1\}$, it places the grid centered on the crop:

$$
\text{grid\_x}(i,j) = \Big(j - \big\lfloor\tfrac{G}{2}\big\rfloor\Big)\cdot r + \tfrac{S}{2},
\qquad
\text{grid\_y}(i,j) = \Big(i - \big\lfloor\tfrac{G}{2}\big\rfloor\Big)\cdot r + \tfrac{S}{2}.
$$

Read it as: take the cell's offset from the grid center $\lfloor G/2\rfloor$, scale by the stride $r$ to convert "cells" into "pixels," then shift so the grid center sits at the crop center $S/2$. The grid never changes during tracking, so it is computed once and reused — a small but real speedup at $30+$ frames per second. Doc 03 uses `grid_x, grid_y` to turn the network's per‑cell edge‑distances into actual pixel boxes.

```
 16×16 score grid laid over the 256×256 search crop (stride 16):

   crop pixel 0      128 (=S/2)        255
        ┌─────────────┬─────────────┐
        │ cell(0,0)   │             │      grid_x(i,j) = (j − 8)·16 + 128
        │   ↕ 16px    │             │      grid_y(i,j) = (i − 8)·16 + 128
        ├─────────────•─────────────┤
        │           center cell     │      • = crop center maps to grid center
        │             │             │
        └─────────────┴─────────────┘
```

---

## 6. The two coordinate spaces of SiamRAM

Finally, the big‑picture geometry. There are **three** coordinate frames in play, and every quantity belongs to exactly one of them. Confusing them is the most common source of error in tracker code, so we name them explicitly.

1. **Full‑resolution frame space.** The original video pixels, possibly very large (e.g. $1920\times1080$). The *input* box on frame 0 and the *output* box every frame live here.

2. **Proc (processing) space.** At the top of every update, the frame is **prescaled** so its long edge does not exceed a cap (e.g. $1280$). Let the scale factor be $s \le 1$. A full‑res point $(x,y)$ becomes a proc point $(sx, sy)$; a full‑res box scales all four numbers by $s$. **Every internal subsystem — homography, EKF, SiamABC, YOLO, memory, frame dynamics — works entirely in proc space.** This is deliberate: it bounds the cost of the expensive operations and, more importantly, makes every subsystem speak the same units (a "pixel" means the same thing to all of them). Only at the very end is the chosen box scaled **back** to full‑res by $1/s$.

3. **Crop space.** Inside SiamABC, the search crop is its own $S\times S$ coordinate system (§4.4, §5). The network predicts in crop space; `_rescale_bbox` maps the prediction back to proc space.

The data flow of coordinate spaces in one frame:

```
 full‑res box (frame t−1)
      │  × s         (prescale)
      ▼
 proc space  ──────────────────────────────────────────────► proc box (frame t)
      │  crop+resize          ▲  _rescale_bbox (× W_ctx/S, + top‑left)
      ▼                       │
 crop space ──► SiamABC ──► crop‑space prediction
                                                              │  × 1/s  (un‑prescale)
                                                              ▼
                                                         full‑res output box (frame t)
```

When later documents say "the center moved 12 pixels," or "the box diagonal is 80," they mean **proc pixels**. When they say "scale the box back," they mean the final $\times\,1/s$ step. Holding these three spaces straight makes the entire codebase legible.

---

## 7. Recap

- A box is `(x,y,w,h)` or `(x1,y1,x2,y2)`; center is $(x+w/2,\,y+h/2)$; the diagonal $\sqrt{w^2+h^2}$ is the natural size unit, and SiamRAM normalizes displacements by it.
- IoU $=\dfrac{\text{intersection}}{\text{union}}\in[0,1]$ measures position‑and‑size overlap; it is $0$ for disjoint boxes, which is why center‑distance‑in‑diagonals is used as a complement.
- The network consumes fixed square crops with *context*; `squared_size` makes the object‑to‑context ratio roughly constant across aspect ratios; the search crop is larger so the target can move and still be found.
- Predictions come out in crop space and must be mapped back with the recorded scale and offset.
- Three coordinate spaces — full‑res, proc, crop — and everything internal lives in **proc** space.

Next: **`03_SIAMESE_TRACKING_SIAMABC.md`**, where these crops become the input to the visual matcher and we follow the math from two images to one box.
