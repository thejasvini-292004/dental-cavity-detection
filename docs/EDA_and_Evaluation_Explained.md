# EDA & Evaluation — What Each Cell Does and Why

A plain-English companion to the updated `google_colab.ipynb`. Read this once so
that in the interview you can explain any cell without hesitating.

---

## How the notebook is now organised

The notebook runs top to bottom in four blocks:

1. **Setup** — mount Drive, install packages, print versions.
2. **EDA** (cells 1–7) — runs *before* training, only *reads* the data. Every
   variable is prefixed `eda_` so it can't collide with or change anything in the
   training code.
3. **Training** — your original pipeline, unchanged in logic, just split into
   ten small cells (`T1`–`T10`) so each step reports on its own.
4. **Evaluation** (cells `E1`–`E6`) — added at the end; scores the trained model
   on the validation set.

Nothing in EDA feeds into training, so you can re-run or skip it freely. The only
thing added to the training code is one dictionary (`training_history`) that
records each epoch's metrics for the evaluation plots — it records numbers, it
does not change how the model trains.

---

## Part A — EDA, cell by cell

### Why EDA at all
Before choosing a model or a loss you need to know the shape of your data. For
this dataset EDA answers three questions that directly drive every later
decision: *Is the data clean and correctly paired? How imbalanced is it? What do
the cavities look like (how many, how big, where)?* The single most important
output is the class-imbalance number, because it is the reason the loss is
**focal + dice** and not plain cross-entropy.

### 1 · Dataset inventory & integrity
Counts images and masks per split and checks that every image has a mask with the
**exact same filename** (that is how this dataset pairs them). It also confirms
all patches are `128×128`, that mask pixels are essentially just `{0, 255}`, and
that **no filename appears in both train and val** (a basic leakage check).

*Why it matters:* silent data bugs — a missing mask, a stray size, an image that
sneaked into both splits — quietly wreck training and inflate scores. Catch them
here. **What you'll see:** 4,426 paired train, 730 paired val, zero overlap, all
`128×128`.

### 2 · Compute per-image statistics
Scans every patch once and stores, per image, its **mean intensity** (how bright
the X-ray is) and its **cavity-pixel fraction** (what share of the mask is
positive). It also runs connected-components on each mask to count **how many
separate cavities** the patch has and **how large** each one is. All of this is
kept in the `eda_stats` dictionary that the plots below draw from.

*Why it matters:* doing one pass and caching the arrays keeps every later plot
instant, and these four quantities are exactly what the distribution plots need.

### 3 · Class imbalance — the headline finding
Prints, per split, the overall cavity-pixel share, the average per-patch share,
how many masks are empty, and the background-to-foreground ratio.

*Why it matters:* this is the defining property of the dataset. Only **~2–4% of
pixels are cavity**, i.e. very roughly a **1 : 30 or worse** foreground-to-
background ratio. Two consequences you should say out loud in the interview:
(1) pixel accuracy is meaningless — a model that predicts "all background" scores
~97% while finding nothing; (2) this is precisely why the loss combines **focal**
(down-weights the easy background pixels) with **dice** (scores region overlap
directly and is immune to imbalance).

### 4 · Distribution plots
Four histograms: **(a)** cavity-pixel fraction per patch — piled up near zero,
the visual proof of imbalance; **(b)** mean image intensity per patch — the
brightness spread, which tells you whether contrast normalization (e.g. CLAHE)
is worth trying; **(c)** number of cavities per patch — usually one, sometimes
several (relevant because the demo app only boxes the largest); **(d)** individual
lesion area on a log scale — many lesions are tiny, which is what makes them hard
to segment.

*Why it matters:* it turns the single imbalance number into a full picture of the
data's shape, and each plot points at a concrete modelling choice.

### 5 · Where do cavities appear? (spatial heatmap)
Averages every training mask into one image, giving a **spatial prior** — a
heatmap of how often each pixel location is labelled cavity.

*Why it matters:* if the map is diffuse, cavities occur all over the patch and the
model must be translation-robust. A hot centre would mean patches were cropped
tightly around lesions — useful to know because it changes how the patch model
will behave on a full radiograph. **What you'll see:** a fairly spread-out map
(peak probability only ~0.07), i.e. cavities are not all stuck in the middle.

### 6 · Visual samples — image / mask / overlay
Shows a handful of patches (choosing ones with a clearly visible cavity, 2–10% of
the patch) as raw X-ray, mask, and mask tinted red over the image.

*Why it matters:* numbers can hide bad labels. Eyeballing overlays is the fastest
way to confirm the annotations actually sit on visible lesions.

### 7 · EDA summary
A compact recap of the facts to carry into modelling and the presentation, plus
the natural next steps the EDA exposes: patient-level (source-image) splitting,
adding cavity-free negatives, and trying contrast normalization.

---

## Part B — Evaluation, cell by cell

### Why these metrics (and not accuracy)
For imbalanced segmentation the right scores measure **overlap between the
predicted and true cavity regions**, not how many pixels you got right overall.
The evaluation runs on the **validation set** (`X_val`, `y_val`) — data the model
was scored on during training but never trained on — so the numbers are an honest
read of generalization.

The core metrics:

- **Dice coefficient** — `2·overlap / (predicted + truth)`. The standard
  segmentation score; equals the F1 of the pixels. Ranges 0–1, higher is better.
- **IoU (Jaccard)** — `overlap / union`. The other standard score; always a
  little lower than Dice. (`Dice = 2·IoU / (1 + IoU)`.)
- **Precision** — of the pixels you called cavity, how many really were. High
  precision = few false alarms.
- **Recall (sensitivity)** — of the true cavity pixels, how many you found. Low
  recall = you're missing cavities.
- **F1** — the balance of precision and recall.
- **Specificity** — how well you leave background alone.
- **Pixel accuracy** — reported only to show *why it's useless* here (it's ~97%+
  no matter what, because background dominates).

### E1 · Predict & define the metric helper
Runs the trained model over the validation set to get per-pixel probabilities
(`y_prob`), and defines `seg_scores`, which turns any binary prediction into all
the metrics above by counting true/false positives and negatives. Computing the
scores from raw counts (rather than the training-time Keras metric) makes them
transparent and lets us re-score at any threshold.

### E2 · Headline metrics @ 0.5 + threshold sweep
Prints all metrics at the usual 0.5 cutoff, then sweeps the threshold from 0.10
to 0.90 and reports the one that maximises Dice.

*Why it matters:* 0.5 is rarely optimal for imbalanced masks. If recall is low, a
**lower** threshold trades some precision for more recall and often lifts Dice.
The sweep both improves your reported score and shows you understand that the
decision threshold is a tunable knob, not a fixed constant. This is a **"good-fit"
evaluation** for this problem: it judges the model on region overlap and on the
precision/recall trade-off that actually matters clinically.

### E3 · Image-level detection rate
Zooms out from pixels to patches: a patch is "detected" if its predicted mask has
at least `MIN_AREA` positive pixels (the same noise filter the demo app uses).
Because every validation patch really does contain a cavity, this fraction is the
model's **patch-level sensitivity** — how often it catches a cavity that's there.

*Why it matters:* the end product's job is "flag patches with cavities," so this
is the metric closest to the product. It also honestly flags a dataset limitation:
there are **no cavity-free patches**, so false-positive rate / specificity at the
image level cannot be measured — which is exactly why "add negative patches" is
your top data fix.

### E4 · Pixel confusion matrix & threshold curve
Left: a 2×2 confusion matrix at 0.5 (as % of all pixels) — you'll literally see
the true-negative cell dominate, the visual of the imbalance. Right: Dice,
precision and recall plotted against threshold, with the best-Dice threshold
marked.

*Why it matters:* one glance explains both why accuracy is meaningless and how the
precision/recall balance shifts as you move the threshold.

### E5 · Training curves & over-fitting check
Plots train vs validation Dice (and loss) across epochs, using the
`training_history` recorded during the loop, and prints the final train–val Dice
gap.

*Why it matters:* a small, stable gap means a healthy fit; a big gap (train ≫ val)
means over-fitting. This is the cell that lets you answer "how do you know you
didn't over-fit?" with a chart instead of a shrug.

### E6 · Sample predictions
Shows val patches as X-ray / ground-truth / prediction side by side, and prints a
one-line final summary (Dice, IoU, precision, recall, F1).

*Why it matters:* qualitative proof to close on — and if you include a case where
the prediction misses a bit, it signals honesty, which interviewers reward.

---

## The thirty-second version

> "EDA first showed the data is clean and correctly paired but **severely
> imbalanced — only a few percent cavity pixels**, which is why I use focal+dice
> loss and report Dice/IoU rather than accuracy. After training I evaluate on the
> held-out validation set with **Dice, IoU, precision, recall and F1**, sweep the
> decision threshold to trade precision against recall, check patch-level
> sensitivity, and confirm a healthy train-vs-val gap on the learning curves."

---

## One caveat to state before they ask
The train/val split here is at the **patch** level, and several patches can come
from the same source X-ray, so patient-level leakage is possible. The clean fix is
to group the split by source image (or patient) so no two patches from the same
radiograph land on opposite sides. The EDA leakage check only catches *identical
filenames*, not same-patient patches — worth saying out loud.
