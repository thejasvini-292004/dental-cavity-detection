# Dental Cavity Detection — Complete Study Guide

A deep, code-linked explanation of the whole project so you can explain and defend
every part in the interview. Read it once end-to-end, then re-skim the code boxes.

**One-line summary of the project:**
> I built a U-Net that does pixel-level segmentation of dental caries (cavities) on
> X-ray patches, trained it with a focal + dice loss to survive severe class
> imbalance, evaluated it honestly on a held-out validation set (where I found and
> diagnosed overfitting), and wrapped it in a small OpenCV/Tkinter desktop app that
> draws a bounding box around the detected cavity.

**The honest headline numbers you must know:**
Train Dice **0.81**, validation Dice **≈ 0.21–0.25**, precision **0.51**, recall
**0.16**, IoU **0.14**, pixel accuracy **97.9%**, patch-level sensitivity **43.7%**.
The large train-vs-val gap (**+0.60**) means the model **overfit** — this is a
finding you own, not a number you hide.

---

## 1 · Problem statement — what am I solving, and where is it useful?

**The task.** Given a dental X-ray, automatically find *where* the cavities are, at
the pixel level. This is **semantic segmentation** (label every pixel as
cavity / not-cavity), not plain classification (one yes/no label per image).

**Why segmentation and not classification.** A classifier that says "cavity: yes"
gives a dentist nothing to verify. Segmentation produces a *mask* — it points at the
exact region — which a clinician can look at and agree or disagree with. That
transparency is what makes it usable as decision support.

**Why it matters clinically.** Early caries are small, low-contrast smudges on a
busy radiograph. They are easy to miss, and a missed cavity becomes a bigger, more
expensive problem later. A model that flags suspicious regions acts as a **second
pair of eyes**.

**Scenarios where this is useful:**
- **Screening assistant** in a dental clinic — pre-flags regions for the dentist to review.
- **Tele-dentistry / low-resource settings** where a specialist isn't on site.
- **Training aid** for dental students learning to read radiographs.
- **Triage** in high-volume settings — sort scans that likely need attention.

**Important framing to say out loud:** this is **decision support, not diagnosis**.
A dentist stays in the loop; the model never decides treatment.

---

## 2 · The data — source, numbers, and what every EDA output means

### 2.1 Where the data comes from
The images are **dental radiograph patches** annotated for caries. The filenames
carry a `.rf.<hash>` signature, which is the export signature of **Roboflow** (a
dataset-annotation platform) — so this is a publicly sourced, Roboflow-managed
caries-segmentation dataset. Full radiographs were **cropped into 128×128 patches**
around lesions, and each patch has a matching **binary mask** (white = cavity,
black = background) with the **same filename**.

> **Note to state proactively:** the original filenames contain patient names and
> scan dates. Real deployment would require de-identification (HIPAA/GDPR). Raising
> this yourself shows maturity about healthcare data.

### 2.2 The headline numbers

| Quantity | Train | Val |
|---|---|---|
| Image–mask pairs | **4,426** | **730** |
| Patch size | 128×128 grayscale | 128×128 grayscale |
| Missing masks | 0 | 0 |
| Filename leakage across splits | **0** (disjoint) | — |
| Mean cavity-pixel share | **3.02%** | **2.15%** |
| Background : cavity ratio | **32 : 1** | **46 : 1** |
| Empty masks (no cavity) | 0 | 0 |

Total ≈ **5,156 patches**, split roughly **86% / 14%**.

**The code that produced these** (EDA cells 1–3 in the notebook):

```python
# inventory + pairing + leakage
for split, (img_dir, msk_dir) in EDA_SPLITS.items():
    imgs = eda_list_pngs(img_dir)
    msks = set(eda_list_pngs(msk_dir))
    paired = [f for f in imgs if f in msks]          # image has a matching mask
overlap = eda_split_files["train"] & eda_split_files["val"]   # -> 0, disjoint

# class imbalance
pf = st["pos_frac"]                    # per-patch cavity fraction
overall = pf.mean() * 100              # -> 3.02% (train), 2.15% (val)
ratio   = (1 - pf.mean()) / pf.mean()  # -> 32:1 (train), 46:1 (val)
```

### 2.3 What each EDA graph means

**(a) Cavity-pixel fraction per patch** — histogram of "what % of this patch is
cavity." It is **piled up near 0%** and has a long thin tail to the right. *Meaning:*
almost every patch is mostly background with only a sliver of cavity — this is the
**visual proof of class imbalance**, and the single most important EDA output. It
directly justifies the focal + dice loss.

**(b) Mean image intensity per patch** — histogram of average brightness (0–1) of
each X-ray. It's a broad bell spanning roughly **0.2–0.7**. *Meaning:* scans vary a
lot in brightness/contrast (different machines, exposures, patients). This tells me
simple `/255` scaling is fine but **contrast normalization (e.g. CLAHE) could help**,
and that the model must be robust to brightness changes.

**(c) Number of cavities (connected components) per patch** — how many separate
cavity blobs are in each patch. The bar at **1 dominates**, with a few patches at 2–6.
*Meaning:* most patches have a single lesion, but some have several — relevant because
the demo app only boxes the **largest** contour (so multi-lesion patches would only
get one box).

**(d) Individual lesion area (log scale)** — size of each cavity blob in pixels. Many
lesions are only a **few hundred pixels** (or fewer). *Meaning:* the targets are
**small**, which is exactly what makes segmentation hard — a few-pixel error costs a
lot of Dice on a tiny object.

**The spatial heatmap** — the average of all 4,426 training masks, i.e. "how often is
each pixel location a cavity." It is **diffuse** (peak probability only ≈ **0.04**),
not a hot centre. *Meaning:* cavities appear **all over** the patch, so the model must
be **position-robust** (translation-invariant); patches were not cropped so tightly
that the lesion is always dead-centre.

**The image/mask/overlay grid** — a sanity check: raw X-ray, its mask, and the mask
tinted red over the image. *Meaning:* confirms the annotations actually sit on visible
lesions (label quality is real, not noise).

The code that generates the four distributions (EDA cell 4):

```python
# (a) cavity-pixel fraction
ax.hist(eda_stats[s]["pos_frac"] * 100, bins=60, ...)
# (c) lesions per patch via connected components
n_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
areas = cc_stats[1:, cv2.CC_STAT_AREA]     # skip label 0 = background
```

---

## 3 · What I inferred from the data (my understanding)

Four takeaways, each of which drove a later decision:

1. **Severe class imbalance (~3% cavity).** → I cannot use pixel accuracy or plain
   cross-entropy. I need an imbalance-aware loss (**focal + dice**) and imbalance-aware
   metrics (**Dice, IoU, precision, recall**).
2. **Small, scattered lesions.** → Fine spatial detail matters, so the model needs
   **skip connections** (U-Net) to preserve edges, and it must be position-robust.
3. **Wide brightness spread.** → Normalization is essential; contrast enhancement is a
   reasonable future improvement.
4. **Validation is sparser than train (2.15% vs 3.02% cavity).** → There is a mild
   **distribution shift between splits**. This was an early warning that generalization
   might be hard — and indeed the model later overfit.

The data is **clean** (0 missing masks, 0 filename leakage, all 128×128, binary masks),
so any performance problem is a **modelling** problem, not a data-integrity problem.

---

## 4 · Data cleaning & normalization — what I did and why

All of this lives in `load_data()` (notebook cell T2 / `training.py`):

```python
def load_data(img_dir, mask_dir):
    files = sorted(os.listdir(img_dir))
    for fname in files:
        msk_path = os.path.join(mask_dir, fname)
        if not os.path.exists(msk_path):        # (1) skip images with no mask
            continue
        img = cv2.imread(img_path, 0)           # (2) read as GRAYSCALE (flag 0)
        msk = cv2.imread(msk_path, 0)
        if img is None or msk is None:          # (3) skip unreadable files
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))   # (4) force 128x128
        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0                       # (5) normalize image to [0,1]
        msk = msk / 255.0
        msk = (msk > 0.5).astype(np.float32)    # (6) BINARIZE the mask to {0,1}
        img = img[..., np.newaxis]              # (7) add channel dim -> (128,128,1)
        msk = msk[..., np.newaxis]
```

Step-by-step **why**:

1. **Pairing check** (`if not os.path.exists`): guarantees every training image has its
   ground-truth mask; unpaired files are dropped so nothing trains against a wrong label.
2. **Grayscale** (`cv2.imread(..., 0)`): X-rays carry no colour; one channel instead of
   three saves memory and parameters with zero information loss.
3. **Read-error guard**: skips corrupt files so one bad file can't crash the run.
4. **Resize to 128×128**: fixes a single tensor shape for the network. (Here the patches
   are already 128×128, so this is mostly a safety net.)
5. **Normalize `/255`**: scales pixels from 0–255 to **0–1**. Neural nets train much more
   stably when inputs are small and consistent — it keeps gradients well-behaved and lets
   the learning rate work across all images.
6. **Binarize the mask** `(msk > 0.5)`: resizing and PNG compression create intermediate
   gray pixels at mask edges. Segmentation targets must be strictly **0 or 1**, or the
   loss gets confused fractional labels. This makes the ground truth crisp.
7. **Add channel axis**: Keras conv layers expect shape `(H, W, C)`; grayscale needs an
   explicit channel of 1 → `(128, 128, 1)`.

**Augmentation** (a form of on-the-fly data expansion, cell T4):

```python
def augment_data(images, masks):
    for i in range(len(images)):
        if np.random.rand() < 0.5:
            images[i] = np.fliplr(images[i]); masks[i] = np.fliplr(masks[i])  # H-flip
        if np.random.rand() < 0.5:
            images[i] = np.flipud(images[i]); masks[i] = np.flipud(masks[i])  # V-flip
```

*Why:* flips are **label-preserving** (a cavity patch has no "correct" orientation), so
they multiply effective data for free and reduce overfitting. **Crucial caveat:** flips
are the **only** augmentation, and the mask is flipped in sync with the image so the
label stays correct. The thinness of this augmentation is one reason the model overfit —
richer augmentation (rotation, brightness, elastic) is on the improvements list.

---

## 5 · How I chose the model — why U-Net, why not others

**Why U-Net.** It is the **standard architecture for biomedical image segmentation**
(Ronneberger et al., 2015). It was literally designed for the situation I have: **few
labelled medical images**, **small structures**, needing a **full-resolution mask** out.
Its **skip connections** preserve the fine spatial detail that tiny lesions need — that's
the deciding factor.

**What else I could have used, and why not:**

| Alternative | What it is | Why not here |
|---|---|---|
| **Plain CNN classifier** | one label per image | gives only yes/no — no location to verify, defeats the purpose |
| **FCN / sliding-window CNN** | early segmentation nets | coarser masks, weaker at fine edges than U-Net's skip connections |
| **Mask R-CNN** | instance segmentation + boxes | heavy, needs lots of box annotations; overkill for one class on 128px patches |
| **DeepLab (atrous conv)** | segmentation for big natural scenes | designed for large, high-res images; needless complexity for 128px, small data |
| **Vision Transformers / SAM** | attention-based, foundation models | very data-hungry; ~5k images is far too few to train them well |

**One-sentence answer:** "For ~5,000 small grayscale patches with tiny targets, a compact
U-Net gives the right capacity and the skip connections I need for lesion detail; the
alternatives are either too coarse, too heavy, or too data-hungry."

---

## 6 · U-Net in detail — architecture, working, and how I used it

### 6.1 The idea
U-Net is an **encoder–decoder** shaped like a "U":
- The **encoder** (contracting path) repeatedly convolves and downsamples, learning
  **what** is in the image while shrinking spatial size.
- The **decoder** (expanding path) upsamples back to full resolution, rebuilding a
  pixel-level mask — recovering **where** things are.
- **Skip connections** copy each encoder feature map and concatenate it into the matching
  decoder level, so fine detail lost during downsampling is handed directly to the decoder.

### 6.2 My exact architecture (`build_unet`, cell T5)

```python
def build_unet(input_shape=(128,128,1)):
    inputs = layers.Input(input_shape)

    # ---- Encoder ----
    c1 = Conv2D(32,3,relu,same)(inputs); c1 = Conv2D(32,3,relu,same)(c1); p1 = MaxPooling2D()(c1)  # 128->64
    c2 = Conv2D(64,3,relu,same)(p1);     c2 = Conv2D(64,3,relu,same)(c2); p2 = MaxPooling2D()(c2)  # 64->32
    c3 = Conv2D(128,3,relu,same)(p2);    c3 = Conv2D(128,3,relu,same)(c3);p3 = MaxPooling2D()(c3)  # 32->16

    # ---- Bottleneck ----
    c4 = Conv2D(256,3,relu,same)(p3);    c4 = Conv2D(256,3,relu,same)(c4)                          # 16x16

    # ---- Decoder (upsample + concat skip + convs) ----
    u5 = UpSampling2D()(c4); u5 = Concatenate()([u5, c3]); c5 = Conv2D(128,..); c5 = Conv2D(128,..) # 16->32
    u6 = UpSampling2D()(c5); u6 = Concatenate()([u6, c2]); c6 = Conv2D(64,..);  c6 = Conv2D(64,..)  # 32->64
    u7 = UpSampling2D()(c6); u7 = Concatenate()([u7, c1]); c7 = Conv2D(32,..);  c7 = Conv2D(32,..)  # 64->128

    outputs = Conv2D(1, 1, activation="sigmoid")(c7)   # per-pixel cavity probability
    return Model(inputs, outputs)
```

**Reading it:**
- **Filters grow 32 → 64 → 128 → 256** going down (more abstract features as resolution
  drops), then **shrink back 128 → 64 → 32** going up.
- **Spatial size** goes 128 → 64 → 32 → 16 (bottleneck) → 32 → 64 → 128.
- Each block is **two 3×3 convolutions with ReLU** and `padding="same"` (keeps size fixed
  within a block).
- **Downsampling** = `MaxPooling2D` (halves H, W). **Upsampling** = `UpSampling2D` (doubles
  H, W by nearest-neighbour) followed by convs — I used this instead of `Conv2DTranspose`
  because it has **fewer parameters and avoids checkerboard artifacts**.
- **`Concatenate()([u, c])`** is the skip connection: it glues the decoder feature map to
  the same-resolution encoder feature map.
- **Output layer**: `Conv2D(1, 1, sigmoid)` → a 128×128×1 map where each value is the
  **probability that pixel is a cavity** (0–1). Sigmoid (not softmax) because it's a single
  foreground class.
- **Size:** ~**1.9 million parameters** — deliberately compact for the small dataset and
  Colab GPU.

### 6.3 Did I fine-tune it?
**No transfer learning / no pretrained weights** — this U-Net is **trained from scratch**
(random init). "Fine-tuning" here means I hand-designed and tuned the architecture and
training setup (depth, filter counts, loss, callbacks) rather than loading a pretrained
backbone. **Using a pretrained encoder (ResNet/EfficientNet-U-Net) is a top improvement** —
it would likely generalize much better from so few images and reduce the overfitting.

---

## 7 · Parameters & hyperparameters — what I chose and why

Set in cell T1:

```python
IMG_SIZE   = 128     # patch size fed to the network
BATCH_SIZE = 16      # images per gradient step
EPOCHS     = 100     # full passes over the data
LR         = 1e-4    # Adam learning rate
```

**How to think about each (and my reason):**

- **Image size = 128.** Balances detail vs compute. Big enough to keep lesion detail; small
  enough that a 3-level U-Net (bottleneck 16×16) still has spatial context and trains fast.
  The patches were already 128, so this matched the data.
- **Batch size = 16.** How many patches per gradient update. Bigger = smoother, more stable
  gradients but more GPU memory; smaller = noisier but sometimes generalizes better. 16 is a
  **standard sweet spot** that fits comfortably in Colab GPU memory at 128×128 → gives
  4426/16 ≈ **277 steps per epoch**.
- **Learning rate = 1e-4 with Adam.** The step size for weight updates. Too high → training
  diverges/oscillates; too low → trains very slowly. **1e-4 is a safe, conservative default
  for Adam** and it trained smoothly (loss fell steadily with no divergence).
- **Epochs = 100.** How many times the model sees the whole dataset. I set a high ceiling and
  relied on callbacks (early stopping) to cut it short if validation stopped improving. In
  practice training Dice kept rising to 100, but **validation Dice plateaued early** — the
  sign of overfitting.
- **Loss = focal + dice** (see section 8).
- **Optimizer = Adam.** Adaptive optimizer that needs little manual LR tuning — a robust
  default for segmentation.

**Honest note on tuning depth:** these are **well-reasoned standard values**, not the result
of an exhaustive grid/random search. If asked "did you tune hyperparameters?", the honest
answer is: "I chose sensible, literature-standard values and validated they trained stably;
systematic tuning (LR schedules, batch size, threshold, augmentation strength) is future
work — and given the overfitting, regularization strength is what I'd tune first."

**Callbacks** (cell T8) — these *adapt* hyperparameters during training:

```python
checkpoint = ModelCheckpoint(SAVE_BEST, monitor="val_dice_coef", save_best_only=True, mode="max")
early_stop = EarlyStopping(monitor="val_dice_coef", patience=15, mode="max", restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor="val_dice_coef", factor=0.5, patience=7, mode="max")
```

- **ModelCheckpoint** saves the best model by validation Dice.
- **EarlyStopping** stops if val Dice doesn't improve for 15 epochs (prevents wasted epochs).
- **ReduceLROnPlateau** halves the LR if val Dice stalls for 7 epochs (helps fine convergence).

---

## 8 · How I dealt with class imbalance

The imbalance is **~3% cavity pixels (32:1 background:cavity)**. Three defenses:

**(1) Never use pixel accuracy.** A model that predicts "all background" scores ~98%
accuracy while finding nothing. I report **Dice, IoU, precision, recall** instead.

**(2) Binarize masks + focus the loss.** The core fix is the **combined loss** (cell T6):

```python
def dice_coef(y_true, y_pred, smooth=1e-6):
    yt = tf.reshape(y_true,[-1]); yp = tf.reshape(y_pred,[-1])
    inter = tf.reduce_sum(yt*yp)
    return (2*inter + smooth) / (tf.reduce_sum(yt)+tf.reduce_sum(yp)+smooth)

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        pt = tf.where(tf.equal(y_true,1), y_pred, 1-y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1.-pt, gamma) * tf.math.log(pt))
    return loss

def focal_dice_loss(y_true, y_pred):
    return focal_loss()(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))
```

- **Focal loss** — a weighted cross-entropy. The `(1 - pt)^gamma` factor **down-weights easy,
  well-classified pixels** (mostly background) so the gradient focuses on the **hard cavity
  pixels**. `gamma=2` makes a confidently-correct pixel contribute ~100× less; `alpha=0.25`
  is a class weighting factor. *Why chosen:* stops the majority background class from
  drowning the learning signal.
- **Dice loss** = `1 - dice_coef`. Dice measures **region overlap** between prediction and
  truth, so background pixels **cannot inflate it** — it's imbalance-proof by construction.
  *Why chosen:* it directly optimizes the exact overlap metric I report. The `smooth=1e-6`
  avoids divide-by-zero on empty masks.
- **Sum of both**: focal gives **stable pixel-level gradients**; dice **targets the metric**.
  Together they're the standard recipe for small-lesion medical segmentation.

**(3) Flip augmentation** slightly increases effective positive examples.

---

## 9 · Did I split the data? (train / val / test)

**What exists:** a **train/val split** — 4,426 train and 730 val patches, with **zero
filename overlap** (verified in EDA). Validation was wired into training
(`validation_data=(X_val, y_val)`), so I could watch generalization each epoch.

**What's missing / imperfect — be honest about this:**
1. **No separate held-out test set.** There's a `testing/` folder, but it's an **exact copy
   of the training images** — so any "test" on it would be **data leakage**. I evaluate on
   the disjoint **validation** set instead, and would add a true held-out test split.
2. **The split is patch-level, not patient-level.** Several patches can come from the **same
   source X-ray**, so patches from one patient can land in both train and val — a subtle
   leakage risk. The clean fix is to **group the split by source image/patient** and
   stratify by cavity load (which also fixes the train-vs-val distribution gap).

**So the honest answer:** "Yes — an 86/14 train/val split with no filename overlap, and I
monitored validation during training. But it's patch-level, and there's no clean held-out
test set yet; patient-level splitting is my first data fix."

---

## 10 · How training works, in detail

Training is a **manual per-epoch loop** (cell T9) rather than a single `model.fit(epochs=100)`:

```python
training_history = {}
for epoch in range(EPOCHS):
    X_train_aug, y_train_aug = augment_data(X_train.copy(), y_train.copy())   # (1) re-augment
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),                                       # (2) watch val
        epochs=1, batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop, reduce_lr], verbose=1)
    for k, v in history.history.items():
        training_history.setdefault(k, []).extend(v)                          # (3) log metrics
```

**Step by step:**
1. **Re-augment each epoch** — a fresh copy of the training set is flipped randomly, so the
   network sees different variations every epoch (the whole reason for the loop).
2. **`model.fit(..., epochs=1)`** runs one pass: for each batch of 16, it does a **forward
   pass** (U-Net → probability map), computes **focal+dice loss** vs the true mask, and
   **Adam backpropagates** to update weights. `validation_data` computes val metrics after
   the pass.
3. **`training_history`** accumulates train/val Dice and loss so I can plot learning curves.

Then the model is compiled beforehand (cell T7):

```python
model.compile(optimizer=Adam(1e-4), loss=focal_dice_loss,
              metrics=[dice_coef, Precision(name="precision"), Recall(name="recall")])
```

and saved after (cell T10): `model.save(SAVE_FINAL)` → `unet_cavity_final.h5`.

**One honest technical subtlety (good to know):** calling `fit(epochs=1)` in a loop **resets
the internal state of EarlyStopping/ReduceLROnPlateau each call**, so those callbacks don't
accumulate patience the way they would in a single long `fit`. The right design is a single
`fit(epochs=100)` with a `tf.data` pipeline that does the flipping. Knowing this is a plus —
it shows you audited your own code.

---

## 11 · How testing / inference works, in detail

Inference is the desktop app in `testing.py`. Flow is **front → back → front**:

```python
# ---- load the trained model (custom objects required!) ----
model = tf.keras.models.load_model(MODEL_PATH,
    custom_objects={"dice_coef": dice_coef, "focal_dice_loss": focal_dice_loss})

# ---- FRONT: user picks an image ----
file_path = filedialog.askopenfilename(...)      # Tkinter file dialog

# ---- preprocess EXACTLY like training ----
gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(gray, (128,128)) / 255.0
img_input = np.expand_dims(img_resized[...,np.newaxis], axis=0)   # (1,128,128,1)

# ---- BACK: predict + threshold ----
pred_mask = model.predict(img_input)[0,:,:,0]     # probability map
binary_mask = (pred_mask > 0.5).astype(np.uint8)  # threshold at 0.5
binary_mask_full = cv2.resize(binary_mask, (orig_W, orig_H))   # back to original size

# ---- BACK: classical CV turns mask into a decision ----
contours,_ = cv2.findContours(binary_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
if cv2.contourArea(largest) > 100:                # noise filter
    x,y,w,h = cv2.boundingRect(largest)
    cv2.rectangle(result_img, (x,y), (x+w,y+h), (0,0,255), 2)   # red box
    cv2.putText(result_img, "Cavity Detected", ...)

# ---- FRONT: show 4-panel figure + verdict ----
```

**What each step does and why:**
- **`custom_objects`**: the `.h5` file stores the *names* of the custom loss/metric; Keras
  needs the actual Python functions to reload the model, so I pass them in.
- **Identical preprocessing**: the image must be gray / 128 / normalized **exactly like
  training**, or the model sees an out-of-distribution input.
- **`model.predict`** → a **probability map** (each pixel's cavity likelihood).
- **Threshold 0.5** → a crisp **binary mask** (cavity vs not).
- **Resize mask back** to the original resolution so the box lines up with the real image.
- **`findContours` + `boundingRect`**: classical OpenCV converts the mask into a **bounding
  box**; `max(..., contourArea)` takes the biggest blob, and **area > 100** rejects tiny
  noise specks. Then it draws a red box + "Cavity Detected".
- **Display**: original | probability map | binary mask | boxed result, and prints
  "CAVITY DETECTED" / "NO CAVITY DETECTED".

**The tagline:** *deep learning does the seeing (the mask); classical CV makes the call (the
box + verdict).*

---

## 12 · The outputs — what they are and why

- **Probability map** (model output): continuous 0–1 per pixel — how confident the model is
  each pixel is cavity. *Why:* sigmoid output; keeps information before thresholding.
- **Binary mask** (after > 0.5): the yes/no cavity region. *Why:* a decision needs a hard
  boundary; 0.5 is the default cut-off.
- **Bounding box + label** (after OpenCV): a human-friendly result a dentist can read at a
  glance. *Why:* a box + "Cavity Detected" is more usable than a raw mask.

**Why the outputs look the way they do (honest):** on **large, clear lesions** the model
predicts a solid mask and the box lands correctly. On **small, faint lesions** the prediction
is often **blank or a tiny speck** — because the model **overfit** and plays it safe (very
high precision, very low recall). The sample-prediction grid shows exactly this: mostly empty
prediction columns, with a good hit only on the biggest lesion.

---

## 13 · Validation output & reasoning

Evaluation runs on the **held-out validation set** (cells E1–E6). Real numbers:

| Metric | Value | What it means |
|---|---|---|
| **Dice** | **0.25** | overlap between predicted and true cavity — low |
| **IoU** | **0.14** | stricter overlap (intersection/union) — low |
| **Precision** | **0.51** | of pixels called cavity, ~half are right |
| **Recall** | **0.16** | of true cavity pixels, only 16% were found |
| **Specificity** | **0.996** | background is handled almost perfectly |
| **Pixel accuracy** | **0.979** | *meaningless here* — background dominates |
| **Patch sensitivity** | **43.7%** | catches a cavity in 319 / 730 patches |
| **Best-threshold Dice** | **0.26 @ 0.10** | tuning the cut-off barely helps |

**How they're computed** (E1–E2):

```python
y_prob = model.predict(X_val)                 # probabilities on held-out val
m05 = seg_scores(y_true, (y_prob > 0.5))      # counts TP/FP/FN/TN -> dice, iou, P, R...
# threshold sweep 0.10..0.90 -> best Dice 0.26 at 0.10
```

**The key reasoning — overfitting:** the training-curve check (E5) shows **train Dice 0.81
vs val Dice 0.21, a gap of +0.60**, and **val loss never falls while train loss drops**.
That divergence is the textbook signature of **overfitting**: the model memorized the
training patches but didn't learn to generalize.

**Why recall is so low:** with 32:1 imbalance and thin augmentation, the model minimizes loss
by predicting cavity **rarely** — hence high precision (0.51), very low recall (0.16). A
threshold sweep can't fix it (best Dice only 0.26), which proves the ceiling is the **model /
training**, not the cut-off.

**Why present this honestly:** claiming the training 0.82 as "results" is the exact trap an
interviewer catches. Presenting the real val numbers + a correct **diagnosis** (overfitting)
+ a **fix plan** demonstrates real data-science maturity — which is what they're grading.

---

## 14 · The whole code workflow / architecture

The notebook runs top-to-bottom in four blocks:

```
SETUP
  cell 0  mount Google Drive
  cell 3  pip install
  cell 4  version check
      |
EDA  (read-only, before training)
  1 Inventory & integrity  -> 4426/730, 0 missing, 0 leakage, 128x128, binary
  2 Per-image statistics   -> intensity + cavity-fraction + lesion counts/areas
  3 Class imbalance         -> 3.02% / 2.15% cavity, 32:1 / 46:1
  4 Distribution plots      -> (a) imbalance (b) brightness (c) #lesions (d) sizes
  5 Spatial heatmap         -> diffuse, peak ~0.04
  6 Sample overlays         -> label sanity check
  7 EDA summary
      |
TRAINING  (T1..T10)
  T1 config/hyperparams  ->  T2 load_data()  ->  T3 load train/val
  T4 augment  ->  T5 build_unet  ->  T6 focal+dice loss
  T7 compile + summary  ->  T8 callbacks  ->  T9 100-epoch loop  ->  T10 save .h5
      |
EVALUATION  (E1..E6)
  E1 predict on val + metric helper
  E2 metrics @0.5 + threshold sweep
  E3 patch-level detection rate (43.7%)
  E4 confusion matrix + threshold curve
  E5 training curves + overfitting gap (0.81 vs 0.21)
  E6 sample predictions
```

**Separate files:**
- `training.py` — the standalone training script (same logic as notebook T-cells).
- `testing.py` — the Tkinter/OpenCV inference app (section 11).
- `unet_cavity_final.h5` — the saved trained model (~23 MB).
- `patch_dataset/train|val/images|masks/` — the data.
- `testing/` — **a duplicate of the training images** (not a real test set — leakage).

**Data flow in one line:**
> Drive dataset → `load_data()` (gray, resize, normalize, binarize) → in-memory tensors →
> per-epoch (augment → U-Net forward → focal+dice loss → Adam backprop → callbacks) ×100 →
> save `.h5` → evaluate on val → `testing.py` loads `.h5` → predict → threshold → OpenCV box.

---

## Quick self-test (say these out loud before the interview)

1. What's the task? *Pixel-level segmentation of caries on X-ray patches.*
2. How imbalanced? *~3% cavity pixels, 32:1 — drives focal+dice loss and Dice/IoU metrics.*
3. Why U-Net? *Skip connections preserve tiny-lesion detail; works with little data.*
4. Model size & output? *~1.9M params; 128×128×1 sigmoid probability map.*
5. Loss? *focal (down-weights easy background) + (1 − dice) (optimizes overlap).*
6. Key hyperparameters? *128 px, batch 16, Adam 1e-4, 100 epochs + callbacks.*
7. Results? *Train Dice 0.81 but val Dice ~0.21–0.25, recall 0.16 — overfitting.*
8. Biggest fixes? *Regularization + richer augmentation, patient-level split, negatives, transfer learning.*
9. Split? *86/14 train/val, no leakage by filename, but patch-level and no clean test set yet.*
10. Testing flow? *Pick image → preprocess → predict → threshold → contour → box + verdict.*
