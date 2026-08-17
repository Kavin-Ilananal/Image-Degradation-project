# Three-Stage Cascaded RCAN — Blind Restoration of Degraded Semiconductor Inspection Images

**SEMICON India Hackathon 2026 — Track 1 (KLA) · PS01**
*AI-Based Restoration of Degraded Images for Semiconductor Inspection*

Blind restoration of grayscale inspection images degraded by **speckle noise +
additive Gaussian noise + 2× downsampling**, applied in unknown order.
128×128 → 256×256.

**4 621 017 parameters · three cascaded RCAN stages · two-phase training with
automatic termination**

---

## Quick start

```bash
git clone https://github.com/Kavin-Ilananal/Image-Degradation-project.git
cd Image-Degradation-project
pip install -r requirements.txt
python inference.py --input_dir <degraded_dir> --output_dir <restored_dir>
```

No other arguments are required and no file needs editing. Default paths resolve
relative to `inference.py` itself, so this runs from any working directory on a
fresh clone. Weights are committed — no external download.

Reads `.npy` (float32, any size), writes `.npy` at 2× resolution, float32 in
[0, 1], one file per input with the same filename. Falls back to CPU with a
warning if CUDA is absent.

Verified over the full 400-image test set: **400/400 written, 0 non-finite,
0 blank, all 256×256 float32 in [0.0000, 1.0000]**.

### Optional test-time augmentation

```bash
python inference.py --input_dir <in> --output_dir <out> --tta
```

8-way (4 rotations × 2 flips). Gains ~0.07 dB PSNR at **8× the inference cost**,
so it is **off by default** — the reported latency is the latency actually
incurred. Turn it on only if quality is scored and time is not.

---

## Results

Measured on a **texture-cluster holdout**: images are described by a
25-dimensional texture feature vector, grouped into 10 clusters with k-means,
and clusters 2, 3 and 8 are withheld *entirely* from training. Validation
therefore contains structurally unseen images, not merely unseen files.

**2585 train / 615 validation** (80.8 % / 19.2 %) from the 3200 released pairs.
Verified at split time: no train/validation cluster overlap, no file overlap,
all 3200 pairs accounted for.

| model | PSNR ↑ | SSIM ↑ |
|---|---|---|
| **Phase 1 — PSNR champion, with TTA** | **29.79** | 0.6983 |
| Phase 2 — SSIM champion, with TTA | 29.64 | **0.7032** |
| Phase 1, single pass (best epoch) | 29.72 | 0.6954 |
| Phase 2, single pass (best epoch) | 29.58 | 0.7021 |

Both checkpoints are shipped: the two training phases produce **different
champions**, so the choice depends on which metric matters downstream.

### Distribution — Phase 1 with TTA

| | PSNR | SSIM |
|---|---|---|
| worst | 11.19 | 0.3101 |
| p10 | 24.61 | 0.3091 |
| **median** | **30.57** | **0.9144** |
| p90 | 35.45 | 0.9418 |
| best | 41.94 | 0.9579 |

Worst case `000958.npy`, best `003117.npy`. The full distribution is reported
rather than the mean alone: a high average with an 11 dB worst case is a
different model from one with a 20 dB worst case, and for inspection the tail is
what matters.

---

## Method

### Architecture

```
3-channel input ──> head 3x3 ──> Stage 1 ──> Stage 2 ──┐
                                    │                  ├─> fusion 1x1 ──> Stage 3
                                    └──────────────────┘                     │
                                                              PixelShuffle x2 │
bilinear x2 of channel 0 ────────────────────────────(+)<── tail 3x3 ─────────┘
```

Each **Stage** is 3 residual groups of 6 RCABs, plus a group convolution —
54 RCABs in total. Each **RCAB** is conv → ReLU → conv → channel attention,
wrapped in a residual connection. Channel attention pools each feature map to a
scalar and gates it, letting the network weight channels by usefulness rather
than treating them equally — helpful when one image contains both flat regions
dominated by noise and structured regions carrying signal.

**Global residual learning.** The tail predicts a *correction* to the bilinear
upsample of the raw input, not the image itself. If the network fails, the
output degrades toward interpolation rather than toward invented structure —
which matters when the output is inspection evidence.

**Stage-1 / stage-2 fusion.** Features from both stages are concatenated and
mixed by a 1×1 convolution before stage 3, so later processing can draw on both
early and intermediate representations.

### Three-channel input

Each image is expanded into three channels before the network sees it:

| channel | content | why |
|---|---|---|
| 0 | raw noisy array, **unclipped** | values exceed 1.0 in this dataset and that excursion is signal, not error |
| 1 | bilateral filter (d=9, σ=75/75) | edge-preserving smoothing — suppresses noise without dissolving boundaries |
| 2 | log-domain Fourier low-pass | speckle is *multiplicative*, so it becomes additive under a log transform; filtering there and exponentiating back targets speckle specifically |

Channel 2 is the interesting one: taking the log turns a multiplicative
corruption into an additive one, which a linear low-pass filter can then
attack — a direct answer to the speckle component rather than generic smoothing.

### Two-phase training with automatic termination

**Phase 1 — PSNR.** Trained from scratch with warmup then cosine decay from
4.0e-4, optimising reconstruction error.

**Phase 2 — SSIM.** The Phase-1 champion is reloaded and fine-tuned at 1.0e-6
for 30 epochs against structural similarity.

Training stops on either trigger rather than at a fixed epoch count:
- PSNR varies by less than 0.02 dB over 5 epochs
- SSIM drops for 3 consecutive epochs

Total: **84 epochs, ~2.6 h on 2 GPUs**, against a 150-epoch budget.

---

## Repository layout

```
inference.py            EVALUATION ENTRY POINT — --input_dir / --output_dir, no edits needed
train.py                reproduces training from scratch
model.py                CascadedRCAN definition (run it to print the parameter count)
weights/
  rcan_phase1_psnr.pth  PSNR champion — 29.79 dB   (default)
  rcan_phase2_ssim.pth  SSIM champion — 0.7032
results/restored_test/  model output for all 400 competition test images
requirements.txt        pinned dependencies
```

---

## Known limitations

- **Inference cost is high.** All 54 RCABs run at full 128×128 resolution; the
  network only upsamples at the very end, so there is no cheap low-resolution
  stage. Measured at ~470 ms/image on CPU single-pass, and 8× that with TTA.
  Reducing stage depth or downsampling inside the stages would cut this
  substantially. **KLA benchmarks inference time as well as quality**, so this
  should be measured on GPU and reported honestly.
- **Worst case is 11.19 dB**, and SSIM at the 10th percentile is 0.3091 against a
  0.9144 median — structural fidelity collapses on the hardest images.
- **TTA is reported but not shipped by default.** The headline scores use it;
  the default run does not. Quote whichever configuration matches the latency
  you report.
- **Numbers are not comparable to other teams' cluster-holdout numbers** unless
  the same split file is used. Which clusters you withhold materially changes the
  score, so a "texture-cluster split" from a different feature set, a different
  *k*, or a different choice of held-out clusters is a different exam.

---

## References

- Zhang et al., *Image Super-Resolution Using Very Deep Residual Channel
  Attention Networks* (RCAN), ECCV 2018
- Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018 — channel attention
- Shi et al., *Real-Time Single Image and Video Super-Resolution Using an
  Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016 — PixelShuffle
- Tomasi & Manduchi, *Bilateral Filtering for Gray and Color Images*, ICCV 1998
- Wang et al., *Image Quality Assessment: From Error Visibility to Structural
  Similarity* (SSIM), IEEE TIP 2004
- KLA / SEMICON India Hackathon 2026, Track 1 PS01 — provided dataset
  (3200 training pairs, 400 test images)
