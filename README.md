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
python run.py <input-dir> <output-dir>
```

**Positional arguments, no flags, no file editing.** The output directory is
created if it does not exist. Every path resolves relative to `run.py` itself,
so this runs from any working directory on a fresh clone. Runs on an NVIDIA GPU
with no internet access, no API keys, no additional model downloads, no user
interaction and no manual configuration — the weights ship in `models/`.

`inference.py` remains available for development use (flags, optional TTA), but
**`run.py` is the submission entry point**.

Reads `.npy` (float32, any size), writes `.npy` at 2× resolution, float32 in
[0, 1], one file per input with the same filename. Falls back to CPU with a
warning if CUDA is absent.

Verified over the full 400-image test set:

| check | result |
|---|---|
| one output per input, same filenames | 400 / 400 |
| shape | all (256, 256) |
| dtype | float32 |
| values within [0, 1] | [0.0000, 1.0000] |
| NaN or Inf | none |
| blank frames | none |

### Optional test-time augmentation

```bash
python inference.py --input_dir <in> --output_dir <out> --tta
```

8-way (4 rotations × 2 flips). Gains ~0.07 dB PSNR at **8× the inference cost**,
so it is **off by default** — the reported latency is the latency actually
incurred. Turn it on only if quality is scored and time is not.

---

## Results

Measured on a **family-grouped holdout**. Every pair of images whose
full-resolution ground truth correlates above 0.90 is joined by an edge; the
connected components of that graph are "families"; whole families are assigned
to one side of the split. Near-duplicate frames therefore cannot straddle it.

**2560 train / 640 validation** (80 % / 20 %) from the 3200 released pairs.
Verified at split time: maximum train-to-validation correlation **0.8881** — zero
pairs above 0.90, and by the same measurement zero above 0.95 or 0.98.

| model | PSNR (dB) | SSIM |
|---|---|---|
| Phase 1 — PSNR champion, with TTA | **28.24** | 0.7681 |
| Phase 2 — SSIM champion, with TTA | 28.11 | **0.7686** |
| Phase 1, single pass (best epoch) | 28.17 | — |
| Phase 2, single pass (best epoch) | — | 0.7671 |

Both checkpoints ship. Phase 2 is the better all-round choice: it takes SSIM and
gives up 0.14 dB of PSNR. Under an earlier, leakier split the two phases looked
like genuinely different champions; they no longer do.

### Distribution — Phase 1 with TTA

| | PSNR (dB) | SSIM |
|---|---|---|
| worst | 11.70 | 0.3545 |
| p10 | 22.86 | 0.7495 |
| **median** | **28.17** | **0.8063** |
| p90 | 34.70 | 0.9071 |
| best | 41.97 | 0.9572 |

Worst case `002537.npy`, best `003117.npy`. The full distribution is reported
rather than the mean alone: a high average with an 11 dB worst case is a
different model from one with a 20 dB worst case, and for inspection the tail is
what matters.

### Robustness

Across all 400 competition test images, single pass: **0 NaN or Inf, 0 blank
frames**, every output inside [0.0000, 1.0000]. On 90 held-out crops drawn from
unrelated imaging domains the model beats bicubic in **88 of 90** cases with no
degenerate outputs.

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

Total: **98 epochs on 2 GPUs**, against a 150-epoch budget — 68 in phase 1
plus the 30-epoch SSIM fine-tune, which ran to its full length.

---

## Repository layout

```
run.py                  SUBMISSION ENTRY POINT — python run.py <input-dir> <output-dir>
inference.py            development entry point — flags, optional TTA
train.py                reproduces training from scratch
model.py                CascadedRCAN definition (run it to print the parameter count)
models/
  rcan_phase1_psnr.pth  PSNR champion — 28.24 dB   (default)
  rcan_phase2_ssim.pth  SSIM champion — 0.7686
results/restored_test/  model output for all 400 competition test images
requirements.txt        pinned dependencies
```

---

## Known limitations

- **Inference cost.** All 54 RCABs run at full 128x128 resolution; the network
  only upsamples at the very end, so there is no cheap low-resolution stage.
  Measured at **30.7 ms/image on GPU** single-pass (12.3 s for the 400-image test
  set); TTA multiplies that by 8. Reducing stage depth or downsampling inside the
  stages is the lever if latency becomes binding.
- **Worst case is 11.70 dB.** The 10th-percentile SSIM of 0.7495 against a 0.8063
  median is far healthier than it looks at the mean, but the PSNR floor is still
  low and a handful of images are restored poorly.
- **TTA is reported but not shipped by default.** The headline scores use it;
  the default run does not. Quote whichever configuration matches the latency
  you report.
- **Validation is composed entirely of single-image families.** The split packs
  families into train largest-first and sends the remainder to validation, so
  every multi-image family ends up in training and the 640 validation images are
  all structurally isolated. Measured, this makes them no harder than the rest of
  the data (bicubic 23.01 dB vs 23.21 dB, p = 0.48), so the headline is not
  distorted — but the model is never validated on the duplicate-rich subset.
- **Numbers are not comparable to other teams' holdout numbers** unless the same
  split file is used. The correlation threshold, the image the correlation is
  computed on, and how families are packed all change which images end up in
  validation, and therefore the score.

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
