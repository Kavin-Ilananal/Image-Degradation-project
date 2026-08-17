# Repo checklist — Image-Degradation-project

Current state: **1 file (README.md, 109 bytes)**. As it stands the submission
cannot be benchmarked, and the brief is explicit that unscored submissions
cannot win. Everything below is required.

---

## The six mandatory items

| # | required | present | what to do |
|---|---|---|---|
| 1 | `README.md` — clone and run without contacting you | placeholder | replace with the supplied README, then fill in anything marked |
| 2 | **Evaluation script (.py)** — `--input_dir`, `--output_dir`, runs as-is | **MISSING** | **highest priority — see below** |
| 3 | Training script (.py or .ipynb) | **MISSING** | push the notebook/script that produced the log |
| 4 | Trained weights | **MISSING** | both `.pth` files (Phase 1 and Phase 2) |
| 5 | Restored test outputs | **MISSING** | run the model on all 400 test images, commit the folder |
| 6 | `requirements.txt` | **MISSING** | `pip freeze`, or the packages actually imported, pinned |

Repository must also be **public**. Verify by opening it in a private browser
window — if it 404s, KLA sees a 404 too.

---

## 1. The evaluation script — do this first

KLA runs this file **as-is** on an H100 to measure both quality and inference
time. If it needs any manual edit, the submission cannot be scored.

It must accept exactly:

```bash
python inference.py --input_dir <test_images> --output_dir <restored>
```

Requirements that are easy to get wrong:

- **Default weight paths must resolve relative to the script, not the working
  directory.** Use:
  ```python
  import os
  HERE = os.path.dirname(os.path.abspath(__file__))
  DEFAULT_CKPT = os.path.join(HERE, 'weights', 'three_stage_rcan_best_phase1.pth')
  ```
  A default like `checkpoints/best.pth` fails the moment the reviewer runs it
  from another directory.
- **Weights must be in the repo**, not on Drive, unless the script downloads
  them automatically.
- **Handle CPU.** Fall back with a warning rather than crashing if CUDA is absent.
- **Write one `.npy` per input**, same filename, float32, 2× resolution.
- **Decide on TTA.** Reported scores use it, so either leave it on and quote the
  slower latency, or turn it off and quote the faster one. The script and the
  slides must agree.

### Test it the way KLA will

```bash
cd /some/empty/dir
git clone https://github.com/Kavin-Ilananal/Image-Degradation-project.git
cd Image-Degradation-project
pip install -r requirements.txt
python inference.py --input_dir <test> --output_dir ./out
```

Run that in a **fresh folder**, not your working copy. Testing in place hides
files that exist locally but were never committed — the most common way this
fails.

---

## 2. Weights

Both checkpoints, since the two phases are different champions:

```
weights/three_stage_rcan_best_phase1.pth        PSNR champion  — 29.79 dB
weights/three_stage_rcan_best_phase2_ssim.pth   SSIM champion  — 0.7032
```

They currently store `{'model_state_dict': ...}`, so the loader must unwrap that
key. Under 100 MB each commits directly; larger needs Git LFS or a documented
download link.

---

## 3. Restored test outputs

Run the model over all 400 images in `Test_NoisyLR/` and commit the folder.
Then verify before pushing:

```python
import numpy as np, glob
fs = sorted(glob.glob('results/restored_test/*.npy'))
a = [np.load(f) for f in fs]
print(len(fs), a[0].shape, a[0].dtype,
      'finite', all(np.isfinite(x).all() for x in a),
      'blank', sum(x.std() < 1e-6 for x in a))
```

Expect `400 (256, 256) float32 finite True blank 0`. A blank frame means the
network diverged and a NaN was silently converted to zeros — worth catching now
rather than having KLA score it.

---

## 4. Two things worth measuring before submitting

**Latency.** KLA benchmarks inference time. Measure it yourself on your own GPU
and report that, with `torch.cuda.synchronize()` around the timed region or the
number will be wrong. Do not quote an H100 estimate.

**Parameter count.** 4 621 017 — already in the log, and worth putting on the
slides.

---

## 5. Suggested commit order

```bash
git add inference.py train.py model.py requirements.txt README.md
git commit -m "code, dependencies and documentation"
git add weights/
git commit -m "trained weights: phase 1 (PSNR) and phase 2 (SSIM) champions"
git add results/restored_test/
git commit -m "restored outputs for all 400 competition test images"
git push origin main
```

Then re-clone into a fresh directory and run the evaluation command once more.
That final check is the one that matters.
