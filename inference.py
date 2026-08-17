"""Evaluation entry point — Three-Stage Cascaded RCAN.

    python inference.py --input_dir <degraded_dir> --output_dir <restored_dir>

No other arguments are required and no file needs editing. Default paths
resolve relative to THIS FILE, not the working directory, so it runs from any
cwd on a fresh clone.

Reads float32 .npy of any size, writes float32 .npy at 2x resolution with the
same filename. Falls back to CPU with a warning if CUDA is absent.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, 'models', 'rcan_phase1_psnr.pth')
SEED = 42

try:
    import cv2
    HAVE_CV2 = True
except ImportError:                                    # graceful, not fatal
    HAVE_CV2 = False


def preprocess(noisy):
    """Build the 3-channel input: raw | bilateral | log-Fourier speckle filter.

    The RAW channel is deliberately left unclipped — values exceed 1.0 in this
    dataset and that excursion carries signal. Only the two derived channels use
    a clipped copy, because both operate in uint8 / log space.
    """
    noisy = np.squeeze(noisy).astype(np.float32)
    raw = torch.from_numpy(noisy).float()

    clipped = np.clip(noisy, 0.0, 1.0)

    # bilateral: edge-preserving smoothing
    if HAVE_CV2:
        u8 = (clipped * 255.0).astype(np.uint8)
        bil = cv2.bilateralFilter(u8, d=9, sigmaColor=75, sigmaSpace=75)
        bil = bil.astype(np.float32) / 255.0
    else:                                              # box-blur stand-in
        t = torch.from_numpy(clipped)[None, None]
        bil = F.avg_pool2d(F.pad(t, (2, 2, 2, 2), mode='reflect'), 5, 1)[0, 0].numpy()
    bilateral = torch.from_numpy(np.ascontiguousarray(bil)).float()

    # speckle is multiplicative, so it becomes additive in the log domain:
    # low-pass there, then exponentiate back
    log_img = np.log(clipped + 1e-5)
    fft_shift = np.fft.fftshift(np.fft.fft2(log_img))
    rows, cols = clipped.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask = np.exp(-(((x - ccol) ** 2 + (y - crow) ** 2)) / (2.0 * 15 ** 2))
    spec = np.exp(np.real(np.fft.ifft2(np.fft.ifftshift(fft_shift * mask))))
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    speckle = torch.from_numpy(spec.astype(np.float32)).float()

    return torch.stack([raw, bilateral, speckle], dim=0).unsqueeze(0)


def load_model(ckpt=None, device=None):
    ckpt = ckpt or DEFAULT_CKPT
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print('[warn] CUDA unavailable - running on CPU. '
              'Latency will not reflect GPU performance.')
    model = build_model()
    obj = torch.load(ckpt, map_location=device)
    state = obj.get('model_state_dict', obj) if isinstance(obj, dict) else obj
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval(), device


@torch.no_grad()
def forward_tta(model, x):
    """8-way: 4 rotations x {identity, horizontal flip}. Deterministic."""
    preds = []
    for i in range(4):
        xr = torch.rot90(x, i, [2, 3])
        preds.append(torch.rot90(model(xr), -i, [2, 3]))
        xf = torch.flip(xr, [3])
        preds.append(torch.rot90(torch.flip(model(xf), [3]), -i, [2, 3]))
    return torch.stack(preds).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT,
                    help='default: models/rcan_phase1_psnr.pth beside this script. '
                         'Use models/rcan_phase2_ssim.pth for the SSIM champion.')
    ap.add_argument('--tta', action='store_true',
                    help='8-way test-time augmentation: ~+0.07 dB PSNR at 8x the '
                         'inference cost. OFF by default so the reported latency '
                         'is the latency actually incurred.')
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, device = load_model(args.ckpt)
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    if not files:
        raise SystemExit(f'no .npy files found in {args.input_dir}')

    # warmup so the first image does not absorb allocation and kernel setup
    w = preprocess(np.load(files[0])).to(device)
    with torch.no_grad():
        for _ in range(3):
            model(w)
    if device == 'cuda':
        torch.cuda.synchronize()

    lat = []
    for f in files:
        x = preprocess(np.load(f)).to(device)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            y = forward_tta(model, x) if args.tta else model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)

        out = y.squeeze().float().cpu().numpy()
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        np.save(os.path.join(args.output_dir, os.path.basename(f)),
                np.clip(out, 0.0, 1.0).astype(np.float32))

    lat = np.array(lat)
    print(f'{len(files)} images | device={device} | TTA={"on" if args.tta else "off"}')
    print('  mean %6.2f ms  median %6.2f ms  p95 %6.2f ms  total %.2f s'
          % (lat.mean(), np.median(lat), np.percentile(lat, 95), lat.sum() / 1000))


if __name__ == '__main__':
    main()
