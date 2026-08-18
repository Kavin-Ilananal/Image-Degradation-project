"""Competition entry point — Three-Stage Cascaded RCAN.

    python run.py <input-dir> <output-dir>

Reads every .npy in <input-dir>, restores it, and writes one .npy per input to
<output-dir> with the same filename. The output directory is created if it does
not exist.

Output contract:
  - grayscale, shape (H*2, W*2)
  - float32, values within [0, 1]
  - no NaN, no Inf

Runs on GPU when available and falls back to CPU otherwise. Requires no internet
access, no API keys, no additional downloads and no manual configuration: the
trained weights ship in models/ beside this script, and every path resolves
relative to this file rather than the working directory.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, 'models', 'rcan_phase1_psnr.pth')
SEED = 42

try:
    import cv2
    HAVE_CV2 = True
except ImportError:                       # keep running rather than crash
    HAVE_CV2 = False


def preprocess(noisy):
    """3-channel input: raw | bilateral | log-Fourier speckle estimate.

    The RAW channel is deliberately left unclipped — values exceed 1.0 in this
    dataset and that excursion carries signal. Only the two derived channels use
    a clipped copy, since both operate in uint8 / log space.
    """
    noisy = np.squeeze(noisy).astype(np.float32)
    raw = torch.from_numpy(noisy).float()
    clipped = np.clip(noisy, 0.0, 1.0)

    if HAVE_CV2:
        u8 = (clipped * 255.0).astype(np.uint8)
        bil = cv2.bilateralFilter(u8, 9, 75, 75).astype(np.float32) / 255.0
    else:
        t = torch.from_numpy(clipped)[None, None]
        bil = F.avg_pool2d(F.pad(t, (2, 2, 2, 2), mode='reflect'), 5, 1)[0, 0].numpy()

    # speckle is multiplicative, so it is additive in the log domain: low-pass
    # there, then exponentiate back
    fs = np.fft.fftshift(np.fft.fft2(np.log(clipped + 1e-5)))
    rows, cols = clipped.shape
    y, x = np.ogrid[:rows, :cols]
    mask = np.exp(-(((x - cols // 2) ** 2 + (y - rows // 2) ** 2)) / (2.0 * 15 ** 2))
    sp = np.exp(np.real(np.fft.ifft2(np.fft.ifftshift(fs * mask))))
    sp = (sp - sp.min()) / (sp.max() - sp.min() + 1e-8)

    return torch.stack([raw,
                        torch.from_numpy(np.ascontiguousarray(bil)).float(),
                        torch.from_numpy(sp.astype(np.float32))], 0).unsqueeze(0)


def load_model(device):
    model = build_model()
    obj = torch.load(CKPT, map_location=device)
    state = obj.get('model_state_dict', obj) if isinstance(obj, dict) else obj
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def restore(model, x):
    """Single pass, with the raw input plane as a fallback anchor."""
    out = model(x).float()
    if not torch.isfinite(out).all():
        out = F.interpolate(x[:, :1].float(), scale_factor=2,
                            mode='bilinear', align_corners=False)
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    if float(out.max() - out.min()) < 1e-6:          # degenerate: use the anchor
        out = F.interpolate(x[:, :1].float(), scale_factor=2,
                            mode='bilinear', align_corners=False)
    return out.clamp(0.0, 1.0)


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python run.py <input-dir> <output-dir>')
    input_dir, output_dir = sys.argv[1], sys.argv[2]

    if not os.path.isdir(input_dir):
        sys.exit(f'input directory not found: {input_dir}')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.endswith('.npy'))
    if not files:
        sys.exit(f'no .npy files found in {input_dir}')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device)
    print(f'Three-Stage Cascaded RCAN | {len(files)} images | device={device}')

    w = preprocess(np.load(os.path.join(input_dir, files[0]))).to(device)
    for _ in range(3):
        restore(model, w)
    if device == 'cuda':
        torch.cuda.synchronize()

    lat = []
    for name in files:
        x = preprocess(np.load(os.path.join(input_dir, name))).to(device)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = restore(model, x)
        if device == 'cuda':
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        np.save(os.path.join(output_dir, name),
                y.squeeze().cpu().numpy().astype(np.float32))

    lat = np.array(lat)
    print('  mean %.2f ms  median %.2f ms  total %.2f s'
          % (lat.mean(), np.median(lat), lat.sum() / 1000))
    print(f'  wrote {len(files)} files to {output_dir}')


if __name__ == '__main__':
    main()
