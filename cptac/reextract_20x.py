"""
CPTAC-COAD tile embeddings at the discovery cohort's resolution and field of view (20x, ~128 um).

The discovery cohort was tiled at pyramid level 1 (~0.5 um/px): 256 px tiles = ~128 um. CPTAC-COAD
slides are scanned at 40x (0.25 um/px) and are tiled by CLAM at level 0 with 256 px tiles
(scripts/01_patch.sh). This script keeps every second row and column of that lattice, reads a
512 px level-0 crop at each kept coordinate (= 128 um, non-overlapping), optionally applies
StainNet toward the discovery cohort, and lets the frozen encoder's transform resize the crop to
224 px, as in the discovery-cohort feature extraction.

Conditions (the two in the paper):
  scale20x           raw pixels
  scale20x_stainnet  StainNet-normalized pixels (weights: cswsi_config.STAINNET_WEIGHTS)

Output: <work>/features/cptac/<encoder>_<condition>/{h5,pt}_files/<slide_id>.{h5,pt}, the layout
CLAM's eval_cancerscout.py reads.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'stain_normalization'))
sys.path.insert(0, os.path.join(_REPO, 'clam'))
import cswsi_config as cfg  # noqa: E402
from stainnet import StainNet  # noqa: E402
from models import get_encoder  # noqa: E402

PATCH_SIZE = 256        # CLAM lattice on CPTAC level 0
CROP = 512              # level-0 crop = 128 um = one discovery-cohort tile
TARGET_IMG_SIZE = 224   # encoder input, as for the discovery cohort
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER_SHORT = {'conch_v1': 'conch', 'resnet50_trunc': 'resnet', 'uni_v1': 'uni'}
ENCODER_EMBED_DIM = {'conch_v1': 512, 'resnet50_trunc': 1024, 'uni_v1': 1024}


def decimate_coords(coords, factor=CROP // PATCH_SIZE):
    """Keep a non-overlapping sublattice, so the `factor`x larger crops do not overlap.

    Uses the parity of the floor-divided grid index rather than a residue against a single
    per-slide origin: CLAM anchors the patch grid separately per tissue contour, so the
    sub-PATCH_SIZE offset can differ between regions of one slide. Floor division is invariant to
    that offset: for x = off + 256k with 0 <= off < 256, x // 256 == k.
    """
    kx = coords[:, 0] // PATCH_SIZE
    ky = coords[:, 1] // PATCH_SIZE
    keep = ((kx % factor) == 0) & ((ky % factor) == 0)
    return coords[keep]


class CropBag(Dataset):
    def __init__(self, slide_path, coords, condition, img_transforms):
        self.slide_path, self.coords, self.condition = slide_path, coords, condition
        self.img_transforms = img_transforms
        self._wsi = None
        self._stainnet = None

    def _get_wsi(self):
        if self._wsi is None:
            self._wsi = openslide.OpenSlide(self.slide_path)
        return self._wsi

    def _apply_stainnet(self, arr):
        if self._stainnet is None:
            self._stainnet = StainNet()
            self._stainnet.load_state_dict(torch.load(cfg.STAINNET_WEIGHTS, map_location='cpu'))
            self._stainnet.eval()
        t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = self._stainnet(t)[0].permute(1, 2, 0).numpy()
        return np.clip((out + 1) * 127.5, 0, 255).astype(np.uint8)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        region = self._get_wsi().read_region((int(x), int(y)), 0, (CROP, CROP))
        # a crop can run past the slide edge, where openslide returns transparent pixels;
        # composite onto white so out-of-bounds reads look like empty glass, not black tissue
        white = Image.new('RGB', region.size, (255, 255, 255))
        white.paste(region, mask=region.split()[-1])
        arr = np.array(white)
        if self.condition == 'scale20x_stainnet':
            arr = self._apply_stainnet(arr)
        return {'img': self.img_transforms(Image.fromarray(arr)), 'coord': np.array([x, y])}


def process_slide(slide_id, model, img_transforms, condition, dst_dir, batch_size, num_workers, embed_dim):
    with h5py.File(f'{cfg.PATCHES_CPTAC}/patches/{slide_id}.h5', 'r') as hf:
        coords = decimate_coords(hf['coords'][:])
    loader = DataLoader(CropBag(f'{cfg.SLIDES_CPTAC}/{slide_id}.svs', coords, condition, img_transforms),
                        batch_size=batch_size, num_workers=num_workers, pin_memory=(DEVICE == 'cuda'))
    feats, pos = np.empty((len(coords), embed_dim), dtype=np.float32), 0
    for batch in loader:
        with torch.inference_mode():
            f = model(batch['img'].to(DEVICE, non_blocking=True)).cpu().numpy().astype(np.float32)
        feats[pos:pos + len(f)] = f
        pos += len(f)
    os.makedirs(f'{dst_dir}/h5_files', exist_ok=True)
    os.makedirs(f'{dst_dir}/pt_files', exist_ok=True)
    with h5py.File(f'{dst_dir}/h5_files/{slide_id}.h5', 'w') as hf:
        hf.create_dataset('features', data=feats)
        hf.create_dataset('coords', data=coords)
    torch.save(torch.from_numpy(feats), f'{dst_dir}/pt_files/{slide_id}.pt')
    return len(coords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_list', type=str, required=True, help='one slide id (file name without .svs) per line')
    parser.add_argument('--model_name', type=str, required=True, choices=list(ENCODER_SHORT))
    parser.add_argument('--condition', type=str, required=True, choices=['scale20x', 'scale20x_stainnet'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--skip_existing', action='store_true', help='skip slides whose .pt file already exists')
    args = parser.parse_args()

    if args.condition == 'scale20x_stainnet' and not os.path.exists(cfg.STAINNET_WEIGHTS):
        sys.exit(f'StainNet weights not found: {cfg.STAINNET_WEIGHTS} (see models/README.md)')
    slides = [l.strip() for l in open(args.slide_list) if l.strip()]
    dst_dir = f'{cfg.FEATURES_CPTAC}/{ENCODER_SHORT[args.model_name]}_{args.condition}'
    model, img_transforms = get_encoder(args.model_name, target_img_size=TARGET_IMG_SIZE)
    model = model.eval().to(DEVICE)
    for i, slide_id in enumerate(slides):
        if args.skip_existing and os.path.exists(f'{dst_dir}/pt_files/{slide_id}.pt'):
            continue
        t0 = time.time()
        n = process_slide(slide_id, model, img_transforms, args.condition, dst_dir, args.batch_size,
                          args.num_workers, ENCODER_EMBED_DIM[args.model_name])
        dt = time.time() - t0
        print(f'[{i + 1}/{len(slides)}] {slide_id}: {n} tiles, {dt:.1f}s', flush=True)
    print('Done.')


if __name__ == '__main__':
    main()
