"""
Build the (source, teacher) training pairs for StainNet: sample CPTAC tiles across the
50-slide subset, generate a teacher target for each via the FIXED-source-stats Macenko
transform (macenko.normalize_fixed_source: source stain vectors and maximum fitted once on the
ten CPTAC tiles listed in configs/stainnet_source_pool.csv, read from the slides, target fitted once on the three discovery-cohort reference
tiles, so the teacher is one consistent color map rather than a per-tile refit), and save both as
uint8 arrays for train_stainnet.py.
"""
import glob
import os
import sys

import h5py
import numpy as np
import openslide
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import macenko  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402
SRC_PATCHES_DIR = f'{cfg.PATCHES_CPTAC}/patches'   # CPTAC 256 px level-0 patch coordinates
SLIDE_DIR = cfg.SLIDES_CPTAC
REF_TILES_DIR = cfg.STAIN_REFERENCE                 # three discovery-cohort target tiles (data release)
SOURCE_POOL = os.path.join(cfg.REPO, 'configs', 'stainnet_source_pool.csv')  # ten CPTAC tiles: slide_id, level-0 x, y
OUT_DIR = os.path.join(cfg.WORK, 'stainnet', 'data')
PATCH_SIZE = 256
TILES_PER_SLIDE = 40
SEED = 11


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ref_paths = sorted(glob.glob(f'{REF_TILES_DIR}/*.png'))
    ref_images = [np.array(Image.open(p).convert('RGB')) for p in ref_paths]
    ref_stain_matrix, ref_max_c = macenko.fit_reference(ref_images)

    pool = pd.read_csv(SOURCE_POOL)
    src_images = [np.array(openslide.OpenSlide(f'{SLIDE_DIR}/{r.slide_id}.svs')
                           .read_region((int(r.x), int(r.y)), 0, (PATCH_SIZE, PATCH_SIZE)).convert('RGB'))
                  for r in pool.itertuples()]
    src_stain_matrix, src_max_c = macenko.fit_reference(src_images)

    print('ref (CancerScout) stain matrix:\n', ref_stain_matrix)
    print('src (CPTAC, fixed) stain matrix:\n', src_stain_matrix)

    subset = pd.read_csv(os.path.join(cfg.REPO, 'configs', 'stainnet_training_slides.csv'))
    slides = subset['slide_id'].tolist()

    rng = np.random.RandomState(SEED)
    sources, teachers = [], []
    n_failed = 0

    for i, slide_id in enumerate(slides):
        with h5py.File(f'{SRC_PATCHES_DIR}/{slide_id}.h5', 'r') as hf:
            coords = hf['coords'][:]
        wsi = openslide.OpenSlide(f'{SLIDE_DIR}/{slide_id}.svs')
        idxs = rng.choice(len(coords), size=min(TILES_PER_SLIDE, len(coords)), replace=False)
        for ix in idxs:
            x, y = coords[ix]
            raw = np.array(wsi.read_region((int(x), int(y)), 0, (PATCH_SIZE, PATCH_SIZE)).convert('RGB'))
            try:
                teacher = macenko.normalize_fixed_source(raw, src_stain_matrix, src_max_c,
                                                           ref_stain_matrix, ref_max_c)
            except ValueError:
                n_failed += 1
                continue
            sources.append(raw)
            teachers.append(teacher)
        if (i + 1) % 10 == 0:
            print(f'{i+1}/{len(slides)} slides, {len(sources)} pairs so far', flush=True)

    sources = np.stack(sources)
    teachers = np.stack(teachers)
    print(f'Done. {len(sources)} pairs ({n_failed} tiles skipped: too little tissue signal).')
    np.save(f'{OUT_DIR}/sources.npy', sources)
    np.save(f'{OUT_DIR}/teachers.npy', teachers)
    np.save(f'{OUT_DIR}/ref_stain_matrix.npy', ref_stain_matrix)
    np.save(f'{OUT_DIR}/ref_max_c.npy', ref_max_c)
    np.save(f'{OUT_DIR}/src_stain_matrix.npy', src_stain_matrix)
    np.save(f'{OUT_DIR}/src_max_c.npy', src_max_c)


if __name__ == '__main__':
    main()
