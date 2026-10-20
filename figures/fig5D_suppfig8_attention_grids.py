#!/usr/bin/env python3
"""
Fig. 5D (the C2 example) and Supplementary Fig. 8 (one example per proteomic subgroup, C1-C6):
CLAM attention maps of the CONCH models on held-out discovery slides.

Input: the CLAM blockmaps written by scripts/11_heatmaps.sh (<work>/heatmaps/raw_cancerscout/
HEATMAP_pubfig_*), the H&E slides and the slide-level labels. Example slides, the fold whose map is
shown, and crops are read from configs/example_slides.yaml.
  * no case accession numbers in the panel titles or row headers;
  * print size 6.9 in wide, 7-8 pt titles;
  * the C5 example holds two serial sections side by side (aspect 2.2), so its panels are cropped to
    the left section (cancerscout_example_crop in the config), the same crop for the H&E and all four
    attention maps;
  * the flat red that the scanner vendor writes into unscanned areas (pure RGB ~254,0,0) is removed:
    it is located on the H&E thumbnail (pure-red pixels plus the anti-aliased reddish pixels touching
    them), widened by 0.4% of the width, and painted with each image's own background color in the
    H&E and in all four attention maps of that row. Attention reds never reach pure red.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import openslide
from scipy import ndimage
import pandas as pd
from PIL import Image

from paper_style import FULL_W, SUBGROUPS, save, setup

Image.MAX_IMAGE_PIXELS = None
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402
import yaml  # noqa: E402
RAW = cfg.HEATMAPS + '/raw_cancerscout'
SLIDES = cfg.SLIDES_CS
DS = cfg.LABELS
EX = yaml.safe_load(open(os.path.join(cfg.REPO, 'configs', 'example_slides.yaml')))
# one held-out example slide per subgroup code (C1-C6), as released
GROUPS = {str(k): v for k, v in EX['cancerscout_subgroup_examples'].items()}
TASKS = [('proteomic_classification', 'Proteomic subgroup'), ('cptac', 'CPTAC subtype'),
         ('tumor_vs_normal', 'Tumor detection'), ('msi_status_consensus', 'MSI status (consensus)')]


def labels(task):
    m = {}
    for s in ('train', 'eval'):
        p = f'{DS}/{task}/{s}.csv'
        if os.path.exists(p):
            m.update(pd.read_csv(p).set_index('slide_id').label.astype(str).to_dict())
    return m


LAB = {t: labels(t) for t, _ in TASKS[1:]}


def blockmap(task, slide):
    for exp in sorted(glob.glob(f'{RAW}/HEATMAP_pubfig_{task}_{slide}_fold*')):
        hits = glob.glob(f'{exp}/*/{slide}/{slide}_blockmap.png')
        if hits:
            return hits[0]
    raise FileNotFoundError(task, slide)


def red_fill_mask(img):
    """Pixels of the vendor's flat red fill (pure RGB ~254,0,0) plus the anti-aliased edge around it."""
    a = np.asarray(img.convert('RGB')).astype(np.int16)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    seed = (R >= 235) & (G <= 30) & (B <= 30)
    reddish = (R > 120) & (R - G > 60) & (R - B > 60)
    return seed | (reddish & ndimage.binary_dilation(seed, iterations=3))


def paint(img, mask, grow):
    """Paint mask (grown by `grow` px) with the image's own background color."""
    a = np.asarray(img.convert('RGB')).astype(np.int16)
    if mask.shape != a.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((a.shape[1], a.shape[0]),
                                                                            Image.NEAREST)) > 0
    mask = mask | red_fill_mask(img)
    if not mask.any():
        return Image.fromarray(a.astype(np.uint8))
    mask = ndimage.binary_dilation(mask, iterations=grow)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    white = (R > 215) & (G > 215) & (B > 215) & ~mask
    bg = np.median(a[white], axis=0) if white.sum() > 100 else np.array([240, 240, 240])
    a[mask] = bg
    return Image.fromarray(a.astype(np.uint8))


# Crop box (fraction of width) for slides with two sections side by side: keep the left one.
CROP_X = {GROUPS[str(k)]: tuple(v) for k, v in (EX.get('cancerscout_example_crop') or {}).items()}


def crop(img, slide):
    if slide not in CROP_X:
        return img
    x0, x1 = CROP_X[slide]
    return img.crop((int(x0 * img.width), 0, int(x1 * img.width), img.height))


def row(axes, group, slide, header=True):
    w = openslide.OpenSlide(f'{SLIDES}/{slide}.tiff')
    lvl = min(5, w.level_count - 1)
    he = w.read_region((0, 0), lvl, w.level_dimensions[lvl]).convert('RGB')
    # the red fill is located on the H&E and blanked at the same place in every panel of the row: the
    # attention maps were drawn over it, so there it is blended and no longer pure red
    fill = red_fill_mask(he)
    grow = max(3, round(0.004 * he.width))
    axes[0].imshow(crop(paint(he, fill, grow), slide))
    axes[0].set_title('H&E\n' if header else '', fontsize=7.2, linespacing=1.1)
    for ax, (task, name) in zip(axes[1:], TASKS):
        ax.imshow(crop(paint(Image.open(blockmap(task, slide)), fill, grow), slide))
        truth = SUBGROUPS[group] if task == 'proteomic_classification' else LAB[task][slide]
        ax.set_title(f'{name}\n{truth}', fontsize=7.2, linespacing=1.1)
    for ax in axes:
        ax.axis('off')


def main():
    setup()
    fig, axes = plt.subplots(1, 5, figsize=(FULL_W, 1.6))
    row(axes, '2', GROUPS['2'])
    fig.tight_layout(w_pad=0.3)
    save(fig, 'Fig5D_attention_C2')
    shown = dict(GROUPS)   # C1..C6 in order; C5 cropped to one of its two sections (see CROP_X)
    fig, axes = plt.subplots(len(shown), 5, figsize=(FULL_W, 1.6 * len(shown)))
    for i, (g, s) in enumerate(shown.items()):
        row(axes[i], g, s, header=True)
    fig.tight_layout(h_pad=1.1, w_pad=0.3)
    save(fig, 'SuppFig8_attention_by_subgroup')


if __name__ == '__main__':
    main()
