"""load_crop, copied verbatim from analysis/stain_normalization/plot_cohort_comparison.py (importing that
module re-renders a thesis figure as a side effect)."""
import matplotlib.image as mpimg
import numpy as np


def load_crop(path, margin=0.0):
    """Crop a thumbnail to its tissue bounding box, ignoring near-white and near-red fill."""
    return crop_array(mpimg.imread(path), margin)


def crop_array(im, margin=0.0):
    if im.dtype != np.uint8:
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    im = im[..., :3]
    R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    near_white = (R > 225) & (G > 225) & (B > 225)
    near_red = (R > 180) & (G < 90) & (B < 90)
    tissue = ~near_white & ~near_red
    if tissue.sum() < 50:
        return im

    def largest_run(density):
        """Largest contiguous run of rows/cols above a tissue-density threshold.

        Using the largest contiguous block (not global min/max) means an isolated
        tissue speck stranded inside the red fill region cannot drag the crop box
        back across the red.
        """
        keep = density > density.max() * 0.08
        idx = np.where(keep)[0]
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        best = max(runs, key=len)
        return best.min(), best.max()

    x0, x1 = largest_run(tissue.sum(0))
    y0, y1 = largest_run(tissue.sum(1))
    h, w = im.shape[:2]
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1 = int(max(0, x0 - mx)), int(min(w, x1 + mx))
    y0, y1 = int(max(0, y0 - my)), int(min(h, y1 + my))
    return im[y0:y1, x0:x1]


