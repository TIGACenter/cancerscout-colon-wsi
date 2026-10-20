"""
Tissue tiling of the discovery-cohort (CancerScout colon) H&E slides: 256 px tiles at pyramid level 1
(20x, ~0.5 um/px), non-overlapping, written as CLAM patch files (<out>/patches/<slide_id>.h5, dataset
'coords' with level-0 coordinates) that extract_features_fp.py reads.

The tissue mask is computed once per slide on the level-6 thumbnail and resized to the tile grid:
  1. the flat red that the scanner writes into unscanned areas (RGB between (180, 0, 0) and
     (255, 30, 30)) is set to white;
  2. background = (mean RGB > 210) AND (no texture: |Laplacian| of the gray image, median-blurred
     with a 15 px kernel, <= 5), followed by a binary opening;
  3. the background mask is resized to the 256 px tile grid at level 1 (nearest neighbor), inverted
     to a tissue mask, and tissue islands and holes smaller than 20 tiles are removed;
  4. every tissue tile becomes one coordinate.
Masks are also saved as PNG (<out>/stitches/<slide_id>.png) for inspection.

Usage: python tile_cancerscout.py --source <dir of .tiff> --save_dir <out>
"""
import argparse
import glob
import os

import cv2
import h5py
import numpy as np
import openslide
import pandas as pd
from scipy import ndimage
from skimage.morphology import remove_small_holes, remove_small_objects

TILE = 256
LEVEL = 1             # pyramid level of the tiles (20x)
THUMB_LEVEL = 6       # pyramid level the tissue mask is computed on
MIN_TILES = 20        # smallest tissue island / hole kept, in tiles


def red_to_white(arr):
    """Set the scanner's flat red fill (unscanned area) to white."""
    arr = arr.copy()
    red = np.all((arr >= [180, 0, 0]) & (arr <= [255, 30, 30]), axis=-1)
    arr[red] = 255
    return arr


def background_mask(rgb):
    """Bright and texture-free pixels, opened once."""
    color = np.mean(rgb, axis=-1) > 210
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.medianBlur(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)).astype('uint8'), 15)
    _, no_texture = cv2.threshold(lap, 5, 255, cv2.THRESH_BINARY_INV)
    return ndimage.binary_opening(color * no_texture)


def tissue_tiles(wsi):
    """Boolean (rows, cols) mask over the 256 px tile grid at LEVEL."""
    thumb = np.array(wsi.get_thumbnail(wsi.level_dimensions[THUMB_LEVEL]))[..., :3]
    w, h = wsi.level_dimensions[LEVEL]
    bg = cv2.resize(background_mask(red_to_white(thumb)).astype(np.int64) * 1, (w // TILE, h // TILE),
                    interpolation=cv2.INTER_NEAREST)
    tissue = (1 - bg).astype(bool)
    tissue = remove_small_objects(tissue, min_size=MIN_TILES, connectivity=1)
    return remove_small_holes(tissue, area_threshold=MIN_TILES, connectivity=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='directory with the slides (*.tiff)')
    ap.add_argument('--save_dir', required=True)
    args = ap.parse_args()
    os.makedirs(f'{args.save_dir}/patches', exist_ok=True)
    os.makedirs(f'{args.save_dir}/stitches', exist_ok=True)
    rows = []
    for path in sorted(glob.glob(f'{args.source}/*.tiff')):
        slide_id = os.path.splitext(os.path.basename(path))[0]
        out = f'{args.save_dir}/patches/{slide_id}.h5'
        if not os.path.exists(out):
            wsi = openslide.OpenSlide(path)
            mask = tissue_tiles(wsi)
            if not mask.any():
                print(f'{slide_id}: no tissue tiles, skipped')
                continue
            cv2.imwrite(f'{args.save_dir}/stitches/{slide_id}.png', mask.astype(np.uint8) * 255)
            ys, xs = np.where(mask)
            coords = np.stack((xs, ys), axis=1) * TILE * 2 ** LEVEL
            with h5py.File(out, 'w') as f:
                d = f.create_dataset('coords', data=coords)
                d.attrs.update({'patch_level': LEVEL, 'patch_size': TILE, 'downsample': [2 ** LEVEL] * 2,
                                'level_dim': wsi.level_dimensions[LEVEL],
                                'downsampled_level_dim': wsi.level_dimensions[LEVEL], 'name': slide_id})
            print(f'{slide_id}: {len(coords)} tiles', flush=True)
        rows.append({'slide_id': slide_id + '.tiff', 'process': 1, 'status': 'processed'})
    # CLAM's feature extraction reads the slide list from this file
    pd.DataFrame(rows).to_csv(f'{args.save_dir}/process_list_autogen.csv', index=False)


if __name__ == '__main__':
    main()
