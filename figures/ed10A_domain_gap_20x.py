#!/usr/bin/env python3
"""
Extended Data Fig. 10A: staining difference between discovery-cohort and CPTAC slides at the
matched 20x resolution, and its reduction by StainNet.

Three rows: thumbnails, example tiles raw -> StainNet, and per-channel intensity distributions of
tissue pixels with the mean per-channel Wasserstein distance to the discovery cohort.
  * tiles at 20x on both sides: discovery cohort 256 px at pyramid level 1 (the training tiles),
    CPTAC 512 px level-0 crops resized to 256 px (cptac/reextract_20x.py 'scale20x'), StainNet
    applied to the crop exactly as in the feature extraction;
  * one random tissue tile per discovery-cohort slide (2,496 slides) and four per CPTAC slide (372
    slides), 1,500 tissue pixels per tile; the reference is the discovery cohort itself.
Tile samples are cached in <tables>/ed10A_tile_sample.npz. Thumbnails: the four discovery-cohort and four
CPTAC slides listed under ed10a_thumbnails in configs/example_slides.yaml. Print size 6.9 in; fonts 7-8.5 pt.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from scipy.stats import wasserstein_distance

from paper_style import FULL_W, TAB, save, setup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402
sys.path.insert(0, os.path.join(cfg.REPO, 'stain_normalization'))
from stainnet import StainNet  # noqa: E402
from thumb_crop import crop_array  # noqa: E402

CS_SLIDES = cfg.SLIDES_CS
CS_H5 = f'{cfg.FEATURES_CS}/conch/h5_files'
CP_SLIDES = cfg.SLIDES_CPTAC
CP_H5 = f'{cfg.FEATURES_CPTAC}/conch_scale20x/h5_files'
DS = f'{cfg.LABELS}/tumor_vs_normal'
CACHE = f'{TAB}/ed10A_tile_sample.npz'
PX_PER_TILE = 1500
COL = {'CancerScout (discovery)': '#1f7a6f', 'CPTAC raw': '#c8791b', 'CPTAC StainNet': '#3f51b5'}

_model = None


def stainnet(arr):
    global _model
    if _model is None:
        _model = StainNet()
        _model.load_state_dict(torch.load(cfg.STAINNET_WEIGHTS, map_location='cpu'))
        _model.eval()
    t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = _model(t)[0].permute(1, 2, 0).numpy()
    return np.clip((out + 1) * 127.5, 0, 255).astype(np.uint8)


def tissue(arr, rng):
    px = arr.reshape(-1, 3)
    px = px[px.mean(1) < 220]
    if len(px) == 0:
        return px
    return px[rng.choice(len(px), min(PX_PER_TILE, len(px)), replace=False)]


def cs_job(args):
    slide, seed = args
    rng = np.random.default_rng(seed)
    with h5py.File(f'{CS_H5}/{slide}.h5', 'r') as h:
        c = h['coords'][:]
    x, y = c[rng.integers(len(c))]
    w = openslide.OpenSlide(f'{CS_SLIDES}/{slide}.tiff')
    t = np.array(w.read_region((int(x), int(y)), 1, (256, 256)).convert('RGB'))
    return slide, t, tissue(t, rng)


def cp_job(args):
    slide, seed, k = args
    rng = np.random.default_rng(seed)
    with h5py.File(f'{CP_H5}/{slide}.h5', 'r') as h:
        c = h['coords'][:]
    w = openslide.OpenSlide(f'{CP_SLIDES}/{slide}.svs')
    out = []
    for x, y in c[rng.choice(len(c), min(k, len(c)), replace=False)]:
        reg = w.read_region((int(x), int(y)), 0, (512, 512))
        white = Image.new('RGB', reg.size, (255, 255, 255))
        white.paste(reg, mask=reg.split()[-1])
        raw = np.array(white)
        norm = stainnet(raw)
        r256 = np.array(Image.fromarray(raw).resize((256, 256), Image.BILINEAR))
        n256 = np.array(Image.fromarray(norm).resize((256, 256), Image.BILINEAR))
        out.append((r256, n256, tissue(raw, rng), tissue(norm, rng)))
    return slide, out


def slide_thumbnail(path, long_edge=1500):
    """Whole-slide thumbnail with a 1,500 px long edge, cropped to the tissue."""
    w = openslide.OpenSlide(path)
    W, H = w.level_dimensions[0]
    sc = long_edge / max(W, H)
    return crop_array(np.array(w.get_thumbnail((int(W * sc), int(H * sc))).convert('RGB')))


def sample():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    cs = sorted(set(pd.concat([pd.read_csv(f'{DS}/{s}.csv') for s in ('train', 'eval')]).slide_id))
    cp = sorted(os.path.splitext(f)[0] for f in os.listdir(CP_SLIDES) if f.endswith('.svs'))
    with ProcessPoolExecutor(16) as ex:
        csr = list(ex.map(cs_job, [(s, i) for i, s in enumerate(cs)], chunksize=8))
        cpr = list(ex.map(cp_job, [(s, 10_000 + i, 4) for i, s in enumerate(cp)], chunksize=2))
    tiles_cs = np.stack([t for _, t, _ in csr])
    px_cs = np.concatenate([p for _, _, p in csr if len(p)])
    tiles_raw = np.stack([o[0] for _, out in cpr for o in out])
    tiles_sn = np.stack([o[1] for _, out in cpr for o in out])
    px_raw = np.concatenate([o[2] for _, out in cpr for o in out if len(o[2])])
    px_sn = np.concatenate([o[3] for _, out in cpr for o in out if len(o[3])])
    d = dict(tiles_cs=tiles_cs, tiles_raw=tiles_raw, tiles_sn=tiles_sn, px_cs=px_cs, px_raw=px_raw,
             px_sn=px_sn, n_cs_slides=len(cs), n_cp_slides=len(cp))
    np.savez_compressed(CACHE, **d)
    return d


def main():
    setup()
    d = sample()
    W = {k: [wasserstein_distance(d[k][:, c], d['px_cs'][:, c]) for c in range(3)] for k in ('px_raw', 'px_sn')}
    pd.DataFrame({'channel': ['R', 'G', 'B', 'mean'],
                  'W_raw': W['px_raw'] + [np.mean(W['px_raw'])],
                  'W_stainnet': W['px_sn'] + [np.mean(W['px_sn'])]}).to_csv(f'{TAB}/ed10A_wasserstein_20x.csv', index=False)
    print('W raw', np.round(W['px_raw'], 1), np.mean(W['px_raw']).round(1),
          '| W StainNet', np.round(W['px_sn'], 1), np.mean(W['px_sn']).round(1),
          '| tiles', len(d['tiles_cs']), len(d['tiles_raw']), '| px', len(d['px_cs']), len(d['px_raw']))

    fig = plt.figure(figsize=(FULL_W, 5.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.55, 1.15], hspace=0.28)
    # thumbnails
    import yaml
    th = yaml.safe_load(open(os.path.join(cfg.REPO, 'configs', 'example_slides.yaml')))['ed10a_thumbnails']
    cs_ids, cp_ids = th['cancerscout'], th['cptac']
    ga = gs[0].subgridspec(1, 8, wspace=0.05)
    for i, path in enumerate([f'{CS_SLIDES}/{s}.tiff' for s in cs_ids] + [f'{CP_SLIDES}/{s}.svs' for s in cp_ids]):
        ax = fig.add_subplot(ga[0, i]); ax.axis('off')
        ax.imshow(slide_thumbnail(path))
        if i in (0, 4):
            ax.set_title('CancerScout (discovery cohort)' if i == 0 else 'CPTAC-COAD (raw)', loc='left',
                         fontsize=7.5, fontweight='bold', color='#1f7a6f' if i == 0 else '#c8791b')
    # example tiles
    rng = np.random.default_rng(0)
    m = d['tiles_raw'].reshape(len(d['tiles_raw']), -1)
    cand = np.where((m.mean(1) > 120) & (m.mean(1) < 205) & (m.std(1) > 25))[0]
    sel = rng.choice(cand, 5, replace=False)
    gb = gs[1].subgridspec(2, 6, wspace=0.05, hspace=0.05)
    for j, ix in enumerate(sel):
        for row, key in ((0, 'tiles_raw'), (1, 'tiles_sn')):
            ax = fig.add_subplot(gb[row, j]); ax.imshow(d[key][ix]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel('Raw' if row == 0 else 'StainNet', fontsize=7.5, fontweight='bold',
                              color='#c8791b' if row == 0 else '#3f51b5', labelpad=3)
            if j == 0 and row == 0:
                ax.set_title('CPTAC tiles at 20x', loc='left', fontsize=7.5, fontweight='bold', color='#c8791b')
    csm = d['tiles_cs'].reshape(len(d['tiles_cs']), -1)
    cs_cand = np.where((csm.mean(1) > 120) & (csm.mean(1) < 205) & (csm.std(1) > 25))[0]
    for row in (0, 1):
        ax = fig.add_subplot(gb[row, 5]); ax.axis('off')
        ax.imshow(d['tiles_cs'][rng.choice(cs_cand)])
        if row == 0:
            ax.set_title('CancerScout', fontsize=7.5, color='#1f7a6f', fontweight='bold')
    # colour distributions
    gc = gs[2].subgridspec(1, 3, wspace=0.18)
    data = {'CancerScout (discovery)': d['px_cs'], 'CPTAC raw': d['px_raw'], 'CPTAC StainNet': d['px_sn']}
    for c, ch in enumerate('RGB'):
        ax = fig.add_subplot(gc[0, c])
        for lab, px in data.items():
            ax.hist(px[:, c], bins=64, range=(0, 255), density=True, histtype='step', lw=1.0, color=COL[lab],
                    label=lab)
        ax.set_title(f'{ch} channel, tissue pixels', fontsize=7.5)
        ax.set_xlim(0, 255); ax.set_yticks([])
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.text(0.02, 0.97, f'W raw {W["px_raw"][c]:.1f}\nW StainNet {W["px_sn"][c]:.1f}',
                transform=ax.transAxes, va='top', fontsize=6.8)
        ax.set_xlabel('Intensity')
        if c == 1:
            ax.legend(frameon=False, loc='upper center', fontsize=6.8, ncol=3, bbox_to_anchor=(0.5, -0.32),
                      handlelength=1.4, columnspacing=1.5)
    save(fig, 'ExtDataFig10A_domain_gap_20x')


if __name__ == '__main__':
    main()
