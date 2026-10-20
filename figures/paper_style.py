"""
Shared print style for the corrected CancerScout colon WSI panels.

Every panel is drawn at the physical width it occupies on the figure page (inches below), so the
font sizes set here are the sizes that appear in print: 7-8.5 pt, against ~3-4 pt effective in
the previous panels, which were drawn on 16-25 inch canvases and scaled down to 3-6 inches.
Colours, encoder palette and chrome are unchanged from plot_clam_combined.py.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402
FIG_OUT = cfg.FIGURES
TAB = cfg.TABLES
EVALD = cfg.EVAL_DIR

FULL_W = 6.9      # full page width used by the Extended Data / Supplementary panels (in)
METHODS = {'resnet': 'ResNet50', 'uni': 'UNI', 'conch': 'CONCH'}
METHOD_COLORS = {'ResNet50': '#2a78d6', 'UNI': '#e34948', 'CONCH': '#1baf7a'}
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASE = '#c3c2b7'

# manuscript names of the six proteomic subgroups (raw group code -> label)
SUBGROUPS = {'1': 'C1 Stromal ECM-rich', '2': 'C2 Mitochondrial CIN', '3': 'C3 Cytoskeletal CIN',
             '4': 'C4 MSI immune-active', '5': 'C5 Indolent CIN', '6': 'C6 Rare adverse state'}
SUBGROUP_COLORS = {'1': '#6E4FA2', '2': '#E07E3C', '3': '#8FB339', '4': '#F0C330',
                   '5': '#3A8EC0', '6': '#A07855'}


def setup():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.title_fontsize': 7.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.edgecolor': BASE,
        'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 600, 'figure.dpi': 150,
    })


def save(fig, name):
    os.makedirs(FIG_OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_OUT, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', os.path.join(FIG_OUT, name + '.{pdf,png}'))
