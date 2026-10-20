"""
Single place for every path the pipeline reads or writes. Set the environment variables (see
configs/paths.env.example) or edit the defaults below; nothing else in the repository holds a
machine-specific path.
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('CSWSI_WORK', os.path.join(REPO, 'work'))

DATA = os.path.join(REPO, 'data')
MODELS = os.environ.get('CSWSI_MODELS', os.path.join(REPO, 'models'))

# inputs: CancerScout (BioImage Archive, on publication) and CPTAC-COAD (TCIA, scripts/00_download_cptac.sh)
SLIDES_CS = os.environ.get('CSWSI_SLIDES_CS', os.path.join(DATA, 'cancerscout', 'slides'))      # *.tiff
LABELS = os.environ.get('CSWSI_LABELS', os.path.join(DATA, 'cancerscout', 'labels'))            # <dataset>/{train,eval}.csv
STAIN_REFERENCE = os.environ.get('CSWSI_STAIN_REFERENCE',
                                 os.path.join(DATA, 'cancerscout', 'stain_reference'))          # 3 target tiles (*.png)
SLIDES_CPTAC = os.environ.get('CSWSI_SLIDES_CPTAC', os.path.join(DATA, 'cptac', 'slides'))      # *.svs
CPTAC_LABELS = os.environ.get('CSWSI_CPTAC_LABELS', os.path.join(REPO, 'configs', 'cptac_labels'))

# produced by the pipeline
PATCHES_CS = os.path.join(WORK, 'patches', 'cancerscout')
PATCHES_CPTAC = os.path.join(WORK, 'patches', 'cptac')
FEATURES_CS = os.path.join(WORK, 'features', 'cancerscout')                   # <encoder>/{h5,pt}_files
FEATURES_CPTAC = os.path.join(WORK, 'features', 'cptac')                      # <encoder>_<condition>/...
SPLITS = os.path.join(WORK, 'splits')
RESULTS = os.path.join(MODELS, 'clam')                                         # trained CLAM models (not released)
EVAL_DIR = os.environ.get('CSWSI_EVAL_DIR', os.path.join(WORK, 'eval_results'))
HEATMAPS = os.path.join(WORK, 'heatmaps')
TABLES = os.environ.get('CSWSI_TABLES', os.path.join(WORK, 'tables'))
FIGURES = os.environ.get('CSWSI_FIGURES', os.path.join(WORK, 'figures'))
STAINNET_WEIGHTS = os.path.join(MODELS, 'stainnet', 'stainnet_final.pt')            # trained StainNet (not released)

for _d in (TABLES, FIGURES):
    os.makedirs(_d, exist_ok=True)
