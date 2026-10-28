#!/usr/bin/env python3
"""
Download the trained models of the paper from Zenodo into models/ (or $CSWSI_MODELS).

The record holds one archive per encoder with the ten CLAM-MB models of every endpoint and their
per-fold decision thresholds, plus one archive with StainNet. Every archive is checked against
configs/model_archives.sha256 before it is unpacked, so a changed or truncated file is never used.

    python scripts/download_models.py                       # all encoders and StainNet
    ENCODERS="conch" python scripts/download_models.py      # a subset
    python scripts/download_models.py --from-dir /path/to/archives   # archives already on disk

The record is set in configs/zenodo.env (ZENODO_RECORD). Archives are kept in models/.archives/ so
a second run does not download again.
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import cswsi_config as cfg  # noqa: E402


def read_env(path):
    env = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"')
    return env


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from-dir', help='take the archives from this directory instead of Zenodo')
    ap.add_argument('--no-stainnet', action='store_true', help='skip the StainNet archive')
    args = ap.parse_args()

    zen = read_env(os.path.join(REPO, 'configs', 'zenodo.env'))
    record = os.environ.get('ZENODO_RECORD', zen.get('ZENODO_RECORD', ''))
    expected = dict(line.split()[::-1] for line in open(os.path.join(REPO, 'configs', 'model_archives.sha256'))
                    if line.strip())
    encoders = os.environ.get('ENCODERS', 'resnet uni conch').split()
    wanted = [f'cswsi_models_clam_{e}.zip' for e in encoders] + ([] if args.no_stainnet else ['cswsi_models_stainnet.zip'])

    cache = os.path.join(cfg.MODELS, '.archives')
    os.makedirs(cache, exist_ok=True)
    for name in wanted:
        if name not in expected:
            sys.exit(f'{name} is not listed in configs/model_archives.sha256')
        dst = os.path.join(cache, name)
        if not (os.path.exists(dst) and sha256(dst) == expected[name]):
            if args.from_dir:
                shutil.copy(os.path.join(args.from_dir, name), dst)
            else:
                if not record or record.startswith('TBD'):
                    sys.exit('ZENODO_RECORD is not set in configs/zenodo.env (see models/README.md)')
                url = f'https://zenodo.org/records/{record}/files/{name}?download=1'
                print('download', url, flush=True)
                urllib.request.urlretrieve(url, dst + '.part')
                os.replace(dst + '.part', dst)
            if sha256(dst) != expected[name]:
                os.remove(dst)
                sys.exit(f'checksum mismatch for {name}; file removed, run again')
        with zipfile.ZipFile(dst) as z:
            z.extractall(cfg.MODELS)
        print('ok', name, '->', cfg.MODELS, flush=True)


if __name__ == '__main__':
    main()
