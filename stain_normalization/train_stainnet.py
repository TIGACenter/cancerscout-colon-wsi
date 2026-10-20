"""
Train StainNet: source (CPTAC) -> teacher (fixed-source Macenko toward CancerScout)
per-pixel regression. L1 loss on [-1,1]-scaled images, Adam lr=2e-4 (not the original paper's SGD - SGD
underfit on a small/low-thousands-tile dataset within a reasonable epoch budget),
batch ~32, ~25 epochs with cosine-annealed LR, checkpoint saved every epoch (not just
at the end - avoids losing an entire run to a silent mid-run death).

No random-crop augmentation: StainNet is a pure per-pixel function (1x1 convs, no
spatial context), so cropping different sub-regions of a tile each epoch provides no
information a per-pixel model could use beyond just having a diverse pool of pixel
colors in the first place - which the 2000-tile source pool (build_stainnet_dataset.py,
40 tiles x 50 slides) already provides directly.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from stainnet import StainNet  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402
DATA_DIR = os.path.join(cfg.WORK, 'stainnet', 'data')
CKPT_DIR = os.path.join(cfg.WORK, 'stainnet', 'checkpoints')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 25
BATCH_SIZE = 32
LR = 2e-4


class PairDataset(Dataset):
    def __init__(self, sources, teachers):
        self.sources = sources
        self.teachers = teachers

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src = self.sources[idx].astype(np.float32) / 127.5 - 1.0
        tgt = self.teachers[idx].astype(np.float32) / 127.5 - 1.0
        src = torch.from_numpy(src).permute(2, 0, 1)  # CHW
        tgt = torch.from_numpy(tgt).permute(2, 0, 1)
        return src, tgt


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    sources = np.load(f'{DATA_DIR}/sources.npy')
    teachers = np.load(f'{DATA_DIR}/teachers.npy')
    print(f'{len(sources)} training pairs')

    dataset = PairDataset(sources, teachers)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    model = StainNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.L1Loss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            out = model(src)
            loss = loss_fn(out, tgt)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * src.size(0)
        sched.step()
        epoch_loss /= len(dataset)
        print(f'epoch {epoch+1}/{EPOCHS}: L1 loss = {epoch_loss:.5f}, lr = {sched.get_last_lr()[0]:.2e}',
              flush=True)
        torch.save(model.state_dict(), f'{CKPT_DIR}/stainnet_epoch{epoch+1}.pt')

    torch.save(model.state_dict(), f'{CKPT_DIR}/stainnet_final.pt')
    print('Done.')


if __name__ == '__main__':
    main()
