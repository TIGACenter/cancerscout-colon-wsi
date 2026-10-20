"""
StainNet (Kang et al. 2021) architecture, reimplemented (3 layers, 32 channels, 1x1 convolutions):
a pure per-pixel color map with no spatial context and no normalization layers.
"""
import torch.nn as nn


class StainNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_layer=3, n_channel=32, kernel_size=1):
        super().__init__()
        layers = [nn.Conv2d(input_nc, n_channel, kernel_size=kernel_size, bias=True,
                             padding=kernel_size // 2),
                  nn.ReLU(True)]
        for _ in range(n_layer - 2):
            layers += [nn.Conv2d(n_channel, n_channel, kernel_size=kernel_size, bias=True,
                                  padding=kernel_size // 2),
                       nn.ReLU(True)]
        layers += [nn.Conv2d(n_channel, output_nc, kernel_size=kernel_size, bias=True,
                              padding=kernel_size // 2)]
        self.rgb_trans = nn.Sequential(*layers)

    def forward(self, x):
        return self.rgb_trans(x)
