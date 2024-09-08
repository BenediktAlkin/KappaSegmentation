import torch.nn as nn


class ConvNormAct2d(nn.Module):
    def __init__(self, *args, bias=False, **kwargs):
        super().__init__()
        assert not bias
        self.conv = nn.Conv2d(*args, bias=bias, **kwargs)
        self.norm = nn.BatchNorm2d(self.conv.out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
