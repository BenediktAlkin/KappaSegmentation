import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from kseg.modules.conv_norm_act_2d import ConvNormAct2d


class UpernetPyramidPooling(nn.Module):
    def __init__(self, in_channels: int, dim: int, pool_scales):
        super().__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvNormAct2d(in_channels, dim, kernel_size=1),
                )
                for pool_scale in pool_scales
            ]
        )

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            output = block(x)
            # interpolate is not supported in bfloat16
            with torch.autocast(device_type=str(x.device).split(":")[0], enabled=False):
                upsampled = interpolate(output.float(), size=x.size()[2:], mode="bilinear")
            outputs.append(upsampled)
        return outputs
