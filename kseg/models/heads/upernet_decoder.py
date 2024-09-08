import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from kappamodules.init import init_xavier_uniform_zero_bias
from kseg.modules.upernet_pyramid_pooling import UpernetPyramidPooling
from kseg.modules.conv_norm_act_2d import ConvNormAct2d
from ksuit.models import SingleModel


class UpernetDecoder(SingleModel):
    def __init__(self, pool_scales, dropout: float = 0., **kwargs):
        super().__init__(**kwargs)
        self.pool_scales = pool_scales
        self.dropout = dropout
        dim, _, _ = self.input_shape

        # lateral_convs
        self.lateral_convs = nn.ModuleList([
            ConvNormAct2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
            )
            for _ in range(len(pool_scales) - 1)
        ])

        # psp
        self.psp_module = UpernetPyramidPooling(
            in_channels=dim,
            dim=dim,
            pool_scales=self.pool_scales,
        )
        self.psp_bottleneck = ConvNormAct2d(
            in_channels=dim + len(self.pool_scales) * dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
        )

        # fpn + bottleneck
        self.fpn_convs = nn.ModuleList([
            ConvNormAct2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=1,
            )
            for _ in range(len(pool_scales) - 1)
        ])
        self.fpn_bottleneck = ConvNormAct2d(
            in_channels=dim + (len(self.pool_scales) - 1) * dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
        )

        # head
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, self.output_shape[0], kernel_size=1),
        )

        # init
        init_xavier_uniform_zero_bias(self)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        # build laterals
        laterals = [lateral_conv(x[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        psp_out = self.psp_bottleneck(torch.concat([x[-1], *self.psp_module(x[-1])], dim=1))
        laterals.append(psp_out)

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # interpolate is not supported in bfloat16
            with torch.autocast(device_type=str(x[0].device).split(":")[0], enabled=False):
                upsampled = interpolate(laterals[i].float(), size=prev_shape, mode="bilinear")
            laterals[i - 1] = laterals[i - 1] + upsampled

        # fpn
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(psp_out)
        # interpolate is not supported in bfloat16
        with torch.autocast(device_type=str(x[0].device).split(":")[0], enabled=False):
            for i in range(len(laterals) - 1, 0, -1):
                fpn_outs[i] = interpolate(fpn_outs[i].float(), size=fpn_outs[0].shape[2:], mode="bilinear")
        output = self.fpn_bottleneck(torch.concat(fpn_outs, dim=1))

        # classify
        output = self.head(output)
        # interpolate is not supported in bfloat16
        with torch.autocast(device_type=str(x[0].device).split(":")[0], enabled=False):
            output = interpolate(output.float(), size=self.output_shape[1:], mode="bilinear")
        return output
