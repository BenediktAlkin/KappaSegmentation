import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from ksuit.models import SingleModel
from kseg.modules.conv_norm_act_2d import ConvNormAct2d
from kappamodules.init import init_xavier_uniform_zero_bias


class UpernetAuxiliary(SingleModel):
    def __init__(
            self,
            feature_index,
            dim: int,
            num_hidden_layers: int = 0,
            kernel_size: int = 3,
            dilation: int = 1,
            dropout: float = 0.,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_index = feature_index
        input_dim, _, _ = self.input_shape
        self.dim = dim
        self.num_hidden_layers = num_hidden_layers
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout

        assert kernel_size % 2 == 1
        padding = kernel_size // 2 * dilation
        conv_kwargs = dict(
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.stem = ConvNormAct2d(in_channels=input_dim, out_channels=self.dim, **conv_kwargs)
        self.hidden = nn.Sequential(
            *[
                ConvNormAct2d(in_channels=self.dim, out_channels=self.dim, **conv_kwargs)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(self.dim, self.output_shape[0], kernel_size=1),
        )

        # initialize
        init_xavier_uniform_zero_bias(self)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = x[self.feature_index]
        x = self.stem(x)
        x = self.hidden(x)
        x = self.head(x)
        # interpolate is not supported in bfloat16
        with torch.autocast(device_type=str(x.device).split(":")[0], enabled=False):
            x = interpolate(x.float(), size=self.output_shape[1:], mode="bilinear", align_corners=False)
        return x
