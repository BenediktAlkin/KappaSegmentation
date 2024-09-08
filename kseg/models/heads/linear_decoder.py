import torch
import torch.nn as nn
from kappamodules.init import init_xavier_uniform_zero_bias
import torch.nn.functional as F

from ksuit.models import SingleModel
from ksuit.factory import MasterFactory

class LinearDecoder(SingleModel):
    def __init__(self, pooling, **kwargs):
        super().__init__(**kwargs)
        self.pooling = MasterFactory.get("pooling").create(pooling, static_ctx=self.static_ctx)
        dim, _, _ = self.pooling.get_output_shape(self.input_shape)
        self.head = nn.Conv2d(dim, self.output_shape[0], kernel_size=1)

        # initialize
        init_xavier_uniform_zero_bias(self)

    def forward(self, x):
        x = self.pooling(x)
        x = self.head(x)
        # interpolate is not supported in bfloat16
        with torch.autocast(device_type=str(x.device).split(":")[0], enabled=False):
            x = F.interpolate(x.float(), size=self.output_shape[1:], mode="bilinear", align_corners=False)
        return x
