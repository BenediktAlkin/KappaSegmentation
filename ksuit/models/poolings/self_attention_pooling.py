import einops
import torch.nn.functional as F
from torch import nn

from .base.pooling_base import PoolingBase


class SelfAttentionPooling(PoolingBase):
    def __init__(self, dim, num_tokens, aggregate="flatten", **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.weights = nn.Linear(dim, num_tokens)
        self.aggregate = aggregate

    def get_output_shape(self, input_shape):
        _, dim = input_shape
        if self.aggregate == "flatten":
            return self.num_tokens * dim
        raise NotImplementedError

    def forward(self, all_tokens, *_, **__):
        assert all_tokens.ndim == 3
        weights = F.softmax(self.weights(all_tokens), dim=1)
        pooled = einops.einsum(all_tokens, weights, "bs seqlen dim, bs seqlen num_tokens -> bs num_tokens dim")
        if self.aggregate == "flatten":
            return einops.rearrange(pooled, "bs num_tokens dim -> bs (num_tokens dim)")
        raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}(num_tokens={self.num_tokens})"
