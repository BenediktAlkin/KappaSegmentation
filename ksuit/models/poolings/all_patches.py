from .base.pooling_base import PoolingBase


class AllPatches(PoolingBase):
    def get_output_shape(self, input_shape):
        num_tokens, dim = input_shape
        if "num_aux_tokens" in self.static_ctx:
            num_patch_tokens = num_tokens - self.static_ctx["num_aux_tokens"]
        else:
            num_patch_tokens = num_tokens
        assert num_patch_tokens > 0
        return num_patch_tokens, dim

    def forward(self, all_tokens, *_, **__):
        if "num_aux_tokens" in self.static_ctx:
            return all_tokens[:, self.static_ctx["num_aux_tokens"]:]
        else:
            return all_tokens

    def __str__(self):
        return type(self).__name__
