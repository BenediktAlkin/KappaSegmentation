from ksuit.initializers import TorchhubInitializer


class TorchhubVitInitializer(TorchhubInitializer):
    def _get_model_kwargs(self):
        model = self.model.lower().replace("_", "")
        if "deit3-tiny" in model:
            return dict(patch_size=16, dim=192, num_attn_heads=3, depth=12)
        if "small8" in model or "s8" in model:
            return dict(patch_size=8, dim=384, num_attn_heads=6, depth=12)
        if "small16" in model or "s16" in model:
            return dict(patch_size=16, dim=384, num_attn_heads=6, depth=12)
        if "base8" in model or "b8" in model:
            return dict(patch_size=8, dim=768, num_attn_heads=12, depth=12)
        if "base16" in model or "b16" in model:
            return dict(patch_size=16, dim=768, num_attn_heads=12, depth=12)
        if "large16" in model or "l16" in model:
            return dict(patch_size=16, dim=1024, num_attn_heads=16, depth=24)
        if "huge16" in model or "h16" in model:
            return dict(patch_size=16, dim=1280, num_attn_heads=16, depth=32)
        if "huge14" in model or "h14" in model:
            return dict(patch_size=14, dim=1280, num_attn_heads=16, depth=32)
        if "twob14" in model or "2b14" in model:
            return dict(patch_size=14, dim=2560, num_attn_heads=32, depth=24)
        raise NotImplementedError(f"get_model_kwargs of '{self.model}' is not implemented")
