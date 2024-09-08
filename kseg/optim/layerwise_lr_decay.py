from ksuit.optim.param_group_modifiers import ParamGroupModifierBase


class LayerwiseLrDecay(ParamGroupModifierBase):
    def __init__(self, layerwise_lr_decay):
        self.layerwise_lr_decay = layerwise_lr_decay

    def get_properties(self, model, name, param):
        # adapted from BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        # this will split the model into len(blocks) + 2 "layers"
        # stem (patch_embed, cls_token, pos_embed) -> blocks -> last norm
        # this means that the last block will already be decayed
        if hasattr(model, "blocks"):
            num_layers = len(model.blocks) + 1
        elif hasattr(model, "model"):
            # e.g. torch_hub_model
            assert hasattr(model.model, "blocks")
            num_layers = len(model.model.blocks) + 1
            if name.startswith("model."):
                name = name[len("model."):]
        else:
            raise NotImplementedError
        scales = list(self.layerwise_lr_decay ** (num_layers - i) for i in range(num_layers))

        if (
                name.startswith("cls_token")
                or name.startswith("pos_embed")
                or name == "mask_token"
        ):
            return dict(lr_scale=scales[0])
        if name.startswith("patch_embed") or name.startswith("embed_norm"):
            return dict(lr_scale=scales[0])
        elif name.startswith("block"):
            layer = int(name.split('.')[1]) + 1
            return dict(lr_scale=scales[layer])
        elif name.startswith("norm.") or name.startswith("head."):
            # last norm is not scaled (i.e. original learning rate)
            return {}
        else:
            raise NotImplementedError(name)

    def __str__(self):
        return (
            f"{type(self).__name__}("
            f"layerwise_lr_decay={self.layerwise_lr_decay},"
            f")"
        )
