import einops
import torch
import torch.nn.functional as F
from torch import nn
from ksuit.models import SingleModel


class Dinov2Segmenter(SingleModel):
    def __init__(self, model="dinov2_vits14", **kwargs):
        super().__init__(**kwargs)
        self.backbone = torch.hub.load("facebookresearch/dinov2", model)
        head = torch.hub.load_state_dict_from_url(
            f"https://dl.fbaipublicfiles.com/dinov2/{model}/{model}_ade20k_linear_head.pth",
            map_location="cpu",
        )
        num_classes, dim, _, _ = head["state_dict"]["decode_head.conv_seg.weight"].shape
        self.head = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, num_classes, kernel_size=1),
        )
        sd = head["state_dict"]
        sd = {key.replace("decode_head.bn.", "0."): value for key, value in sd.items()}
        sd = {key.replace("decode_head.conv_seg.", "1."): value for key, value in sd.items()}
        self.head.load_state_dict(sd)

    def forward(self, x):
        _, _, height, width = x.shape
        assert height % self.backbone.patch_size == 0
        assert width % self.backbone.patch_size == 0
        seqlen_h = height // self.backbone.patch_size
        seqlen_w = width // self.backbone.patch_size

        x = self.backbone.forward_features(x)["x_norm_patchtokens"]
        x = einops.rearrange(
            x,
            "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
            seqlen_h=seqlen_h,
            seqlen_w=seqlen_w,
        )
        x = self.head(x)
        x = F.interpolate(x, size=(height, width), mode="nearest")
        return x

    def segment(self, x):
        return self(x)