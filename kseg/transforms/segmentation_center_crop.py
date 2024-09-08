from torchvision.transforms.functional import center_crop

from ksuit.data.transforms import StochasticTransform
from ksuit.utils.param_checking import to_2tuple

class SegmentationCenterCrop(StochasticTransform):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = to_2tuple(size)

    def __call__(self, xseg, ctx=None):
        x, seg = xseg
        x = center_crop(x, self.size)
        seg = center_crop(seg, self.size)
        return x, seg

    def get_params(self, height, width):
        top = int(self.rng.integers(max(0, height - self.size[0]) + 1, size=(1,)))
        left = int(self.rng.integers(max(0, width - self.size[1]) + 1, size=(1,)))
        crop_height = min(height, self.size[0])
        crop_width = min(width, self.size[1])
        return top, left, crop_height, crop_width
