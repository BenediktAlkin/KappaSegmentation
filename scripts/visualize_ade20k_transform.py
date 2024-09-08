from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from kseg.datasets.ade20k import Ade20k
from kseg.sample_wrappers.segmentation_transform_wrapper import SegmentationTransformWrapper
from kseg.transforms.segmentation_pad import SegmentationPad
from kseg.transforms.segmentation_random_crop import SegmentationRandomCrop
from kseg.transforms.segmentation_random_horizontal_flip import SegmentationRandomHorizontalFlip
from kseg.transforms.segmentation_random_resize import SegmentationRandomResize
from ksuit.utils.tensor_hashing import hash_rgb


def main():
    torch.manual_seed(0)
    root = Path("~/Documents/data/ade20k").expanduser()
    assert root.exists()
    out = Path("temp")
    out.mkdir(exist_ok=True)

    root_dataset = Ade20k(global_root=root, split="training")
    dataset = SegmentationTransformWrapper(
        dataset=root_dataset,
        transform=[
            SegmentationRandomResize(ratio_resolution=[2048, 512], ratio_range=[0.5, 2.0], interpolation="bicubic"),
            SegmentationRandomCrop(size=512, max_category_ratio=0.75, ignore_index=-1),
            SegmentationRandomHorizontalFlip(),
            SegmentationPad(size=512, fill=-1),
        ]
    )
    for i in tqdm(range(10)):
        ctx = {}
        og_x = root_dataset.getitem_x(i)
        og_y = hash_rgb(root_dataset.getitem_segmentation(i), dim=0)
        x = dataset.getitem_x(i, ctx=ctx)
        seg = dataset.getitem_segmentation(i, ctx=ctx)
        seg = hash_rgb(seg, dim=0)
        og_x.save(out / f"{i:03d}_ogx.png")
        save_image(og_y, out / f"{i:03d}_ogy.png")
        x.save(out / f"{i:03d}_tx.png")
        save_image(seg, out / f"{i:03d}_ty.png")


if __name__ == "__main__":
    main()
