import os
from argparse import ArgumentParser
from pathlib import Path

from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    return vars(parser.parse_args())


def main(root):
    root = Path(root).expanduser()
    assert root.exists() and root.is_dir(), root.as_posix()
    fnames = list(sorted(os.listdir(root)))
    min_height = float("inf")
    min_width = float("inf")
    max_height = -float("inf")
    max_width = -float("inf")
    for fname in tqdm(fnames):
        img = to_tensor(default_loader(root / f"{fname}"))
        _, height, width = img.shape
        min_height = min(min_height, height)
        min_width = min(min_width, width)
        max_height = max(max_height, height)
        max_width = max(max_width, width)
    print(f"min_height {min_height}")
    print(f"min_width {min_width}")
    print(f"max_height {max_height}")
    print(f"max_width {max_width}")


if __name__ == "__main__":
    main(**parse_args())
