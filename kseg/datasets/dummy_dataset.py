import torch

from ksuit.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, resolution=64, size=32, **kwargs):
        super().__init__(**kwargs)
        self.resolution = resolution
        self.size = size


    def getitem_x(self, idx):
        return torch.rand(3, self.resolution, self.resolution, generator=torch.manual_seed(idx % 2))

    def getitem_segmentation(self, idx):
        return torch.randint(size=(self.resolution, self.resolution), low=-1, high=150, generator=torch.manual_seed(idx % 2))

    @staticmethod
    def getshape_segmentation():
        return 150,

    def __len__(self):
        return self.size
