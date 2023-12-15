from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
)

class TestDataset(Dataset):
    def __init__(
        self, img_paths, resize=224, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)