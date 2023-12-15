from base import BaseDataset
from PIL import Image
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
)

class TrainDataset(BaseDataset):
    def __init__(self, data_dir, val_ratio=0.2):
        super().__init__(data_dir, val_ratio)

        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(224, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __getitem__(self, index):
        image = self.load_image(index)
        multi_class_label = self.load_multi_class_label(index)

        if self.transform is not None:
            image = self.transform(image)

        return image, multi_class_label