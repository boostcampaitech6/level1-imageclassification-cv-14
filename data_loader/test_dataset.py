from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TestDataset(Dataset):
    def __init__(
        self, img_paths, resize=224, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.transform = A.Compose([
            A.Normalize(mean, std),
            A.Resize(resize, resize),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image

    def __len__(self):
        return len(self.img_paths)