from base import BaseDataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrainDataset(BaseDataset):
    def __init__(self, data_dir, val_ratio=0.2):
        super().__init__(data_dir, val_ratio)

        self.transform = A.Compose([   
            A.Normalize(self.mean, self.std),
            A.CenterCrop(400, 300),
            A.Resize(224, 224),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        image_path = self.load_image_path(index)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        multi_class_label = self.load_multi_class_label(index)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, multi_class_label