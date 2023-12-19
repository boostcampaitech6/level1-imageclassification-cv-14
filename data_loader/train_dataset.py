import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from base import BaseDataset

class TrainDataset(BaseDataset):
    def __init__(self, data_dir, calc_mean, calc_std, use_all_data):
        super().__init__(data_dir, calc_mean=calc_mean, calc_std=calc_std, use_all_data=use_all_data)

        self.transform = A.Compose([
            A.CenterCrop(400, 300),  
            A.Resize(224, 224), 
            A.Normalize(self.mean, self.std),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        image_path = self.load_image_path(index)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        multi_class_label = self.load_multi_class_label(index)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, multi_class_label