from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.5607, 0.5253, 0.5027), std=(0.6174, 0.5885, 0.5695)):
        super().__init__()

        self.img_paths = img_paths
        self.mean = mean
        self.std = std

        self.transform = A.Compose([
            A.CenterCrop(400, 300),  
            A.Resize(resize, resize), 
            A.Normalize(self.mean, self.std),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(str(self.img_paths[index])), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        return image

    def __len__(self):
        return len(self.img_paths)