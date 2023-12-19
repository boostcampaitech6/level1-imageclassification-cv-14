import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import MaskLabels, GenderLabels, AgeLabels, encode_multi_class
import random
from tqdm import tqdm


class BaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), 
                 calc_mean=False, calc_std=False, use_all_data=False):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.calc_mean = calc_mean
        self.calc_std = calc_std
        self.use_all_data = use_all_data
        
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []

        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                if '_' in file_name:
                    _file_name = file_name.split('_')[0]
                else:
                    _file_name, _ = os.path.splitext(file_name)
                    
                if _file_name not in self._file_names:
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def load_image_path(self, index):
        image_path = self.image_paths[index]
        return image_path
    
    def load_multi_class_label(self, index):
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)

        multi_class_label = encode_multi_class(mask_label, gender_label, age_label)
        return multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        do_calc = self.calc_mean or self.calc_std
        
        if not do_calc:
            assert has_statistics

        if do_calc:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            
            if self.use_all_data:
                image_paths = self.image_paths
            else:
                image_paths = random.sample(self.image_paths, 5000)
            
            sums = []
            squared = []
            for image_path in tqdm(image_paths):
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            if self.calc_mean or self.mean is None:
                self.mean = np.mean(sums, axis=0) / 255
            if self.calc_std or self.std is None:
                self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

            print("mean:", self.mean)
            print("std", self.std)
