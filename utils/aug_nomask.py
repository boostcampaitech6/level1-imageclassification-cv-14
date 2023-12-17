import os
import cv2
import random
import argparse
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class DataAugment():
    def __init__(self, suffix, data_dir):
        self.data_dir = data_dir
        self.dest_dir = data_dir+suffix
        self.n_cpu = multiprocessing.cpu_count()

        # params
        self.brightness = 64
        self.contrast = 64

        # new id
        self.new_id = 7000

        # categorical path init
        self.src_elder_profiles_dir = []
        self.src_mask_pathes = []
        self.src_normal_pathes = []
        self.src_incorrect_pathes = []
        self.dest_elder_profiles_dir = []
        self.dest_mask_pathes = []
        self.dest_normal_pathes = []
        self.dest_incorrect_pathes = []

        # setup
        self.setup()

    def makefolder(self, new_dir_path):
        os.makedirs(new_dir_path, exist_ok=True)

    def get_img(self, img_path):
        return cv2.imread(img_path)

    def blur(self, img):
        return cv2.GaussianBlur(img, (0,0), sigmaX=3)

    def flip(self, img):
        return cv2.flip(img, 1)

    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv 
    def jitter(self, img):
        brightness, contrast = self.brightness, self.contrast
        if brightness != 0:
            shadow = brightness if brightness > 0 else 0
            highlight = 255 if brightness > 0 else 0
            img = cv2.convertScaleAbs(img, alpha=(highlight - shadow)/255, beta=shadow)

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            img = cv2.convertScaleAbs(img, alpha=f, beta=127*(1-f))

        return img
    
    def single_process(self, src_img_path, dest_img_path, funcs):
        img = self.get_img(src_img_path)
        if funcs != [''] :
            for fun in funcs :
                img = getattr(self,fun)(img)
        cv2.imwrite(dest_img_path, img)

    def multiple_process(self,src_list, dest_list, n_funcs):
        with ProcessPoolExecutor(max_workers=self.n_cpu) as excutor:
            excutor.map(self.single_process, src_list, dest_list, n_funcs)

    def aug_data(self):
        process_types = [[''], ['blur'], ['flip'], ['jitter'], ['blur', 'flip']]

        for src_paths, dest_paths, category in [(self.src_mask_pathes, self.dest_mask_pathes, 'mask'),
                                                (self.src_normal_pathes, self.dest_normal_pathes, 'normal'),
                                                (self.src_incorrect_pathes, self.dest_incorrect_pathes, 'incorrect')]:
            for funcs in process_types:
                suffix = ''.join(funcs)
                mod_dest_paths = [p+f'_{suffix}.jpg' for p in dest_paths]
                self.multiple_process(src_paths, mod_dest_paths, funcs*len(src_paths))

    def setup(self):
        profiles = [p for p in os.listdir(self.data_dir) if not p.startswith('.')]

        for profile in profiles:
            src_profile_dir = os.path.join(self.data_dir, profile)
            dest_profile_dir = os.path.join(self.dest_dir, profile)
            self.makefolder(dest_profile_dir)

            id, gender, race, age = profile.split("_")
            if int(age)>=60:
                self.handle_elder_profiles(src_profile_dir, gender, race, age)

            for file_name in os.listdir(src_profile_dir):
                self.categorize_file(src_profile_dir, dest_profile_dir, file_name)
    
    def handle_elder_profiles(self, src_dir, gender, race, age):
        self.src_elder_profiles_dir.append(src_dir)
        for _ in range(5):
            elder_id = str(self.new_id).zfill(6)
            self.new_id += 1

            dest_elder_profile = '_'.join([elder_id, gender, race, age])
            dest_elder_dir = os.path.join(self.dest_dir, dest_elder_profile)
            self.dest_elder_profiles_dir.append(dest_elder_dir)
            self.makefolder(dest_elder_dir)
    
    def categorize_file(self, src_dir, dest_dir, file_name):
        _file_name, ext = os.path.splitext(file_name)
        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, _file_name)

        if _file_name.startswith('mask'):
            self.src_mask_pathes.append(src_path)
            self.dest_mask_pathes.append(dest_path)
        elif _file_name.startswith('normal'):
            self.src_normal_pathes.append(src_path)
            self.dest_normal_pathes.append(dest_path)
        elif _file_name.startswith('incorrect'):
            self.src_incorrect_pathes.append(src_path)
            self.src_incorrect_pathes.append(dest_path)
    
    def aug_elder_data(self):
        process_types = [['blur'], ['flip'], ['jitter'], ['blur', 'jitter'], ['flip', 'jitter']]
        b_list = [b for b in range(0,65,13)]
        c_list = [c for c in range(0,65,13)]
        b_c_comb = [(b,c) for b in b_list for c in c_list]
        b_c_comb = random.sample(b_c_comb, 5)

        tasks = []
        idx = 0
        for elder_dir in self.src_elder_profiles_dir:
            for file_name in os.listdir(elder_dir):
                src_path = os.path.join(elder_dir, file_name)
                base_name, ext = os.path.splitext(file_name)

                for funcs in process_types:
                    for brightness, contrast in b_c_comb:
                        self.brightness = brightness
                        self.contrast = contrast
                        suffix = '_' + '_'.join(funcs) + f'_b{brightness}_c{contrast}'
                        dest_path = self.dest_elder_profiles_dir[idx]
                        dest_path = os.path.join(dest_path, base_name + suffix + ext)
                        tasks.append((src_path, dest_path, funcs))
                        idx += 1

        with ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
            executor.map(lambda p: self.single_process(*p), tasks)


def main(suffix, data_dir):
    aug_data = DataAugment(suffix, data_dir)
    aug_data.aug_data()
    aug_data.aug_elder_data()
    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-n', '--suffix', default='aug', type=str,
                      help='add folder name to aug ver data folder')
    args.add_argument('-d', '--data_dir', default="/data/ephemeral/home/level1-imageclassification-cv-14/data/train/debug_images", type=str,
                      help='data folder path (default: ./data/train)')
    
    args = args.parse_args()
    main(args.suffix, args.data_dir)