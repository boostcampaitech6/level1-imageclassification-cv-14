import os
import cv2
import json
import random
import argparse
import multiprocessing
import numpy as np

from rembg import remove
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class AugElder():
    def __init__(self, config):
        self.src_dir = config["src_dir"]
        self.dest_dir = config["dest_dir"]
        self.n_cpu = multiprocessing.cpu_count()

        # params
        self.brightness = 0
        self.contrast = 0

        # files paths init
        self.src_files_paths = []
        self.dest_files_paths = []

        # cate info
        self.cate = config["class"]

        # process tyoes
        self.transforms = config["transforms"]

        self.n_repeat = len(config["transforms"])

        # setup
        self.setup()

    def setup(self):
        profiles = [p for p in os.listdir(self.src_dir) if not p.startswith('.')]

        for profile in profiles:
            _, _, _, age = profile.split("_")

            # unchosen cate
            if ((not self.cate["young"]) and int(age) < 30) or\
               ((not self.cate["midle"]) and (int(age) >= 30) and (int(age) < 60)) or\
               ((not self.cate["old"]) and int(age) >= 60):
                continue
            
            src_profile_dir = os.path.join(self.src_dir, profile)
            dest_profile_dir = os.path.join(self.dest_dir, profile)
            self.makefolder(dest_profile_dir)

            src_file_paths = []
            dest_file_paths = []

            for file_name in os.listdir(src_profile_dir):
                _file_name, _ = os.path.splitext(file_name)

                if not file_name.startswith('.'):
                    src_file_paths.append(os.path.join(src_profile_dir, file_name))
                    dest_file_paths.append(os.path.join(dest_profile_dir, _file_name))
                
            self.src_files_paths.append(src_file_paths)
            self.dest_files_paths.append(dest_file_paths)

    def makefolder(self, new_dir_path):
        os.makedirs(new_dir_path, exist_ok = False)
    
    def aug_data(self):
        b_c_comb = [(b,c) for b in list(range(0, 65, 13)) for c in list(range(0, 65, 13))]
        b_c_comb = random.sample(b_c_comb, self.n_repeat)

        for idx, funcs in enumerate(self.transforms):
            b, c = b_c_comb[idx]
            self.brightness, self.contrast = b, c
            suffix = '_'.join(funcs) + f'_b{b}_c{c}' if 'jitter' in funcs else '_'.join(funcs)

            src_files_paths = []
            mod_dest_files_paths = []

            # searching inside profile
            for src_file_paths, dest_file_paths in zip(self.src_files_paths, self.dest_files_paths):
                src_img_paths = [p for p in src_file_paths]               
                mod_dest_img_paths = [p + f'_{suffix}.jpg' for p in dest_file_paths]
            
                src_files_paths.extend(src_img_paths)
                mod_dest_files_paths.extend(mod_dest_img_paths)
            
            self.multiple_process(src_files_paths, mod_dest_files_paths, [funcs]*len(src_files_paths))

    def multiple_process(self, src_file_paths, dest_file_paths, n_funcs):
        with ProcessPoolExecutor(max_workers = self.n_cpu) as executor:
            list(tqdm(executor.map(self.single_process, src_file_paths, dest_file_paths, n_funcs), total = len(src_file_paths)))

    def single_process(self, src_img_path, dest_img_path, funcs):
        img = self.get_img(src_img_path)
        for func in funcs :
            img = getattr(self,func)(img)
        cv2.imwrite(dest_img_path, img)

    def get_img(self, img_path):
        return cv2.imread(img_path)

    def blur(self, img):
        return cv2.GaussianBlur(img, (0,0), sigmaX = 3)

    def flip(self, img):
        return cv2.flip(img, 1)

    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv 
    def jitter(self, img):
        brightness, contrast = self.brightness, self.contrast
        if brightness != 0:
            shadow = brightness if brightness > 0 else 0
            highlight = 255 if brightness > 0 else 0
            img = cv2.convertScaleAbs(img, alpha = (highlight - shadow)/255, beta = shadow)

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            img = cv2.convertScaleAbs(img, alpha = f, beta = 127*(1-f))

        return img

    def clahe(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # CLAHE 생성
        img[:, :, 0] = clahe.apply(img[:, :, 0])        # 밝기 채널에 CLAHE 적용
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        return img

    def remove_back(img):
        # rembg
        img = remove(img)
        img = np.array(img)

        # white background
        white_bg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

        # extract alpa
        if img.shape[2] == 4:
            alpha_channel = img[:, :, 3]
            img = img[:, :, :3]

            # Overlay the original image on a white background using an alpha channel
            for c in range(0, 3):
                white_bg[:, :, c] = white_bg[:, :, c] * (1 - alpha_channel / 255.0) + img[:, :, c] * (alpha_channel / 255.0)

        return white_bg

def main(config):
    random.seed(42)
    aug_mask = AugElder(config['age'])
    aug_mask.aug_data()
    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-c', '--config', default='/opt/workspace/week6/level1-imageclassification-cv-14/data_aug_utils/aug_config.json', type=str,
                      help='config file path (default: None)')

    args = args.parse_args()
    
    with open(args.config, 'r') as f :
        config = json.load(f)
    main(config)