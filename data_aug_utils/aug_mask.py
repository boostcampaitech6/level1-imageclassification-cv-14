import os
import cv2
import json
import random
import argparse
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class AugNoMask():
    def __init__(self, config):
        self.src_dir = config["src_dir"]
        self.dest_dir = config["dest_dir"]
        self.n_cpu = multiprocessing.cpu_count()

        # params
        self.brightness = 0
        self.contrast = 0

        # cate info
        self.cate = config["class"]

        # categorical paths init
        self.src_incorrect_paths = []
        self.dest_incorrect_paths = []
        self.src_mask_paths = []
        self.dest_mask_paths = []
        self.src_normal_paths = []
        self.dest_normal_paths = []
        
        # process types
        self.transforms = config["transforms"]

        # setup
        self.setup()

    def setup(self):
        profiles = [p for p in os.listdir(self.src_dir) if not p.startswith('.')]

        for profile in profiles:
            src_profile_dir = os.path.join(self.src_dir, profile)
            dest_profile_dir = os.path.join(self.dest_dir, profile)
            self.makefolder(dest_profile_dir)

            for file_name in os.listdir(src_profile_dir):
                _file_name, ext = os.path.splitext(file_name)
                src_file_path = os.path.join(src_profile_dir, file_name)
                dest_file_path = os.path.join(dest_profile_dir, _file_name)

                if self.cate["mask"] and _file_name.startswith('mask'):
                    self.src_mask_paths.append(src_file_path)
                    self.dest_mask_paths.append(dest_file_path)
                if self.cate["normal"] and _file_name.startswith('normal'):
                    self.src_normal_paths.append(src_file_path)
                    self.dest_normal_paths.append(dest_file_path)
                if self.cate["incorrect"] and _file_name.startswith('incorrect'):
                    self.src_incorrect_paths.append(src_file_path)
                    self.dest_incorrect_paths.append(dest_file_path)

    def makefolder(self, new_dir_path):
        os.makedirs(new_dir_path, exist_ok=True)

    def aug_data(self):
        b_c_comb = [(b,c) for b in list(range(0, 65, 13)) for c in list(range(0, 65, 13))]
        b_c_comb = random.sample(b_c_comb, len(self.transforms))
        for src_file_paths, dest_file_paths, category in ([(self.src_mask_paths, self.dest_mask_paths, 'mask'),
                                                         (self.src_normal_paths, self.dest_normal_paths, 'normal'),
                                                         (self.src_incorrect_paths, self.dest_incorrect_paths, 'incorrect')]):
            for idx, funcs in enumerate(self.transforms):
                b, c = b_c_comb[idx]
                self.brightness, self.contrast = b, c
                suffix = '_'.join(funcs) + f'_b{b}_c{c}' if 'jitter' in funcs else '_'.join(funcs)
                mod_dest_file_paths = [p + f'_{suffix}.jpg' for p in dest_file_paths]
                self.multiple_process(src_file_paths, mod_dest_file_paths, [funcs] * len(src_file_paths))

    def multiple_process(self, src_file_paths, dest_file_paths, n_funcs):
        if len(src_file_paths) == 0 : return
        with ProcessPoolExecutor(max_workers=self.n_cpu-1) as executor:
            list(tqdm(executor.map(self.single_process, src_file_paths, dest_file_paths, n_funcs), total=len(src_file_paths)))
    
    def single_process(self, src_file_path, dest_file_path, funcs):
        img = self.get_img(src_file_path)
        for func in funcs :
            img = getattr(self,func)(img)
        cv2.imwrite(dest_file_path, img)

    def get_img(self, file_path):
        return cv2.imread(file_path)

    def blur(self, img):
        return cv2.GaussianBlur(img, (0,0), sigmaX=2)

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
    
    def clahe(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # CLAHE 생성
        img[:, :, 0] = clahe.apply(img[:, :, 0])        # 밝기 채널에 CLAHE 적용
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        return img


def main(config):
    random.seed(42)
    aug_mask = AugNoMask(config['mask'])
    aug_mask.aug_data()
    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-c', '--config', default='./data_aug_utils/aug_config.json', type=str,
                      help='config file path (default: None)')

    args = args.parse_args()
    
    with open(args.config, 'r') as f :
        config = json.load(f)
    main(config)
