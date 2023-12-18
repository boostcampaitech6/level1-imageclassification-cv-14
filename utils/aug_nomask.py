import os
import cv2
import argparse
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class AugNoMask():
    def __init__(self, suffix, src_dir, brightness, contrast):
        self.src_dir = src_dir
        self.dest_dir = src_dir + suffix
        self.n_cpu = multiprocessing.cpu_count()

        # params
        self.brightness = brightness
        self.contrast = contrast

        # categorical path init
        self.src_mask_pathes = []
        self.src_normal_pathes = []
        self.src_incorrect_pathes = []
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

    def multiple_process(self, src_list, dest_list, n_funcs):
        with ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
            list(tqdm(executor.map(self.single_process, src_list, dest_list, n_funcs), total=len(src_list)))

    def aug_data(self):
        process_types = [[''], ['blur'], ['flip'], ['jitter'], ['blur', 'flip']]

        for src_paths, dest_paths, category in ([(self.src_mask_pathes, self.dest_mask_pathes, 'mask'),
                                                 (self.src_normal_pathes, self.dest_normal_pathes, 'normal'),
                                                 (self.src_incorrect_pathes, self.dest_incorrect_pathes, 'incorrect')]):
            process = [['']] if category == 'mask' else process_types
            for funcs in process:
                suffix = ''.join(funcs)
                suffix = '_' + suffix if suffix else suffix
                mod_dest_paths = [p + f'{suffix}.jpg' for p in dest_paths]
                self.multiple_process(src_paths, mod_dest_paths, [funcs for _ in range(len(mod_dest_paths))])

    def setup(self):
        profiles = [p for p in os.listdir(self.src_dir) if not p.startswith('.')]

        for profile in profiles:
            src_profile_dir = os.path.join(self.src_dir, profile)
            dest_profile_dir = os.path.join(self.dest_dir, profile)
            self.makefolder(dest_profile_dir)

            for file_name in os.listdir(src_profile_dir):
                self.categorize_file(src_profile_dir, dest_profile_dir, file_name)
    
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
            self.dest_incorrect_pathes.append(dest_path)

def main(suffix, src_dir):
    aug_data = AugNoMask(suffix, src_dir, brightness=64, contrast=64)
    aug_data.aug_data()
    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-n', '--suffix', default='_aug', type=str,
                      help='add folder name to aug ver data folder')
    args.add_argument('-d', '--src_dir', default="./data/train/images", type=str,
                      help='data folder path (default: ./data/train)')
    
    args = args.parse_args()
    main(args.suffix, args.src_dir)