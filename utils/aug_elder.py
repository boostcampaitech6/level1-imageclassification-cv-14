import os
import cv2
import random
import argparse
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class AugElder():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.n_cpu = multiprocessing.cpu_count()

        # params
        self.brightness = 64
        self.contrast = 64

        # new id
        self.new_id = 7000

        # dir path init
        self.src_dirs = []
        self.dest_dirs = []

        self.src_files_paths = []
        self.dest_files_paths = []

        # setup
        self.setup()

    def makefolder(self, new_dir_path):
        os.makedirs(new_dir_path, exist_ok=False)

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
        for fun in funcs :
            img = getattr(self,fun)(img)
        cv2.imwrite(dest_img_path, img)

    def multiple_process(self,src_list, dest_list, n_funcs):
        with ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
            list(tqdm(executor.map(self.single_process, src_list, dest_list, n_funcs), total=len(src_list)))

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
            id, gender, race, age = profile.split("_")

            if int(age)<60: continue

            self.handle_elder_profiles(src_profile_dir, gender, race, age)

        for i in range(len(self.src_dirs)):
            for file_name in os.listdir(self.src_dirs[i]):
                self.categorize_file(self.src_dirs[i], self.dest_dirs[i], file_name)

    def handle_elder_profiles(self, src_dir, gender, race, age):
        self.src_dirs.append(src_dir)
        duplicated = []

        for _ in range(5):
            elder_id = str(self.new_id).zfill(6)
            self.new_id += 1

            dest_elder_profile = '_'.join([elder_id, gender, race, age])
            dest_elder_dir = os.path.join(self.data_dir, dest_elder_profile)
            self.makefolder(dest_elder_dir)
            duplicated.append(dest_elder_dir)
        self.dest_dirs.append(duplicated)
    
    def categorize_file(self, src_dir, dest_dirs, file_name):
        _file_name, ext = os.path.splitext(file_name)
        src_file_path = os.path.join(src_dir, file_name)
        
        dest_files_paths = []
        for dest_dir in dest_dirs:
            dest_files_paths.append(os.path.join(dest_dir,_file_name))
        
        self.src_files_paths.append(src_file_path)
        self.dest_files_paths.append(dest_files_paths)
    
    def aug_elder_data(self):
        process_types = [['blur'], ['flip'], ['jitter'], ['blur', 'jitter'], ['flip', 'jitter']]
        b_list = [b for b in range(0,65,13)]
        c_list = [c for c in range(0,65,13)]
        b_c_comb = [(b,c) for b in b_list for c in c_list]
        b_c_comb = random.sample(b_c_comb, 5)

        tasks = [[],[],[]]
        for i in range(len(self.src_files_paths)):
            src_file_path = self.src_files_paths[i]
            dest_files_path = self.dest_files_paths[i]
            
            for j in range(5):
                funcs = process_types[j]
                b, c = b_c_comb[j]
                dest_file_path = dest_files_path[j]
                self.brightness = b
                self.contrast = c
                suffix = '_' + '_'.join(funcs) + f'_b{b}_c{c}.jpg'
                dest_file_path += suffix
                tasks[0].append(src_file_path)
                tasks[1].append(dest_file_path)
                tasks[2].append(funcs)

        self.multiple_process(tasks[0], tasks[1], tasks[2])

def main(data_dir):
    aug_data = AugElder(data_dir)
    aug_data.aug_elder_data()
    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-d', '--data_dir', default=None, type=str,
                      help='data folder path (default: ./data/train)')
    
    args = args.parse_args()
    main(args.data_dir)