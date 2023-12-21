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
SEED = 42

def makefolder(new_dir_path):
    os.makedirs(new_dir_path, exist_ok=True)

def multiple_process(src_file_paths, dest_file_paths, n_cpu):
    assert len(src_file_paths) != 0, 'no file in src file'
    with ProcessPoolExecutor(max_workers=n_cpu-1) as executor:
        list(tqdm(executor.map(single_process, src_file_paths, dest_file_paths), total=len(src_file_paths)))

def single_process(src_file_path, dest_file_path):
    img = get_img(src_file_path)
    img = rmback(img)
    cv2.imwrite(dest_file_path, img)

def get_img(file_path):
    return cv2.imread(file_path)

def rmback(img):
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


def main(src_dir, dest_dir):
    n_cpu = multiprocessing.cpu_count()
    makefolder(dest_dir)
    file_paths = os.listdir(src_dir)
    src_file_paths = []
    dest_file_paths = []
    
    for file in file_paths :
        src_file_paths.append(os.path.join(src_dir, file))
        dest_file_paths.append(os.path.join(dest_dir, file))
    
    multiple_process(src_file_paths, dest_file_paths, n_cpu)



    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-s', '--src_dir', default='./data/eval/images', type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--dest_dir', default='./data/rmback_test', type=str,
                      help='config file path (default: None)')
    args = args.parse_args()

    main(args.src_dir, args.dest_dir)