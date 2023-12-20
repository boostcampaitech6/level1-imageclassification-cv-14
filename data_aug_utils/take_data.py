import os
import argparse
import shutil
import multiprocessing
import pandas as pd

def makefolder(new_dir_path):
    os.makedirs(new_dir_path, exist_ok=True)

def main(src_dir, valid_dir, train_dir, csv):
    val_csv = pd.read_csv(csv)
    val_list = []
    for idx, row in val_csv.iterrows() :
        val_list.append(int(row.item()))
    val_list = sorted(list(map(int, val_list)))

    profiles = [p for p in os.listdir(src_dir) if (not p.startswith('.'))]
    profiles.sort()

    point_train = 0
    point_val_list = 0
    # print(val_list)
    for profile in profiles :
        src_profile_path = os.path.join(src_dir, profile)
        valid_profle_path = os.path.join(valid_dir, profile)
        train_profile_path = os.path.join(train_dir,profile)
        file_names = [p for p in os.listdir(src_profile_path) if (not p.startswith('.'))]

        for idx, file_name in enumerate(file_names) :
            src_file_path = os.path.join(src_profile_path, file_name)
            valid_file_path = os.path.join(valid_profle_path, file_name)
            train_file_path = os.path.join(train_profile_path, file_name)

            if point_val_list < len(val_list) and (point_train == int(val_list[point_val_list])) :
                makefolder(valid_profle_path)
                shutil.copy(src_file_path, valid_file_path)
                point_val_list += 1
            else :
                makefolder(train_profile_path)
                shutil.copy(src_file_path, train_file_path)
            point_train += 1

    print('-----------------')
    print('completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-s', '--src_dir', default='./data/train/images', type=str,
                      help='src dir')
    args.add_argument('-v', '--valid_dir', default='./data/split_valid/images', type=str,
                      help='valid dir path')
    args.add_argument('-t', '--train_dir', default='./data/split_train/images', type=str,
                      help='train dir path')
    args.add_argument('-c', '--csv', default='./data/valid_set_seed_951.csv', type=str,
                      help='csv dir path')
    args = args.parse_args()

    main(args.src_dir, args.valid_dir, args.train_dir, args.csv)