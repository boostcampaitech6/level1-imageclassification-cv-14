import os
import shutil
from tqdm import tqdm

def copy_files(src_dir, dst_dir):

    copied_num = 0;
    for subdir in tqdm(os.listdir(src_dir)):
        src_subdir_path = os.path.join(src_dir, subdir)
        dst_subdir_path = os.path.join(dst_dir, subdir)

        if os.path.isdir(src_subdir_path):
            if not os.path.exists(dst_subdir_path):
                os.makedirs(dst_subdir_path)

            for file_path in os.listdir(src_subdir_path):
                src_file_path = os.path.join(src_subdir_path, file_path)
                dst_file_path = os.path.join(dst_subdir_path, file_path)

                if not file_path.startswith('.'):
                    shutil.copy2(src_file_path, dst_file_path)
                    copied_num += 1

    print(f'{copied_num} file copy has been completed.')

if __name__ == '__main__':
    src_dirs = []
    dst_dir = './data/train/aug/aug1'

    assert src_dirs != []
    
    for src_dir in src_dirs:
        copy_files(src_dir, dst_dir)
