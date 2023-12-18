from aug_nomask import *
from aug_elder import *

def main(src_dir, dest_dir):
    aug_nomask_data = AugNoMask(src_dir, dest_dir, brightness=64, contrast=64)
    aug_elder_data = AugElder(dest_dir)

    aug_nomask_data.aug_data()
    aug_elder_data.aug_data()
    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-s', '--src_dir', default='/data/ephemeral/home/level1-imageclassification-cv-14/data/train/debug_images', type=str,
                      help='src data folder path (default: ./data/train/images)')
    args.add_argument('-d', '--dest_dir', default='/data/ephemeral/home/level1-imageclassification-cv-14/data/train/debug_images_aug', type=str,
                      help='src data folder path (default: ./data/train/images_aug)')

    args = args.parse_args()
    main(args.src_dir, args.dest_dir)