from aug_nomask import *
from aug_elder import *

def main(suffix, data_dir):
    AugNoMask(suffix, data_dir, brightness=64, contrast=64).aug_data()
    AugElder(data_dir+suffix).aug_elder_data()
    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-n', '--suffix', default='_aug', type=str,
                      help='add folder name to aug ver data folder')
    args.add_argument('-d', '--data_dir', default="./data/train/images", type=str,
                      help='data folder path')
    
    args = args.parse_args()
    main(args.suffix, args.data_dir)