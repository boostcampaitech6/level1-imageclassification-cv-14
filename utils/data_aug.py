from aug_nomask import *
from aug_elder import *

def main(suffix, src_data):
    aug_nomask_data = AugNoMask(suffix, src_data, brightness=64, contrast=64)
    aug_elder_data = AugElder(src_data+suffix)

    aug_nomask_data.aug_data()
    aug_elder_data.aug_data()
    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-s', '--suffix', default='_aug', type=str,
                      help='add folder name to aug ver data folder')
    args.add_argument('-d', '--src_data', default="./data/train/images", type=str,
                      help='data folder path')
    
    args = args.parse_args()
    main(args.suffix, args.src_data)