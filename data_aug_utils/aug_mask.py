import os
import cv2
import argparse
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class AugNoMask():
    def __init__(self, src_dir, dest_dir, brightness, contrast):
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.n_cpu = multiprocessing.cpu_count()

        # params
        self.brightness = brightness
        self.contrast = contrast

        # categorical path init
        self.src_mask_paths = []
        self.src_normal_paths = []
        self.src_incorrect_paths = []
        self.dest_mask_paths = []
        self.dest_normal_paths = []
        self.dest_incorrect_paths = []

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
                src_img_path = os.path.join(src_profile_dir, file_name)
                dest_img_path = os.path.join(dest_profile_dir, _file_name)

                if _file_name.startswith('mask'):
                    self.src_mask_paths.append(src_img_path)
                    self.dest_mask_paths.append(dest_img_path)
                elif _file_name.startswith('normal'):
                    self.src_normal_paths.append(src_img_path)
                    self.dest_normal_paths.append(dest_img_path)
                elif _file_name.startswith('incorrect'):
                    self.src_incorrect_paths.append(src_img_path)
                    self.dest_incorrect_paths.append(dest_img_path)

    def makefolder(self, new_dir_path):
        os.makedirs(new_dir_path, exist_ok=True)

    
    def single_process(self, src_img_path, dest_img_path, funcs):
        img = self.get_img(src_img_path)
        if funcs != [''] :
            for func in funcs :
                img = getattr(self,func)(img)
        cv2.imwrite(dest_img_path, img)

    def multiple_process(self, src_img_paths, dest_img_paths, n_funcs):
        with ProcessPoolExecutor(max_workers=self.n_cpu) as executor:
            list(tqdm(executor.map(self.single_process, src_img_paths, dest_img_paths, n_funcs), total=len(src_img_paths)))

    def aug_data(self):
        process_types = [[''], ['blur'], ['flip'], ['jitter'], ['blur', 'flip']]

        for src_img_paths, dest_img_paths, category in ([(self.src_mask_paths, self.dest_mask_paths, 'mask'),
                                                 (self.src_normal_paths, self.dest_normal_paths, 'normal'),
                                                 (self.src_incorrect_paths, self.dest_incorrect_paths, 'incorrect')]):
            process = [['']] if category == 'mask' else process_types
            for funcs in process:
                suffix = '_'.join(funcs)
                mod_dest_img_paths = [p + f'_{suffix}.jpg' for p in dest_img_paths]
                self.multiple_process(src_img_paths, mod_dest_img_paths, [funcs for _ in range(len(src_img_paths))])

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

def main(src_dir, dest_dir):
    aug_data = AugNoMask(src_dir, dest_dir, brightness=64, contrast=64)
    aug_data.aug_data()
    print('Data augmentation completed.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='data preprocessing')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--src_dir', default='./data/train/images', type=str,
                      help='src data folder path (default: ./data/train/images)')
    args.add_argument('-d', '--dest_dir', default='./data/train/images_aug', type=str,
                      help='add folder name to aug ver data folder')

    args = args.parse_args()
    main(args.src_dir, args.dest_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['-e', '--exp_name'], type=str, target='wandb;exp_name'),
        CustomArgs(['-n', '--exp_num'], type=int, target='wandb;exp_num'),
        CustomArgs(['--project_name'], type=str, target='wandb;project_name'),
        CustomArgs(['--entity'], type=str, target='wandb;entity')
    ]
    config = TrainConfigParser.from_args(args, options)
    main(config)
