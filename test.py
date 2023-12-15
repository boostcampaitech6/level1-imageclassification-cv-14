import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import model.model as module_arch
from parse_config import ConfigParser
from PIL import Image
from torchvision import transforms

def main(config):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # Load model
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Write model output to submission.csv 
        with open(config['input_file'], 'r') as input_file, open(config['output_file'],'w') as output_file:
            titles = input_file.readline()
            output_file.write(titles)

            for line in tqdm(input_file.readlines()):
                image = Image.open(os.path.join(config['test_dataset']['args']['img_paths'],line.split(',')[0]))
                transform = transforms.Compose([transforms.Resize(config['test_dataset']['args']['resize']),
                                                transforms.ToTensor(),
                                                transforms.CenterCrop(150),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
                image = transform(image)
                image = image.to(device)
                output = model(image.unsqueeze(0))
                    
                class_idx = np.argmax(output.cpu().numpy(), axis=1)          
                modified = line.strip()[:-1]+str(class_idx[0])
                output_file.write(modified+'\n')
                    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)