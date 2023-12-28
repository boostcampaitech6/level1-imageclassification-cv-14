import numpy as np
import pandas as pd
import argparse
import collections
import torch
from tqdm import tqdm
from pathlib import Path
import model as module_arch
from data_loader import TestDataset
from parse_config import TestConfigParser
from utils import encode_multi_class

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = Path(config['single_model']['saved_dir']) / \
                    config['single_model']['saved_exp_name'] / \
                    str(config['single_model']['saved_exp_num']) / \
                    config['single_model']['saved_model']

    is_multi_task = config['single_model']['is_multi_task']
    
    info = pd.read_csv(config['info_path'])
    img_paths = [Path(config['image_path']) / img_id for img_id in info.ImageID]

    test_set = TestDataset(img_paths, config['resize'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    checkpoint = torch.load(model_path)

    model_config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    model = model_config.init_obj('arch', module_arch)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    output = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader)):
            data = data.to(device)
            _output = model(data).logits
            output.extend(_output.cpu().numpy())
            
    output = np.array(output)

    if is_multi_task:
        pred_mask = np.argmax(output[:, :3], 1)
        pred_gender = np.argmax(output[:, 3:5], 1)
        pred_age = np.argmax(output[:, 5:], 1)
        preds = encode_multi_class(pred_mask, pred_gender, pred_age)
    else:
        preds = np.argmax(output, 1)
        
    info['ans'] = preds
    output_path = config['output_path']
    info.to_csv(output_path, index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-s', '--saved_dir'], type=str, target='single_model;saved_dir')
    ]

    config = TestConfigParser.from_args(args, options)
    main(config)
