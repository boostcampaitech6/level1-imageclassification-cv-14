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

    saved_models = config['multi_model']['saved_models']
    saved_path = config['multi_model']['saved_dir']

    info = pd.read_csv(config['info_path'])
    img_paths = [Path(config['image_path']) / img_id for img_id in info.ImageID]

    test_set = TestDataset(img_paths, config['resize'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    is_multi_task = config['multi_model']['is_multi_task']
    is_deep_model = config['multi_model']['is_deep_model']

    if is_deep_model:
        outputs = [[], [], []]
        num_classes = [3, 2, 3] # mask, gender, age

        for idx, model_pth in enumerate(saved_models):
            model_path = Path(saved_path) / model_pth
            checkpoint = torch.load(model_path)

            model_config = checkpoint['config']
            state_dict = checkpoint['state_dict']

            model = model_config.init_obj('arch', module_arch, num_classes=num_classes[idx])
            model.load_state_dict(state_dict)

            model = model.to(device)
            model.eval()

            output = []
            with torch.no_grad():
                for _, data in enumerate(tqdm(test_loader)):
                    data = data.to(device)
                    _output = model(data).logits
                    output.extend(_output.cpu().numpy())

            outputs[idx] = np.array(output)

        pred_mask = np.argmax(outputs[0], 1)
        pred_gender = np.argmax(outputs[1], 1)
        pred_age = np.argmax(outputs[2], 1)
        preds = encode_multi_class(pred_mask, pred_gender, pred_age)

    else:
        outputs = None

        for idx, model_pth in enumerate(saved_models):
            model_path = Path(saved_path) / model_pth
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

            if outputs is None:
                outputs = np.array(output)
            else:
                outputs += np.array(output)

        if is_multi_task:
            pred_mask = np.argmax(outputs[:, :3], 1)
            pred_gender = np.argmax(outputs[:, 3:5], 1)
            pred_age = np.argmax(outputs[:, 5:], 1)
            preds = encode_multi_class(pred_mask, pred_gender, pred_age)
        else:
            preds = np.argmax(outputs, 1)
        
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
        CustomArgs(['-e', '--exp_name'], type=str, target='multi_model;saved_exp_name'),
        CustomArgs(['-n', '--exp_num'], type=int, target='multi_model;saved_exp_num')
    ]

    config = TestConfigParser.from_args(args, options)
    main(config)
