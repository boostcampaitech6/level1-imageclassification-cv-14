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
    saved_models = config['resume']['saved_models']
    saved_path = Path(config['resume']['saved_dir']) / \
                    config['resume']['saved_exp_name'] / \
                    str(config['resume']['saved_exp_num'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    info = pd.read_csv(config['info_path'])
    img_paths = [Path(config['image_path']) / img_id for img_id in info.ImageID]

    test_set = TestDataset(img_paths, config['resize'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    n_ensemble = len(saved_models)
    mode = config['mode']
    out_features = config['out_features']

    if mode == "mean":
        weights = [1 for _ in range(n_ensemble)]
    elif mode == "weighted":
        weights = config['weights']
        assert len(weights) == n_ensemble
    else:
        assert False

    if out_features == 8:
        is_multi_label = True
    elif out_features == 18:
        is_multi_label = False
    else:
        assert False

    outputs = None
    for idx, model_pth in enumerate(saved_models):
        model_path = saved_path / model_pth

        checkpoint = torch.load(model_path)

        model_config = checkpoint['config']
        state_dict = checkpoint['state_dict']

        model = model_config.init_obj('arch', module_arch)
        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = []
            for _, data in enumerate(tqdm(test_loader)):
                data = data.to(device)
                _output = model(data)
                output.extend(_output.cpu().numpy())

            if outputs is None:
                outputs = np.array(output) * weights[idx]
            else:
                outputs += np.array(output) * weights[idx]

    if is_multi_label:
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
        CustomArgs(['-e', '--exp_name'], type=str, target='resume;saved_exp_name'),
        CustomArgs(['-n', '--exp_num'], type=int, target='resume;saved_exp_num')
    ]

    config = TestConfigParser.from_args(args, options)
    main(config)
