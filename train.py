import argparse
import collections
import torch
import numpy as np
import torch.utils.data as torch_utils
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
import trainer as module_trainer
from parse_config import ConfigParser
from utils import prepare_device, encode_multi_class
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
 

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config, exp_name, exp_num, project_name, entity,):
    logger = config.get_logger('train')

    dataset = config.init_obj('dataset', module_data)
    labels = [
        encode_multi_class(mask, gender, age)
        for mask, gender, age in zip(
            dataset.mask_labels, dataset.gender_labels, dataset.age_labels
        )
    ]
    
    k_splits = config['k_splits']
    skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=SEED)

    results = []
    for fold, (train_ids, valid_ids) in enumerate(skf.split(dataset, labels)):
        train_subset = Subset(dataset, train_ids)
        valid_subset = Subset(dataset, valid_ids)

        train_sampler = torch_utils.RandomSampler(train_subset)
        valid_sampler = torch_utils.SequentialSampler(valid_subset)

        train_loader = config.init_obj('train_loader', module_data, train_subset, sampler=train_sampler)
        valid_loader = config.init_obj('valid_loader', module_data, valid_subset, sampler=valid_sampler)

        model = config.init_obj('arch', module_arch)
        logger.info(model)

        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer_kwargs = {
            'model': model,
            'criterion': criterion,
            'metrics': metrics,
            'optimizer': optimizer,
            'config': config,
            'fold': fold + 1,
            'exp_name': exp_name,
            'exp_num': exp_num,
            'project_name': project_name,
            'entity': entity,
            'device': device,
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'lr_scheduler': lr_scheduler
        }
        trainer = config.init_obj('trainer', module_trainer, **trainer_kwargs)

        result = trainer.train()
        results.append(result)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-e', '--exp_name', default='exp', type=str,
                      help='name of experiment. ex) cnn_test')
    args.add_argument('-n', '--exp_num', default=0, type=int,
                      help='experience number')
    args.add_argument('--project_name', default='Image Classification', type=str,
                      help='our project name (default: Image Classification)')
    args.add_argument('--entity', default='cv-14', type=str,
                      help='our team name')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    main_kwargs = {
        'config': config,
        'exp_name': args.exp_name,
        'exp_num': args.exp_num,
        'project_name': args.project_name,
        'entity': args.entity,
    }
    main(**main_kwargs)
