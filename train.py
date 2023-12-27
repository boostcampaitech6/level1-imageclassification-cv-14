import numpy as np
import random
import argparse
import collections
import torch
import torch.utils.data as torch_utils
from sklearn.model_selection import StratifiedKFold
import data_loader as module_data
import model as module_arch
import model.loss as module_loss
import model.metric as module_metric
import trainer as module_trainer
from parse_config import TrainConfigParser
from utils import prepare_device, encode_multi_class


# fix random seeds for reproducibility
SEED = 951
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    device, device_ids = prepare_device(config['n_gpu'])

    use_kfold = config['train_set']['use_kfold']

    if not use_kfold:
        train_set = config.init_obj('train_set', module_data, logger=logger)
        valid_set = config.init_obj('valid_set', module_data)

        train_sampler = torch_utils.RandomSampler(train_set)
        valid_sampler = torch_utils.SequentialSampler(valid_set)

        train_loader = config.init_obj('train_loader', module_data, train_set, sampler=train_sampler)
        valid_loader = config.init_obj('valid_loader', module_data, valid_set, sampler=valid_sampler)

        model = config.init_obj('arch', module_arch)
        logger.info(model)
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
            'device': device,
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'lr_scheduler': lr_scheduler,
            'fold': None
        }
        trainer = config.init_obj('trainer', module_trainer, **trainer_kwargs)
        trainer.train()
        
    else:
        k_splits = config['k_splits']
        skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=SEED)

        dataset = config.init_obj('train_set', module_data, logger=logger)
        labels = [
            encode_multi_class(mask, gender, age)
            for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)
        ]

        for fold, (train_ids, valid_ids) in enumerate(skf.split(dataset, labels)):
            train_subset = torch_utils.Subset(dataset, train_ids)
            valid_subset = torch_utils.Subset(dataset, valid_ids)

            train_sampler = torch_utils.RandomSampler(train_subset)
            valid_sampler = torch_utils.SequentialSampler(valid_subset)

            train_loader = config.init_obj('train_loader', module_data, train_subset, sampler=train_sampler)
            valid_loader = config.init_obj('valid_loader', module_data, valid_subset, sampler=valid_sampler)

            model = config.init_obj('arch', module_arch)
            logger.info(model)
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
                'device': device,
                'train_loader': train_loader,
                'valid_loader': valid_loader,
                'lr_scheduler': lr_scheduler,
                'fold': fold + 1
            }
            trainer = config.init_obj('trainer', module_trainer, **trainer_kwargs)
            trainer.train()

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
