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
from utils import prepare_device
 

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    dataset = config.init_obj('dataset', module_data)
    train_set, valid_set = dataset.split_dataset()

    train_sampler = torch_utils.RandomSampler(train_set)
    valid_sampler = torch_utils.RandomSampler(valid_set)

    train_loader = config.init_obj('train_loader', module_data, train_set, sampler=train_sampler)
    valid_loader = config.init_obj('valid_loader', module_data, valid_set, sampler=valid_sampler)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
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
        'lr_scheduler': lr_scheduler
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
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
