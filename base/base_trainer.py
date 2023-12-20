import torch
import wandb
from abc import abstractmethod
from numpy import inf


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metrics, optimizer, config, fold):
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.config = config
        self.fold = fold

        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.epochs = config['trainer']['epochs']
        self.save_period = config['trainer']['save_period']
        self.monitor = config['trainer'].get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = config['trainer'].get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.checkpoint_dir = config.save_dir
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
        
        self.start_epoch = 1

        # wandb
        self.wandb_tag = [config['arch']['type']]
        self.exp_name = config['wandb']['exp_name']
        self.exp_num = config['wandb']['exp_num']
        self.project_name = config['wandb']['project_name']
        self.entity = config['wandb']['entity']

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        # wandb init
        wandb.init(
            name=f'{self.exp_name}_{self.exp_num}_fold{self.fold}',
            project=self.project_name,
            entity=self.entity
        )
        wandb.config.batch_size = self.config['train_loader']['args']['batch_size']
        wandb.config.epoch = self.config['trainer']['epochs']
        wandb.config.loss = self.config['loss']
        wandb.config.optimizer = self.config['optimizer']['type']
        wandb.config.init_lr = self.config['optimizer']['args']['lr']
        wandb.config.lr_scheduler = {
            'type': self.config['lr_scheduler']['type'],
            'mode': self.config['lr_scheduler']['args']['mode'],
            'patience': self.config['lr_scheduler']['args']['patience'],
            'min_lr': self.config['lr_scheduler']['args']['min_lr']
        }
        wandb.watch(self.model)

        not_improved_count = 0
        best_result = {}
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'fold': self.fold, 'epoch': epoch}
            log.update(result)

            # wandb logging
            wandb.log(result, step=epoch)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self._save_best_checkpoint(self.fold, epoch, result)
                    best_result = log
                    not_improved_count = 0
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
                
            self.logger.info("-" * 60)

        wandb.finish()
        return best_result

    def _save_best_checkpoint(self, fold, epoch, result):
        """
        Saving best checkpoints
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'log': result,
            'config': self.config
        }

        best_path = str(self.checkpoint_dir / 'model_best_fold{}.pth'.format(fold))
        torch.save(state, best_path)
        self.logger.info('Saving current best: model_best_fold{}.pth ...'.format(fold))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
