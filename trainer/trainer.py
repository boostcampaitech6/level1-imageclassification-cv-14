import numpy as np
import torch
import wandb
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class BasicTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metrics, optimizer, config, device,
                 exp_name, exp_num, project_name, entity,
                 train_loader, valid_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metrics, optimizer, config)
        self.config = config
        self.device = device
        self.train_loader = train_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:
            # iteration-based training
            self.train_loader = inf_loop(train_loader)
            self.len_epoch = len_epoch
        self.valid_loader = valid_loader
        self.do_validation = self.valid_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)

        # wandb
        self.wandb_tag = [self.config._config["arch"]["type"]]
        self.exp_name = exp_name
        self.exp_num = exp_num
        self.project_name = project_name
        self.entity = entity
        self.global_step = 0

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # wandb init
        wandb.init(
            name=f'{self.exp_name}_{self.exp_num}',
            project=self.project_name,
            entity=self.entity
        )
        wandb.config.batch_size = self.config._config["train_loader"]["args"]["batch_size"]
        wandb.config.epoch = self.config._config["trainer"]["epochs"]
        # wandb.config.k_fold <<< need to develop
        wandb.watch(self.model)

        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metrics:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                # wandb logging
                wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr'],
                          "Train loss": loss.item()}, self.global_step)
                self.global_step += 1

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                #wandb logging
                wandb.log({"Val Loss": loss.item()}, self.global_step)
                self.global_step += 1

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)