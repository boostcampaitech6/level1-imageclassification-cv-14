import torch
from base import BaseTrainer
from utils import MetricTracker
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metrics, optimizer, config, fold,
                 exp_name, exp_num, project_name, entity,
                 device, train_loader, valid_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metrics, optimizer, config, fold,
                         exp_name, exp_num, project_name, entity,)
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.do_validation = self.valid_loader is not None
        self.lr_scheduler = lr_scheduler
        self.fold = fold

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(tqdm(
            self.train_loader, 
            desc="[Fold {} - Train Epoch {}]".format(self.fold, epoch)
        )):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metrics:
                self.train_metrics.update(met.__name__, met(output, target))

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
            for batch_idx, (data, target) in enumerate(tqdm(
                self.valid_loader, 
                desc="[Fold {} - Valid Epoch {}]".format(self.fold, epoch)
            )):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()
