import torch
import torch.nn.functional as F
from tqdm import tqdm
from base import BaseTrainer
from utils import MetricTracker, encode_multi_class, decode_multi_class


class MultiLabelTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metrics, optimizer, config, fold,
                 device, train_loader, valid_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metrics, optimizer, config, fold)
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
        self.model.train()
        self.train_metrics.reset()

        for _, (data, target) in enumerate(tqdm(
            self.train_loader, 
            desc="[Fold {} - Train Epoch {}]".format(self.fold, epoch)
        )):
            data, target = data.to(self.device), target.to(self.device)

            target_mask, target_gender, target_age = decode_multi_class(target)

            self.optimizer.zero_grad()
            output = self.model(data)

            output_mask = output[:, :3]
            output_gender = output[:, 3:5]
            output_age = output[:, 5:]

            loss_mask = self.criterion(output_mask, target_mask)
            loss_gender = self.criterion(output_gender, target_gender)
            loss_age = self.criterion(output_age, target_age)
            loss = loss_mask + loss_gender + loss_age

            loss.backward()
            self.optimizer.step()

            _, pred_mask = torch.max(output_mask, 1)
            _, pred_gender = torch.max(output_gender, 1)
            _, pred_age = torch.max(output_age, 1)
            pred = encode_multi_class(pred_mask, pred_gender, pred_age)
            pred = F.one_hot(pred, num_classes=18)

            self.train_metrics.update('loss', loss.item())
            for met in self.metrics:
                self.train_metrics.update(met.__name__, met(pred, target))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for _, (data, target) in enumerate(tqdm(
                self.valid_loader, 
                desc="[Fold {} - Valid Epoch {}]".format(self.fold, epoch)
            )):
                data, target = data.to(self.device), target.to(self.device)

                target_mask, target_gender, target_age = decode_multi_class(target)

                output = self.model(data)

                output_mask = output[:, :3]
                output_gender = output[:, 3:5]
                output_age = output[:, 5:]

                loss_mask = self.criterion(output_mask, target_mask)
                loss_gender = self.criterion(output_gender, target_gender)
                loss_age = self.criterion(output_age, target_age)
                loss = loss_mask + loss_gender + loss_age

                _, pred_mask = torch.max(output_mask, 1)
                _, pred_gender = torch.max(output_gender, 1)
                _, pred_age = torch.max(output_age, 1)
                pred = encode_multi_class(pred_mask, pred_gender, pred_age)
                pred = F.one_hot(pred, num_classes=18)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(pred, target))

        return self.valid_metrics.result()
