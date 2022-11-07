import logging
import torch
import torch.nn as nn
from tqdm import tqdm

from scripts.base.data import MetricMonitor
from scripts.base.checkpoint import CheckpointRestorer, CheckpointSaver
from scripts.base.recorder import Backuper, TbRecorder, VisRecorder


logger = logging.getLogger(__name__)


def make_criterion(kind, **kwargs):
    if kind == 'BCEWithLogitsLoss':
        criterion_class = nn.BCEWithLogitsLoss
    elif kind == 'L1Loss':
        criterion_class = nn.L1Loss
    elif kind == 'MSELoss':
        criterion_class = nn.MSELoss
    elif kind == 'CrossEntropyLoss':
        criterion_class = nn.CrossEntropyLoss
    else:
        raise ValueError(f'Unknown criterion kind {kind}')
    return criterion_class(**kwargs)


def make_criterions_dict(oc_criterions, device):
    criterions_dict = {}
    for k, v in dict(oc_criterions).items():
        criterions_dict[k] = {'crit': make_criterion(v.name).to(device),
                              'weight': v.weight}
    return criterions_dict


def make_optimizer(kind, parameters, **kwargs):
    if kind == 'Adam':
        optimizer_class = torch.optim.Adam
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def make_ckpt_saver(kind='base', **kwargs):
    if kind == 'base':
        ckpt_saver_class = CheckpointSaver
    else:
        raise ValueError(f'Unknown checkpoint saver kind {kind}')
    return ckpt_saver_class(**kwargs)


def make_ckpt_restorer(kind='base', **kwargs):
    if kind == 'base':
        ckpt_restorer_class = CheckpointRestorer
    else:
        raise ValueError(f'Unknown checkpoint restorer kind {kind}')
    return ckpt_restorer_class(**kwargs)


def make_backuper(kind='base', **kwargs):
    if kind == 'base':
        backuper_class = Backuper
    else:
        raise ValueError(f'Unknown backuper kind {kind}')
    return backuper_class(**kwargs)


def make_tb_recorder(kind='base', **kwargs):
    if kind == 'base':
        tb_recorder_class = TbRecorder
    else:
        raise ValueError(f'Unknown tb-recorder kind {kind}')
    return tb_recorder_class(**kwargs)


def make_vis_recorder(kind='base', **kwargs):
    if kind == 'base':
        vis_recorder_class = VisRecorder
    else:
        raise ValueError(f'Unknown vis-recorder kind {kind}')
    return vis_recorder_class(**kwargs)


class BaseTrainer:
    def __init__(
            self, lr, epochs, device, checkpoint, recorder, predict_only=False):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.epoch_start = 1
        self.predict_only = predict_only
        if self.predict_only:
            self.ckpt_restorer = make_ckpt_restorer(**checkpoint['restore'])
            self.backuper = make_backuper(**recorder['backup'])
            self.vis_recorder = make_vis_recorder(**recorder['vis'])
            self.model = nn.Identity()
            self.test_loader = None
        else:
            self.ckpt_saver = make_ckpt_saver(**checkpoint['save'])
            self.ckpt_restorer = make_ckpt_restorer(**checkpoint['restore'])
            self.backuper = make_backuper(**recorder['backup'])
            self.tb_recorder = make_tb_recorder(**recorder['tb'])
            self.vis_recorder = make_vis_recorder(**recorder['vis'])
            self.model = nn.Identity()
            self.optimizer = None
            self.train_loader = None
            self.val_loader = None
            self.train_criterions_dict = {}
            self.val_criterions_dict = {}

    def train_per_epoch(self, epoch):
        metric_monitor = MetricMonitor()
        self.model.train()
        stream = tqdm(self.train_loader)
        for i, data in enumerate(stream, start=1):
            images = data['image'].to(self.device, non_blocking=True)
            target = data['target'].to(self.device, non_blocking=True)
            output = self.model(images).squeeze(1)
            ls_loss = []
            for d in self.train_criterions_dict.values():
                ls_loss.append(d['crit'](output, target)*d['weight'])
            loss = sum(ls_loss)
            metric_monitor.update("Loss", loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            stream.set_description(
                f'Epoch: {epoch}. Train.      {metric_monitor}')
            self.tb_recorder.record(
                'loss/total', loss, (epoch-1)*len(self.train_loader)+(i-1))
        logger.info(f'Epoch: {epoch}. Train.      {metric_monitor}')

    def validate_per_epoch(self, epoch):
        metric_monitor = MetricMonitor()
        self.model.eval()
        stream = tqdm(self.val_loader)
        with torch.no_grad():
            for i, data in enumerate(stream, start=1):
                images = data['image'].to(self.device, non_blocking=True)
                target = data['target'].to(self.device, non_blocking=True)
                image_filenames = data['image_filename']
                output = self.model(images).squeeze(1)
                ls_loss = []
                for d in self.val_criterions_dict.values():
                    ls_loss.append(d['crit'](output, target)*d['weight'])
                loss = sum(ls_loss)
                metric_monitor.update("Loss", loss.item())
                stream.set_description(
                    f'Epoch: {epoch}. Validation. {metric_monitor}')
                self.vis_recorder.record(
                    images, image_filenames, epoch, 'src', 'normalized')
                self.vis_recorder.record(
                    target, image_filenames, epoch, 'src', 'target')
                self.vis_recorder.record(
                    output, image_filenames, epoch, 'res', 'output')
                # todo: 'images' are transformed, maybe visrecord in prediction?
        logger.info(f'Epoch: {epoch}. Validation. {metric_monitor}')

    def train_and_val(self):
        for epoch in range(self.epoch_start, self.epochs + 1):
            self.train_per_epoch(epoch)
            self.validate_per_epoch(epoch)
            self.ckpt_saver.save('model', self.model, epoch)
            self.ckpt_saver.save('optimizer', self.optimizer, epoch)
        logger.info(f'Training finished.')
