import logging
import math
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class CheckpointSaver:
    def __init__(
            self, directory, frequency, epochs_total,
            model_prefix, optimizer_prefix):
        self.directory = directory
        Path(self.directory).mkdir(exist_ok=True)
        self.frequency = frequency

        self.z = math.floor(math.log10(epochs_total)) + 1
        self.suffix = 'E{epoch:0{z}}.pth'
        self.model_name_pattern = f'{model_prefix}_{self.suffix}'
        self.optimizer_name_pattern = f'{optimizer_prefix}_{self.suffix}'

    def save(self, kind, to_save, epoch):
        if epoch % self.frequency == 0:
            if kind == 'model':
                checkpoint_name = self.model_name_pattern.format(
                    epoch=epoch, z=self.z)
            elif kind == 'optimizer':
                checkpoint_name = self.optimizer_name_pattern.format(
                    epoch=epoch, z=self.z)
            else:
                raise ValueError(f'Unexpected checkpoint kind {kind}')
            checkpoint_path = (Path(self.directory)/checkpoint_name).resolve().relative_to(Path.cwd())
            torch.save(to_save.state_dict(), str(checkpoint_path))
            logger.info(f'Saved checkpoint as: {checkpoint_path}')


class CheckpointRestorer:
    def __init__(
            self, directory, fetch_epoch,
            model_prefix, optimizer_prefix):
        self.directory = directory
        self.fetch_epoch = fetch_epoch
        self.model_prefix = model_prefix
        self.optimizer_prefix = optimizer_prefix
        self.status_on = self.directory is not None
        self.init_checkpoint_pattern()

    def init_checkpoint_pattern(self):
        if self.status_on:
            if self.fetch_epoch == 'latest':
                model_filenames = sorted(
                    list(Path(self.directory).rglob(f'{self.model_prefix}*.pth')))
                optimizer_filenames = sorted(
                    list(Path(self.directory).rglob(f'{self.optimizer_prefix}*.pth')))

                self.model_path = str(model_filenames[-1])
                self.optimizer_path = str(optimizer_filenames[-1])
            else:
                self.model_path = str(Path(self.directory).rglob(
                    f'{self.model_prefix}_E0*{self.fetch_epoch}.pth'))
                self.optimizer_path = str(Path(self.directory).rglob(
                    f'{self.optimizer_prefix}_E0*{self.fetch_epoch}.pth'))
            self.fetch_epoch_num = int(
                Path(self.model_path).stem.split('_E')[-1])

    def restore(self, kind, to_restore):
        if kind == 'model':
            checkpoint_path = self.model_path
        elif kind == 'optimizer':
            checkpoint_path = self.optimizer_path
        else:
            raise ValueError(f'Unexpected checkpoint kind {kind}')
        checkpoint_path = Path(checkpoint_path).resolve().relative_to(Path.cwd())
        to_restore.load_state_dict(torch.load(str(checkpoint_path)))
        logger.info(f'Restored {kind} checkpoint from: {checkpoint_path}')
