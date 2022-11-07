import logging
import math
from pathlib import Path
from shutil import copytree, ignore_patterns

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

logger = logging.getLogger(__name__)


class BaseRecorder:
    def __init__(self, directory):
        self.directory = directory
        self.status_on = False
        if self.directory is not None:
            Path(self.directory).mkdir(exist_ok=True)
            self.status_on = True


class Backuper(BaseRecorder):
    def __init__(
        self, directory, ignore_patterns, include_folders):
        super().__init__(directory)
        if self.status_on:
            self.ignore_patterns = ignore_patterns
            self.include_folders = include_folders
            self.directory_srcs = [str(Path.cwd()/p) for p in self.include_folders]
                
    def backup(self):
        for dir_src in self.directory_srcs:
            dir_dst = str(Path(self.directory)/Path(dir_src).name)
            copytree(
                dir_src, dir_dst,
                ignore=ignore_patterns(*self.ignore_patterns),
                dirs_exist_ok=True)
        logger.info(f'Backup done.')


class TbRecorder(BaseRecorder):
    def __init__(self, directory):
        super().__init__(directory)
        if self.status_on:
            self.writer = SummaryWriter(log_dir=self.directory)        
        
    def record(self, tag, val, step):
        self.writer.add_scalar(tag, val, global_step=step)


class VisRecorder(BaseRecorder):
    def __init__(self, directory, frequency, epochs_total):
        super().__init__(directory)
        if self.status_on:
            self.frequency = frequency
            self.epochs_total = epochs_total
            for s in ['src', 'res']:
                (Path(self.directory)/s).mkdir(exist_ok=True)
            self.z = math.floor(math.log10(self.epochs_total)) + 1
            self.image_name_pattern = '{image_stem}_E{epoch:0{z}}_{tag}.jpg'

    def record(self, image_tensors, image_filenames, epoch, kind, tag=None):
        if epoch % self.frequency == 0:
            assert kind in ['src', 'res']
            if kind == 'src' and epoch > 1:
                return
            if tag is None:
                tag = kind
            for (image_tensor, image_filename) in zip(image_tensors, image_filenames):
                image_stem = Path(image_filename).stem
                image_folder = Path(self.directory)/kind
                image_name = self.image_name_pattern.format(
                    image_stem=image_stem, epoch=epoch, z=self.z, tag=tag)
                image_path = image_folder/image_name
                save_image(image_tensor, image_path)
