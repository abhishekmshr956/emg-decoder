from typing import Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import tensor, nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from emg_decoder.src.data.datasets import AugKeyDataset
from emg_decoder.src.models.architectures.eegnet import EEGNet
from emg_decoder.src.models.losses.supcon_loss import SupConLoss


class Encoder(LightningModule):
    def __init__(self, num_chans, window_len, num_temporal_filts: int = 32, num_spatial_filts: int = 2,
                 p_dropout: float = 0.5, avgpool_factor1: int = 2):
        super().__init__()
        self.encoder = EEGNet(num_temporal_filts=num_temporal_filts,
                              num_spatial_filts=num_spatial_filts,
                              num_chans=num_chans,
                              window_length=window_len,
                              p_dropout=p_dropout,
                              avgpool_factor=avgpool_factor1)

    def forward(self, x):
        return self.encoder(x)


class SupConEncoder(LightningModule):
    def __init__(self, num_chans, window_len, num_temporal_filts: int = 32, num_spatial_filts: int = 2,
                 p_dropout: float = 0.5, avgpool_factor1: int = 2, head: str = 'linear', output_dim: int = 256,
                 feat_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(num_chans, window_len,
                               num_temporal_filts=num_temporal_filts,
                               num_spatial_filts=num_spatial_filts,
                               p_dropout=p_dropout,
                               avgpool_factor1=avgpool_factor1)
        if head == 'linear':
            self.head = nn.Linear(output_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, feat_dim)
            )
        else:
            raise NotImplementedError(f"Head {head} not supported.")

    def forward(self, x):
        bsz, num_views, num_chans, window_len = x.shape
        x = x.view(bsz * num_views, num_chans, window_len)
        encoding = F.normalize(self.encoder(x))
        projection = F.normalize(self.head(encoding))
        return projection.view(bsz, num_views, -1)


class ConKeyDecoder(LightningModule):
    def __init__(self,
                 config: dict,
                 accelerator: str = 'cpu',
                 num_workers: int = 4) -> None:
        """
        Initialize a continuous decoder.
        :param config: Config dict containing all training and data hyperparameters.
        :param accelerator: Accelerator to use (e.g. "cpu", "gpu", or "mps").
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.accelerator = accelerator

        data_config = self.config['data']
        self.num_workers = data_config.get('num_workers', num_workers)
        self.window_len = int(data_config['subwindow_fraction'] * data_config['window_length'])
        self.dataset_type = AugKeyDataset
        self.num_chans = None
        self.num_classes = None
        self.train_dataset, self.val_dataset = self.prepare_data()

        self.model = SupConEncoder(self.num_chans,
                                   self.window_len,
                                   num_temporal_filts=64,
                                   num_spatial_filts=4,
                                   p_dropout=0.5,
                                   avgpool_factor1=2,
                                   head='linear',
                                   output_dim=256,
                                   feat_dim=128)

        self.loss = SupConLoss(temperature=0.07,
                               contrast_mode='all',
                               base_temperature=0.07,
                               device=self.accelerator)

        self.optimizer = torch.optim.Adam

    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        data_config = self.config['data']
        data = np.load(data_config['path'])
        self.num_chans = data['emg_windows'].shape[-1]
        self.num_classes = len(np.unique(data['key_label']))
        train_dataset, val_dataset = train_test_split(data,
                                                      train_size=data_config['train_proportion'],
                                                      stratify=data['key_label'],
                                                      random_state=self.config['random_seed'])

        return (self.dataset_type(train_dataset['emg_windows'],
                                  train_dataset['key_label'],
                                  num_views=data_config['num_views'],
                                  subwindow_fraction=data_config['subwindow_fraction']),
                self.dataset_type(val_dataset['emg_windows'],
                                  val_dataset['key_label'],
                                  num_views=data_config['num_views'],
                                  subwindow_fraction=data_config['subwindow_fraction']),)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["train_batch_size"],
            shuffle=True,
            num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["val_batch_size"],
            shuffle=False,
            num_workers=self.num_workers)

    def forward(self, x: tensor) -> tensor:
        return self.model(x)

    def configure_optimizers(self) -> Optimizer:
        optimizer_config = self.config['optimizer']
        return self.optimizer(self.parameters(),
                              lr=optimizer_config.get('lr', 1e-3),
                              eps=optimizer_config.get('eps', 1e-8),
                              weight_decay=optimizer_config.get('weight_decay', 0))

    def training_step(self, batch: tensor, batch_idx: int) -> float:
        X, y = batch
        feats = self.model(X)
        loss = self.loss(feats, labels=y)
        self.log_dict({
            'train_loss': loss,
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tensor, batch_idx: int) -> float:
        X, y = batch
        feats = self.model(X)
        loss = self.loss(feats, labels=y)
        self.log_dict({
            'val_loss': loss,
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss
