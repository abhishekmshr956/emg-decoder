import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import tensor, nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from emg_decoder.src.data.datasets import KeyDataset
from emg_decoder.src.models.architectures.eegnet import EEGNet
from emg_decoder.src.models.losses.spatial_loss import SpatialLoss, Keyboard


class MulticlassAccuracy(nn.Module):
    def __init__(self):
        super(MulticlassAccuracy, self).__init__()

    def forward(self, logits: tensor, y: tensor) -> float:
        preds = torch.argmax(logits, dim=-1)
        acc = torch.sum(preds == y) / len(y)
        return acc


class BinaryAccuracy(nn.Module):
    def __init__(self):
        super(BinaryAccuracy, self).__init__()

    def forward(self, logits: tensor, y: tensor) -> float:
        preds = torch.round(torch.sigmoid(logits))
        acc = torch.sum(preds == y) / len(y)
        return acc


class KeyClassifier(LightningModule):
    def __init__(self,
                 num_temporal_filts: int = 8,
                 num_spatial_filts: int = 2,
                 num_chans: int = 32,
                 window_length: int = 100,
                 p_dropout: float = 0.5,
                 avgpool_factor: int = 4,
                 out_dims: int = 256,
                 num_classes: int = 26,
                 binary: bool = False,
                 data_scale: float = 1.0):
        super().__init__()
        self.model = EEGNet(num_temporal_filts=num_temporal_filts,
                            num_spatial_filts=num_spatial_filts,
                            num_chans=num_chans,
                            window_length=window_length,
                            p_dropout=p_dropout,
                            avgpool_factor=avgpool_factor)

        # print(num_chans, out_dims, num_classes)

        self.classifier = nn.Linear(out_dims, num_classes)
        self.binary = binary
        self.register_buffer('data_scale', torch.tensor(data_scale))
        print('data_scale', self.data_scale)
        

    def forward(self, x):
        x = self.data_scale*x
        x = self.model(x)
        # print('x', x.shape)
        # with open('/home/john/emg_decoder/out1.npy', 'wb') as f:
        #     np.save(f, np.array(x.cpu()))
        out = self.classifier(x)
        # print("out", out.shape)
        if self.binary:
            return torch.squeeze(out)
        else:
            return out


class KeyDecoder(LightningModule):
    def __init__(self,
                 config: dict,
                 accelerator: str = 'cpu',
                 num_workers: int = 4,
                 inference: bool = False) -> None:
        """
        Initialize a continuous decoder.
        :param config: Config dict containing all training and data hyperparameters.
        :param accelerator: Accelerator to use (e.g. "cpu", "gpu", or "mps").
        """
        super().__init__()
        self.config = config
        self.accelerator = accelerator

        data_config = self.config['data']
        self.num_workers = data_config.get('num_workers', num_workers)
        self.window_len = data_config['window_length']
        self.dataset_type = KeyDataset
        self.num_chans = data_config.get('num_chans', None)
        self.num_classes = data_config.get('num_classes', None)
        self.classes = data_config.get('classes', None)
        model_config = self.config['model']
        self.binary = model_config["loss"].lower() == "binarycrossentropyloss"
        self.optimizer = torch.optim.Adam

        if not inference:
            self.train_dataset, self.val_dataset, self.test_dataset = self.prepare_data()

        if model_config["loss"].lower() == "spatialloss":
            kb = Keyboard()
            coords = torch.tensor(kb.ascii_sorted_coords, device=self.accelerator, requires_grad=True)
            self.loss = SpatialLoss(coords)
            self.acc = MulticlassAccuracy()
        elif model_config["loss"].lower() == "crossentropyloss":
            self.loss = CrossEntropyLoss()
            self.acc = MulticlassAccuracy()
        elif model_config["loss"].lower() == "binarycrossentropyloss":
            self.loss = BCEWithLogitsLoss()
            self.acc = BinaryAccuracy()
            self.num_classes = 1
            self.binary = True
        else:
            raise NotImplementedError(f"Loss {model_config['loss']} not implemented.")

        data_scale = model_config.get('data_scale', 1.0)
        self.register_buffer('data_scale', torch.tensor(data_scale))

        if not inference:
            self.config['data']['num_chans'] = self.num_chans
            self.config['data']['num_classes'] = self.num_classes
            self.config['data']['classes'] = self.classes

        self.model = KeyClassifier(num_temporal_filts=model_config['num_temporal_filters'],
                                   num_spatial_filts=model_config['num_spatial_filters'],
                                   num_chans=self.num_chans,
                                   window_length=self.window_len,
                                   p_dropout=model_config['p_dropout'],
                                   avgpool_factor=model_config['avgpool_factor'],
                                   out_dims=256,
                                   num_classes=self.num_classes,
                                   binary=self.binary,
                                   data_scale=self.data_scale.item())

        self.save_hyperparameters(self.config)

    def prepare_data(self) -> tuple[KeyDataset, KeyDataset, KeyDataset]:
        data_config = self.config['data']
        train_dataset = np.load(data_config['train_path'])
        val_dataset = np.load(data_config['val_path'])
        print(data_config['test_path'])
        test_dataset = np.load(data_config['test_path'])
        self.num_chans = train_dataset['emg_windows'].shape[-1]
        self.num_classes = len(np.unique(train_dataset['key_label']))
        self.classes = np.sort(np.unique(train_dataset['key_label'])).tolist()

        # Convert key ascii codes to consecutive integers
        for idx, key in enumerate(self.classes):
            train_dataset['key_label'][train_dataset['key_label'] == key] = idx
            val_dataset['key_label'][val_dataset['key_label'] == key] = idx
            test_dataset['key_label'][test_dataset['key_label'] == key] = idx

        return (self.dataset_type(train_dataset['emg_windows'],
                                  train_dataset['key_label'],
                                  binary=self.binary),
                self.dataset_type(val_dataset['emg_windows'],
                                  val_dataset['key_label'],
                                  binary=self.binary),
                self.dataset_type(test_dataset['emg_windows'],
                                  test_dataset['key_label'],
                                  binary=self.binary))

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

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["test_batch_size"],
            shuffle=False,
            num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["test_batch_size"],
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
        # print(X.shape, y.shape, "hi")
        if feats.ndim == 1:
            feats = feats.unsqueeze(dim=0)
        loss = self.loss(feats, y)
        acc = self.acc(feats, y)
        self.log_dict({
            'train_loss': loss,
            'train_acc': acc,
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tensor, batch_idx: int) -> float:
        X, y = batch
        feats = self.model(X)
        if feats.ndim == 1:
            feats = feats.unsqueeze(dim=0)
        loss = self.loss(feats, y)
        acc = self.acc(feats, y)
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: tensor, batch_idx: int) -> float:
        X, y = batch
        feats = self.model(X)
        if feats.ndim == 1:
            feats = feats.unsqueeze(dim=0)
        loss = self.loss(feats, y)
        acc = self.acc(feats, y)
        self.log_dict({
            'test_loss': loss,
            'test_acc': acc
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch
        feats = self.model(X)
        if feats.ndim == 1:
            feats = feats.unsqueeze(dim=0)
        loss = self.loss(feats, y)
        acc = self.acc(feats, y)
        return feats, y, loss, acc
