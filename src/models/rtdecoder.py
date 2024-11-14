from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from emg_decoder.src.data.preprocess import preprocess
from emg_decoder.src.models.architectures.emgnet import EMGNet
from emg_decoder.src.models.losses.levenshtein_loss import LevenshteinLoss
from emg_decoder.src.models.architectures.temporalnet import TemporalClassifier
from emg_decoder.src.data.datasets import ContinuousKeyDataset, IntervalDataset

num_workers = 4


def ctc_collate(data: list) -> Tuple[tensor, tensor, tensor]:
    """
    Collate function for CTC loss.
    :param data: List of tuples containing (X, y) pairs.
    :return: Concatenated emg tensors, concatenated key tensors, input lengths (should all be the same), and target
    lengths (likely differing).
    """
    Xs = []
    ys = []
    target_lens = []
    for X, y in data:
        Xs.append(torch.unsqueeze(X, 0).float())
        ys.append(torch.from_numpy(y))
        target_lens.append(ys[-1].size(dim=-1))
    return torch.cat(Xs, 0), torch.cat(ys, 0), torch.LongTensor(target_lens)


class RTDecoder(LightningModule):
    def __init__(self,
                 config: dict,
                 accelerator: str = 'cpu') -> None:
        """
        Initialize a real-time decoder model.
        :param config: Config dict containing all training and data hyperparameters.
        :param accelerator: Accelerator to use (e.g. "cpu", "gpu", or "mps").
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.model_config = config['model']
        self.data_config = config['data']
        self.pre_config = self.config["preprocessing"]

        self.accelerator = accelerator

        # Metadata-related attributes
        self.effective_sampling_freq = config['metadata']['sampling_frequency'] // config['preprocessing'][
            'downsample_factor']

        self.software = config['metadata'].get('software', 'raspy').lower()
        if self.software != "raspy" and self.software != "rumi":
            raise NotImplementedError("Software must be either 'raspy' or 'rumi'.")

        # Data-related attributes, will be set in preprocess_data() but cleared after prepare_data() for memory
        self.X = None
        self.y = None
        self.window_idxs = None
        self.num_chans = 0
        self.key_list = None

        self.preprocess_data()

        self.num_classes = len(self.key_list) + 1  # number of key classes + blank symbol

        self.window_len = int(self.effective_sampling_freq * self.data_config['window_len'])

        # Network-related attributes
        if self.model_config['type'] == "TemporalNet":
            self.rnn_type = self.model_config['rnn_type']
            self.dataset_type = ContinuousKeyDataset

            self.model = TemporalClassifier(self.model_config["conv_layers"],
                                            num_chans=self.num_chans,
                                            num_classes=self.num_classes,
                                            rnn_type=self.rnn_type)
        elif self.model_config['type'] == "EMGNet":
            self.rnn_type = self.model_config['rnn_type']
            self.dataset_type = IntervalDataset 

            self.model = EMGNet(num_temporal_filts=self.model_config['num_temporal_filters'],
                                num_spatial_filts=self.model_config['num_spatial_filters'],
                                num_chans=self.num_chans,
                                window_length=self.window_len,
                                num_classes=self.num_classes,
                                embed_dim=self.model_config['embed_dims'],
                                num_layers=self.model_config['num_rnn_layers'],
                                effective_fs=self.effective_sampling_freq,
                                avgpool_factor1=self.model_config['avgpool_factor'],
                                p_dropout=self.model_config['p_dropout']
                                )
        else:
            raise NotImplementedError("Model type must be either 'TemporalNet', 'EMGNet', or 'EEGNetGRU'.")

        # Training-related attributes
        self.loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.accuracy_metric = LevenshteinLoss
        optimizer_config = config['optimizer']
        self.lr = optimizer_config['learning_rate']
        self.eps = optimizer_config['eps']
        self.decay = optimizer_config['weight_decay']

        self.train_dataset, self.val_dataset, self.test_dataset, self.predict_dataset = self.prepare_data()

    def preprocess_data(self):
        """Either load preprocessed data from disk or process raw data anew and save."""
        no_data_flag = True

        if self.data_config['preprocessed_data_dir'] is not None:
            processed_data_dir = self.data_config['preprocessed_data_dir']
            print(f"Loading preprocessed data from {processed_data_dir}")
            try:
                self.X = np.load(f"{processed_data_dir}/X.npy")
                self.y = np.load(f"{processed_data_dir}/y.npy")
                self.window_idxs = np.load(f"{processed_data_dir}/windows.npy")
                self.key_list = np.load(f"{processed_data_dir}/key_list.npy").tolist()
                no_data_flag = False
            except FileNotFoundError:
                print(f"Preprocessed data not found in {processed_data_dir}")

        if no_data_flag:
            print(f"Processing raw data from {self.pre_config['raw_data_dir']}")
            processed_data_dir = Path(f"{self.config['root_dir']}/data/processed/{self.config['name']}")
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            self.X, self.y, self.window_idxs, self.key_list = preprocess(self.config, software=self.software)
            print(f"Saving preprocessed data to {processed_data_dir}")
            np.save(f"{processed_data_dir}/X.npy", self.X)
            np.save(f"{processed_data_dir}/y.npy", self.y)
            np.save(f"{processed_data_dir}/windows.npy", self.window_idxs)
            np.save(f"{processed_data_dir}/key_list.npy", self.key_list)
            self.data_config['preprocessed_data_dir'] = processed_data_dir

        self.num_chans = self.X.shape[-1]

    def prepare_data(self) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
        """
        Load emg data, labels, and sliding window indices from preprocessed data directory and turn into train, val
        and test datasets.
        """
        data_config = self.config["data"]

        window_train, window_val_and_test = train_test_split(self.window_idxs,
                                                             train_size=data_config[
                                                                 'train_proportion'],
                                                             shuffle=False)
        window_val, window_test = train_test_split(window_val_and_test,
                                                   train_size=data_config['val_proportion'] / (
                                                           data_config['val_proportion'] +
                                                           data_config['test_proportion']),
                                                   shuffle=False)

        if self.dataset_type == ContinuousKeyDataset:
            train_dataset = ContinuousKeyDataset(self.X, self.y, window_train, window_length=self.window_len)
            val_dataset = ContinuousKeyDataset(self.X, self.y, window_val, window_length=self.window_len)
            test_dataset = ContinuousKeyDataset(self.X, self.y, window_test, window_length=self.window_len)
            predict_dataset = test_dataset
        elif self.dataset_type == IntervalDataset:
            interval_length = int(self.data_config['interval_len'] * self.effective_sampling_freq)
            stride = int(self.data_config['window_stride'] * self.effective_sampling_freq)
            window_length = int(self.data_config['window_len'] * self.effective_sampling_freq)
            train_dataset = IntervalDataset(self.X, self.y, window_train,
                                            interval_length=interval_length,
                                            stride=stride,
                                            window_length=window_length)
            val_dataset = IntervalDataset(self.X, self.y, window_val,
                                          interval_length=interval_length,
                                          stride=stride,
                                          window_length=window_length)
            test_dataset = IntervalDataset(self.X, self.y, window_test,
                                           interval_length=interval_length,
                                           stride=stride,
                                           window_length=window_length)
            # Return raw y values for prediction dataset
            predict_dataset = IntervalDataset(self.X, self.y, window_test,
                                              interval_length=interval_length,
                                              stride=stride,
                                              window_length=window_length,
                                              return_raw=True)
        else:
            raise ValueError("Dataset should be one of ContinuousKeyDataset or IntervalDataset")

        return train_dataset, val_dataset, test_dataset, predict_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["train_batch_size"],
            collate_fn=ctc_collate,
            shuffle=True,
            num_workers=num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["val_batch_size"],
            collate_fn=ctc_collate,
            shuffle=False,
            num_workers=num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["test_batch_size"],
            collate_fn=ctc_collate,
            num_workers=num_workers,
            shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        # Use default collate for raw y values
        return DataLoader(
            self.predict_dataset,
            batch_size=self.config["data"]["test_batch_size"],
            shuffle=False,
            num_workers=num_workers)

    def forward(self, x: tensor) -> tensor:
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.decay)

    def training_step(self, batch: tensor, batch_idx: int) -> float:
        x, y, target_lengths = batch
        log_probs = torch.swapaxes(self.model(x), 0, 1)
        input_length, batch_size, num_classes = log_probs.shape
        input_lengths = torch.full((batch_size,), input_length)
        loss = self.loss(log_probs, y, input_lengths, target_lengths)
        acc = self.accuracy_metric.forward(log_probs, y, target_lengths)

        self.log_dict({
            'train_loss': loss,
            'train_acc': acc,
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tensor, batch_idx: int) -> float:
        x, y, target_lengths = batch
        log_probs = torch.swapaxes(self.model(x), 0, 1)
        input_length, batch_size, num_classes = log_probs.shape
        input_lengths = torch.full((batch_size,), input_length)
        loss = self.loss(log_probs, y, input_lengths, target_lengths)
        acc = self.accuracy_metric.forward(log_probs, y, target_lengths)

        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: tensor, batch_idx: int) -> float:
        x, y, target_lengths = batch
        log_probs = torch.swapaxes(self.model(x), 0, 1)
        input_length, batch_size, num_classes = log_probs.shape
        input_lengths = torch.full((batch_size,), input_length)
        loss = self.loss(log_probs, y, input_lengths, target_lengths)
        acc = self.accuracy_metric.forward(log_probs, y, target_lengths)

        self.log_dict({
            'test_loss': loss,
            'test_acc': acc
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        log_probs = torch.swapaxes(self.model(x), 0, 1)
        return log_probs, y
