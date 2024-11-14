import numpy as np
import torch
from torch.utils.data import Dataset

from emg_decoder.src.data.utils import remove_adj_duplicates


class KeyDataset(Dataset):
    def __init__(self, emg_data, key_labels, binary: bool = False):
        """
        :param emg_data: Field 'emg_windows' in the data file, of shape (num_samples, window_len, num_channels).
        :param key_labels: Field 'key_label' in the data file, of shape (num_samples, ) corresponding to the ASCII code.
        """
        self.emg_data = torch.from_numpy(emg_data.copy()).float()
        if binary:
            self.key_labels = torch.from_numpy(key_labels).float()
        else:
            self.key_labels = torch.from_numpy(key_labels).long()

    def __getitem__(self, idx):
        return self.emg_data[idx], self.key_labels[idx]

    def __len__(self):
        return self.key_labels.shape[0]


class AugKeyDataset(Dataset):
    """
    Dataset for supervised contrastive learning of key labels from EMG data using augmentation.
    """
    def __init__(self, emg_data, key_labels,
                 num_views: int = 5,
                 subwindow_fraction: float = 0.75):
        """
        :param emg_data: Field 'emg_windows' in the data file, of shape (num_samples, window_len, num_channels).
        :param key_labels: Field 'key_label' in the data file, of shape (num_samples, ) corresponding to the ASCII code.
        """
        self.emg_data = torch.from_numpy(emg_data).float()
        _, self.window_len, self.num_chans = self.emg_data.shape
        self.key_labels = torch.from_numpy(key_labels).float()
        self.num_views = num_views
        self.subwindow_fraction = subwindow_fraction
        self.subwindow_len = int(self.window_len * self.subwindow_fraction)

    def __getitem__(self, idx):
        if self.num_views > 0:
            return self.augment_views(idx), self.key_labels[idx]
        else:
            start_idx = int(self.window_len * (1 - self.subwindow_fraction) / 2)
            return (torch.unsqueeze(self.emg_data[idx][start_idx:start_idx + self.subwindow_len], 0),
                    self.key_labels[idx])

    def __len__(self):
        return self.key_labels.shape[0]

    def augment_views(self, idx):
        """
        Augment the EMG data window with multiple subwindows randomly shifted in time with additive white noise scaled
        to the standard deviation of each channel's amplitude.
        :return Augmented EMG data of shape (n_views, window_length, n_channels).
        """
        emg_window = self.emg_data[idx]
        stride = int((self.window_len - self.subwindow_len) / (self.num_views - 1))
        emg_windows = emg_window.unfold(0, self.subwindow_len, stride).swapaxes(1, 2)
        channel_variances = torch.var(emg_window, dim=0)
        white_noise = torch.randn((self.num_views, self.subwindow_len, self.num_chans)
                                  ) * torch.sqrt(channel_variances) / 2
        white_noise[self.num_views // 2, ...].fill_(0)  # The center view is the original window, no noise added.
        return emg_windows + white_noise


class ContinuousKeyDataset(Dataset):
    def __init__(self,
                 emg_data,
                 key_labels,
                 window_idxs,
                 window_length: int = 10000):
        """
        Dataset for creating windows of EMG data.
        :param emg_data: Array of shape (num_samples, num_channels) containing EMG data.
        :param key_labels: Array of shape (num_samples,) containing key labels for each EMG timestep.
        :param window_idxs: Array of shape (num_windows,) containing the start index of each window.
        :param window_length: Length of window in timesteps.
        """

        self.data = emg_data
        self.labels = key_labels
        self.window_idxs = window_idxs
        self.window_size = window_length

    def __len__(self):
        return self.window_idxs.shape[0]

    def __getitem__(self, idx):
        emg_idx = int(self.window_idxs[idx])
        X = torch.from_numpy(self.data[emg_idx:emg_idx + self.window_size, ...]).float()
        raw_y = self.labels[emg_idx:emg_idx + self.window_size].flatten()
        y = remove_adj_duplicates(raw_y)
        return torch.transpose(X, 0, 1), y


class IntervalDataset(torch.utils.data.Dataset):
    def __init__(self,
                 emg_data: np.ndarray,
                 key_labels: np.ndarray,
                 interval_idxs: np.ndarray,
                 interval_length: int,
                 stride: int,
                 window_length: int,
                 return_raw: bool = False) -> None:
        """
        Dataset for creating sliding windows of EMG data.
        :param emg_data: Array of shape (num_samples, num_channels) containing EMG data.
        :param key_labels: Array of shape (num_samples,) containing key labels for each EMG timestep.
        :param interval_idxs: Indices of the start of each interval.
        :param interval_length: Length of each interval.
        :param stride: Stride of the sliding window.
        :param window_length: Length of the sliding window.
        """
        self.emg_data = emg_data
        self.key_labels = key_labels
        self.interval_idxs = interval_idxs
        self.interval_length = interval_length
        self.stride = stride
        self.window_length = window_length
        self.return_raw = return_raw

    def __len__(self) -> int:
        return self.interval_idxs.shape[0]

    def __getitem__(self, idx: int):
        emg_start_idx = self.interval_idxs[idx]
        emg_sample = self.emg_data[emg_start_idx:emg_start_idx + self.interval_length]
        X = np.lib.stride_tricks.sliding_window_view(emg_sample, self.window_length, axis=0)[
                  ::self.stride].copy()
        raw_y = self.key_labels[emg_start_idx:emg_start_idx + self.interval_length].flatten()

        if self.return_raw:
            return torch.from_numpy(X).float(), torch.LongTensor(raw_y)
        else:
            y = remove_adj_duplicates(raw_y)
            return torch.from_numpy(X).float(), y
