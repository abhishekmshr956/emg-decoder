import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.filter import Filter
from src.data.utils import bipolar_conversion, load_data, average_reference


def remove_overlaps(key_data: np.ndarray, window_pre: int, window_post: int) -> np.ndarray:
    """
    Remove keypresses that overlap with other keypresses.
    :param key_data: Structured Numpy array containing keypress data.
    :param window_pre: Number of samples to include before the keypress.
    :param window_post: Number of samples to include after the keypress.
    :return: Key data with overlapping keypresses removed.
    """
    starting_num = key_data.size

    key_times = key_data['emg_stream_step']
    press_intervals = key_times[1:] - key_times[:-1]
    overlap_idxs = np.argwhere(press_intervals < window_pre + window_post).flatten()
    rm_idxs = np.concatenate((overlap_idxs, overlap_idxs + 1))
    clean_key_data = np.delete(key_data, rm_idxs)

    print(f"Removed {starting_num - clean_key_data.size} out of {starting_num} keypresses.")
    return clean_key_data


def select_keys(key_data: np.ndarray, layout: list) -> np.ndarray:
    """
    Select only the keypresses in a given layout from the key data.
    :param key_data: Structured Numpy array containing keypress data.
    :param layout: Nested list containing the chars in the layout of the keyboard.
    :return: Key data with all keypresses not in the layout removed.
    """
    if isinstance(layout[0], list):
        layout = [item for sublist in layout for item in sublist]

    key_conditions = []
    for key in layout:
        if type(key) == str:
            key_code = ord(key)
        elif type(key) == int:
            key_code = key
        else:
            raise ValueError
        key_conditions.append(key_data['keybuffer'][:, 1] == key_code)

    clean_bools = key_conditions[0]
    for i in range(1, len(key_conditions)):
        clean_bools = np.logical_or(clean_bools, key_conditions[i])

    return key_data[clean_bools]


def get_blank_idxs(key_data: np.ndarray, window_pre: int, window_post: int, est_duration: int = None) -> np.ndarray:
    """
    Get nonoverlapping indices for windows of EMG data where no events occur.
    :param key_data: Original unprocessed key data, since we do not want removed keypresses in our blank windows (or
    releases).
    :param window_pre: Number of samples to include before the blank index.
    :param window_post: Number of samples to include after the blank index.
    :param est_duration: Estimated duration of event if no ground truth key release is available, i.e. in rhythm game
    :return: Array of indices of blank EMG data.
    """
    emg_steps = key_data['emg_stream_step']

    blank_idxs = []
    for step_idx, step in enumerate(emg_steps[:-1]):
        blank_idx = int(step + window_post + window_pre)
        if est_duration:
            blank_idx += est_duration
        while blank_idx < emg_steps[step_idx + 1] - (window_pre + window_post):
            blank_idxs.append(blank_idx)
            blank_idx += (window_post + window_pre)
    print(len(blank_idxs))
    return np.array(blank_idxs)


def create_windows(emg_stream: np.ndarray, emg_stream_step_idxs: np.ndarray, window_pre: int, window_post: int,
                   dtype=np.float64) -> np.ndarray:
    """
    Create windows of EMG data around the listed indices.
    :param emg_stream: EMG data stream of shape (num_samples, num_channels).
    :param emg_stream_step_idxs: List of emg_stream indices to create windows around.
    :param window_pre: Number of samples to include before the keypress.
    :param window_post: Number of samples to include after the keypress.
    :param dtype: Data type of the returned array.
    :return: Array of shape (num_windows, window_pre + window_post, num_channels) containing the EMG windows.
    """
    num_chans = emg_stream.shape[-1]
    key_windows = np.zeros((emg_stream_step_idxs.size, window_pre + window_post, num_chans), dtype=dtype)

    breakpoint()

    for i, step in enumerate(emg_stream_step_idxs):
        #print(i)
        key_windows[i] = emg_stream[int(step - window_pre):int(step + window_post), :]
    return key_windows


def augment_data(window: torch.tensor, subwindow_len: int, num_views: int, var_factor: int = 3, add_noise: bool = True,
                 time_shift: bool = True):
    window_len = window.size(dim=0)
    num_chans = window.size(dim=1)
    if not time_shift:
        start_idx = (window_len - subwindow_len) // 2
        window = window[start_idx:start_idx + subwindow_len, :]
        emg_windows = torch.unsqueeze(window, 0).repeat(num_views, 1, 1)
    elif window_len - subwindow_len < num_views:
        stride = 1
        emg_windows = window.unfold(0, subwindow_len, stride).swapaxes(1, 2)
        num_duplicates = int(np.ceil(num_views / (window_len - subwindow_len)))
        emg_windows = emg_windows.repeat(num_duplicates, 1, 1)[:num_views]
    else:
        stride = int((window_len - subwindow_len) // num_views)
        emg_windows = window.unfold(0, subwindow_len, stride).swapaxes(1, 2)[:num_views]  # TODO: Check this

    if not add_noise:
        return emg_windows
    else:
        channel_variances = torch.var(window, dim=0)
        white_noise = torch.randn((num_views, subwindow_len, num_chans)
                                  ) * torch.sqrt(channel_variances) / var_factor
        white_noise[num_views // 2, ...].fill_(
            0)  # The center view is the original window, no noise added. # TODO: Check this
        return emg_windows + white_noise


class KeyDataset(Dataset):
    def __init__(self, emg_stream: np.ndarray, key_data: np.ndarray, layout: list, window_pre: int, window_post: int,
                 subwindow_len: int,
                 start_channel: int = None,
                 end_channel: int = None,
                 bipolar: bool = True,
                 avg_ref: bool = True,
                 include_blanks: bool = True,
                 limit_blanks: bool = False,
                 est_duration: int = None,
                 binary: bool = False,
                 augment: bool = False,
                 filter_params: dict = None,
                 init: bool = True):
        """
        Dataset for creating windows of EMG data around keypresses + blanks.
        :param emg_stream: EMG data stream of shape (num_samples, num_channels).
        :param key_data: Key data of shape (num_keypresses, 5) containing keypress data.
        :param layout: Keyboard layout as a nested list.
        :param window_pre: Number of samples to include before the keypress.
        :param window_post: Number of samples to include after the keypress.
        :param subwindow_len: Length of the sliding subwindows across the original data (this will be the actual window
        length used for training).
        :param start_channel: Index of the first channel to include in the dataset.
        :param end_channel: Index of the last channel to include in the dataset.
        :param bipolar: Whether to convert the EMG data to a bipolar reference.
        :param binary: Whether to use binary labels (keypress vs. blank).
        :param augment: Whether to augment the data by adding noise.
        :param filter_params: Dictionary of filter parameters, containing details of notch and bandpass filters.
        :param init: Whether to conduct all the preprocessing in the constructor (unnecessary for dataset addition).
        """

        self.binary = binary
        self.augment = augment
        self.subwindow_len = subwindow_len

        if not init:
            self.X = None
            self.y = None
            self.num_chans = None
            self.full_windows = None
            return

        if filter_params is None:
            filter_params = {}
        fn = filter_params.get('fn', [60, 120, 180, 60, 300, 240])
        qs = filter_params.get('qs', [10, 5, 2, 4, 2, 2])
        fc = filter_params.get('fc', (10, 1000))
        fs = filter_params.get('fs', 4000)
        filt = Filter(fn, qs, fc=fc, butterworth_type='bandpass', order=4, sampling_freq=fs)

        if start_channel is None:
            start_channel = 0
        if end_channel is None:
            end_channel = emg_stream.shape[-1]

        emg_buffer = emg_stream[..., start_channel:end_channel]

        if avg_ref:
            emg_buffer = average_reference(emg_buffer)

        if bipolar:
            bp_emg_stream = bipolar_conversion(emg_buffer, offset=2, grid_type=["8-8-L-2"])
            filtered_emg_stream = filt.filter_data(bp_emg_stream)
        else:
            filtered_emg_stream = filt.filter_data(emg_buffer)

        self.num_chans = filtered_emg_stream.shape[-1]


        key_start_data = key_data[key_data['keybuffer'][:, 0] != 0]

        for i in range(0, key_start_data['keybuffer'].shape[0]):
            if(i%2 == 1):
                key_start_data['keybuffer'][i][0] = 0

        key_start_data = key_start_data[key_start_data['keybuffer'][:, 0] != 0]     

        #print("key_start_data", key_start_data)

        '''
        classes = [[], [], [], []]

        for i in range(0, key_start_data.shape[0]):
            classes[key_start_data['keybuffer'][i, 1]].append(i)

        for i in range(0, 4):
            perm = np.random.permutation(len(classes[i]))
            z = np.array(classes[i])
            z = z[perm]
            classes[i] = z

        idxs = [0, 0, 0, 0]
        counter = 0
        tot = 0
        perm = []

        while(tot < key_start_data.shape[0]):
            if(idxs[counter] >= len(classes[counter])):
                counter = (counter + 1)%4
                continue
            else:
                perm.append(classes[counter][idxs[counter]])
                idxs[counter] += 1
                counter = (counter + 1)%4
                tot += 1
        
        key_start_data = key_start_data[perm]
        '''

        #nonoverlapping_key_data = remove_overlaps(key_start_data, window_pre, window_post)
        selected_key_data1 = select_keys(key_start_data, layout)

        to_ignore = 6000 # in emg steps (milliseconds * 4)
        stride = 500 # in emg steps 

        # for i in range(0, selected_key_data1.shape[0]):
        #    if(i%2 == 0):
        #        selected_key_data1['emg_stream_step'][i] += to_ignore + window_pre
        #    else:
        #        selected_key_data1['emg_stream_step'][i] -= (to_ignore + window_post)
        # selected_key_data = selected_key_data1
        selected_key_data1['emg_stream_step'] += to_ignore
        selected_key_data = selected_key_data1

        print("selected key data", selected_key_data)

        idx = 0
        '''
        for i in range(0, selected_key_data1.shape[0]):
            idx += 1
            start_time = selected_key_data1['emg_stream_step'][i]
            end_time = start_time + 16000
            for j in range(start_time + stride, end_time, stride):
                #print(idx)
                #breakpoint()
                selected_key_data = np.insert(selected_key_data, idx, selected_key_data1[i])
                selected_key_data['emg_stream_step'][idx] = j
                idx += 1
            #breakpoint()
        '''
        



        key_labels = selected_key_data['keybuffer'][..., 1]
        breakpoint()
        key_windows = create_windows(filtered_emg_stream, selected_key_data['emg_stream_step'], window_pre,
                                     window_post)

        if binary:
            if not include_blanks:
                raise ValueError("Cannot use binary labels without including blanks!")
            self.y = key_labels != 0
        else:
            self.y = key_labels

        if include_blanks:
            blank_idxs = get_blank_idxs(key_data, window_pre, window_post, est_duration=est_duration)
            if limit_blanks:
                np.random.shuffle(blank_idxs)
                classes, counts = np.unique(key_labels, return_counts=True)
                max_count = np.max(counts)
                blank_idxs = blank_idxs[:max_count]
            blank_windows = create_windows(filtered_emg_stream, blank_idxs, window_pre, window_post)
            blank_labels = np.zeros(blank_idxs.size)
            self.full_windows = np.concatenate((key_windows, blank_windows))
            self.y = np.concatenate((self.y, blank_labels))
        else:
            self.full_windows = key_windows

        start_idx = (self.full_windows.shape[1] - subwindow_len) // 2
        self.X = self.full_windows[:, start_idx:start_idx + subwindow_len, :]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.binary:
            return self.X[idx], self.y[idx] != 0
        return self.X[idx], self.y[idx]

    def __add__(self, other):
        if not isinstance(other, KeyDataset):
            raise TypeError("Can only concatenate KeyDatasets with other KeyDatasets!")
        if other.binary != self.binary:
            raise ValueError("Only one KeyDataset is set to binary labels!")
        if other.augment != self.augment:
            raise ValueError("Only one KeyDataset is set to augment data!")

        dataset_sum = KeyDataset(None, None, None, None, None, self.subwindow_len, init=False, binary=self.binary)
        dataset_sum.full_windows = np.concatenate((self.full_windows, other.full_windows))
        dataset_sum.X = np.concatenate((self.X, other.X))
        dataset_sum.y = np.concatenate((self.y, other.y))
        dataset_sum.num_chans = self.num_chans
        return dataset_sum

    def __radd__(self, other):
        if not isinstance(other, KeyDataset):
            raise TypeError("Can only concatenate KeyDatasets with other KeyDatasets!")
        if other.binary != self.binary:
            raise ValueError("Only one KeyDataset is set to binary labels!")
        if other.augment != self.augment:
            raise ValueError("Only one KeyDataset is set to augment data!")

        dataset_sum = KeyDataset(None, None, None, None, None, self.subwindow_len, init=False, binary=self.binary)
        dataset_sum.full_windows = np.concatenate((self.full_windows, other.full_windows))
        dataset_sum.X = np.concatenate((self.X, other.X))
        dataset_sum.y = np.concatenate((self.y, other.y))
        dataset_sum.num_chans = self.num_chans
        return dataset_sum

    def augment_class(self, class_label: int, target_num_samples: int = None, num_views: int = None,
                      var_factor: int = 3, time_shift: bool = True, add_noise: bool = True, in_place: bool = False):
        """
        Augment a class by adding noise to the data. Note that this will not remove original samples.
        :param class_label: Label of the class to augment.
        :param target_num_samples: Number of samples to have after augmentation.
        :param num_views: Number of views to take from each sample.
        :param var_factor: Scale of white noise to apply during augmentation.
        :param time_shift: Whether to shift the data in time during augmentation.
        :param add_noise: Whether to add white noise during augmentation.
        :param in_place: Whether to directly alter the instance data or return a new dataset for use with balancing.
        :return:
        """
        class_idxs = np.argwhere(self.y == class_label)

        num_samples = class_idxs.size

        if target_num_samples is not None:
            if num_samples > target_num_samples:
                print(f"Already have {num_samples}!")
                return self.X, self.y

        if num_views is not None and target_num_samples is not None:
            raise ValueError("Choose one of num_views or target_num_samples to set!")
        if num_views is None and target_num_samples is None:
            raise ValueError("Choose one of num_views or target_num_samples to set!")
        if target_num_samples is not None:
            num_views = int(np.ceil(target_num_samples / num_samples))
        else:
            target_num_samples = num_views * num_samples

        num_chans = self.full_windows.shape[-1]
        augmented_X = np.zeros((num_views * num_samples, self.subwindow_len, num_chans), dtype=np.float64)
        for i, sample in enumerate(self.full_windows[class_idxs]):
            augmented_X[i * num_views:(i + 1) * num_views] = augment_data(torch.squeeze(torch.tensor(sample)),
                                                                          self.subwindow_len,
                                                                          num_views,
                                                                          var_factor=var_factor,
                                                                          add_noise=add_noise,
                                                                          time_shift=time_shift)
        augmented_y = np.ones(num_views * num_samples) * class_label
        # Shuffle before truncating
        shuffle_idxs = np.random.permutation(augmented_y.size)
        augmented_X = augmented_X[shuffle_idxs]
        augmented_y = augmented_y[shuffle_idxs]
        if in_place:
            self.X = np.concatenate((self.X, augmented_X[:target_num_samples]))
            self.y = np.concatenate((self.y, augmented_y[:target_num_samples]))
        else:
            return augmented_X[:target_num_samples], augmented_y[:target_num_samples]



    def balance_classes(self, var_factor: int = 3, time_shift: bool = True, add_noise: bool = True, max_samples: int = None):
        classes, counts = np.unique(self.y, return_counts=True)
        max_count = np.max(counts)
        if max_samples is not None:
            max_count = max_samples
        augmented_data = []
        augmented_labels = []
        for class_label in classes:
            aug_X, aug_y = self.augment_class(class_label, target_num_samples=max_count, var_factor=var_factor,
                                              time_shift=time_shift, add_noise=add_noise)
            augmented_data.append(aug_X)
            augmented_labels.append(aug_y)
        self.X = np.concatenate(augmented_data)
        self.y = np.concatenate(augmented_labels)

    def print_counts(self):
        codes, count = np.unique(self.y, return_counts=True)
        for i, code in enumerate(codes):
            if code == 0:
                char = "NULL"
            else:
                char = chr(int(code))
            print(f"{code}, {char}: {count[i]}")

    def to_numpy(self, include_full_windows: bool = True):
        if include_full_windows:
            dtype = [('full_windows', self.full_windows.dtype, self.full_windows.shape[1:]),
                     ('emg_windows', self.X.dtype, self.X.shape[1:]),
                     ('key_label', self.y.dtype)]
        else:
            dtype = [('emg_windows', self.X.dtype, self.X.shape[1:]),
                     ('key_label', self.y.dtype)]

        data = np.zeros(self.X.shape[0], dtype=dtype)

        if include_full_windows:
            data['full_windows'] = self.full_windows
        data['emg_windows'] = self.X
        data['key_label'] = self.y
        return data


if __name__ == '__main__':
    left_layout = [['q', 'w', 'e', 'r', 't'],
                   ['a', 's', 'd', 'f', 'g'],
                   ['z', 'x', 'c', 'v', 'b']]
    right_layout = [['y', 'u', 'i', 'o', 'p'],
                    ['h', 'j', 'k', 'l'],
                    ['n', 'm']]

    layout = [0, 1, 2, 3]
    window_pre = 0
    window_post = 4000
    subwindow_len = 240
    left_start_chan = 1
    right_start_chan = 65

    #data_dir = "/Users/johnzhou/research/rumi/data/"
    #expt_names = ["John-Zhou_2023-07-17-1628_Open-Loop-Typing-Task",
    #              "John-Zhou_2023-07-17-1702_Open-Loop-Typing-Task"]

    left_datasets = []
    right_datasets = []

    data_dir = "/home/john/rumi/data/"
    expt_names = ["Shreyas-Kaasyap_2023-08-24-1437_Open-Loop-Typing-Task"]

    datasets = []

    for expt_name in expt_names:
        key_fname = data_dir + expt_name + "/data_streams/key_stream.bin"
        emg_fname = data_dir + expt_name + "/data_streams/emg_stream.bin"

        key_data = load_data(key_fname)
        emg_data = load_data(emg_fname)

        emg_stream = emg_data['emgbuffer']

        dataset = KeyDataset(emg_stream, key_data, layout, window_pre, window_post, subwindow_len,
                             start_channel=1, 
                             end_channel=65, bipolar=True, avg_ref=False, binary=False, 
                             include_blanks=False, 
                             limit_blanks=False,
                             est_duration=400,
                             filter_params=None)
        datasets.append(dataset)

    print(datasets[0])

import numpy as np
import torch
from torch.utils.data import Dataset

from emg_decoder.src.data.filter import Filter
from emg_decoder.src.data.utils import bipolar_conversion, load_data, average_reference


def remove_overlaps(key_data: np.ndarray, window_pre: int, window_post: int) -> np.ndarray:
    """
    Remove keypresses that overlap with other keypresses.
    :param key_data: Structured Numpy array containing keypress data.
    :param window_pre: Number of samples to include before the keypress.
    :param window_post: Number of samples to include after the keypress.
    :return: Key data with overlapping keypresses removed.
    """
    starting_num = key_data.size

    key_times = key_data['emg_stream_step']
    press_intervals = key_times[1:] - key_times[:-1]
    overlap_idxs = np.argwhere(press_intervals < window_pre + window_post).flatten()
    rm_idxs = np.concatenate((overlap_idxs, overlap_idxs + 1))
    clean_key_data = np.delete(key_data, rm_idxs)

    print(f"Removed {starting_num - clean_key_data.size} out of {starting_num} keypresses.")
    return clean_key_data


def select_keys(key_data: np.ndarray, layout: list) -> np.ndarray:
    """
    Select only the keypresses in a given layout from the key data.
    :param key_data: Structured Numpy array containing keypress data.
    :param layout: Nested list containing the chars in the layout of the keyboard.
    :return: Key data with all keypresses not in the layout removed.
    """
    if isinstance(layout[0], list):
        layout = [item for sublist in layout for item in sublist]

    key_conditions = []
    for key in layout:
        if type(key) == str:
            key_code = ord(key)
        elif type(key) == int:
            key_code = key
        else:
            raise ValueError
        key_conditions.append(key_data['keybuffer'][:, 1] == key_code)

    clean_bools = key_conditions[0]
    for i in range(1, len(key_conditions)):
        clean_bools = np.logical_or(clean_bools, key_conditions[i])

    return key_data[clean_bools]


def get_blank_idxs(key_data: np.ndarray, window_pre: int, window_post: int, est_duration: int = None) -> np.ndarray:
    """
    Get nonoverlapping indices for windows of EMG data where no events occur.
    :param key_data: Original unprocessed key data, since we do not want removed keypresses in our blank windows (or
    releases).
    :param window_pre: Number of samples to include before the blank index.
    :param window_post: Number of samples to include after the blank index.
    :param est_duration: Estimated duration of event if no ground truth key release is available, i.e. in rhythm game
    :return: Array of indices of blank EMG data.
    """
    emg_steps = key_data['emg_stream_step']

    blank_idxs = []
    for step_idx, step in enumerate(emg_steps[:-1]):
        blank_idx = int(step + window_post + window_pre)
        if est_duration:
            blank_idx += est_duration
        while blank_idx > emg_steps[step_idx + 1] - (window_pre + window_post):
            blank_idxs.append(blank_idx)
            blank_idx += (window_post + window_pre)

    return np.array(blank_idxs)


def create_windows(emg_stream: np.ndarray, emg_stream_step_idxs: np.ndarray, window_pre: int, window_post: int,
                   dtype=np.float64) -> np.ndarray:
    """
    Create windows of EMG data around the listed indices.
    :param emg_stream: EMG data stream of shape (num_samples, num_channels).
    :param emg_stream_step_idxs: List of emg_stream indices to create windows around.
    :param window_pre: Number of samples to include before the keypress.
    :param window_post: Number of samples to include after the keypress.
    :param dtype: Data type of the returned array.
    :return: Array of shape (num_windows, window_pre + window_post, num_channels) containing the EMG windows.
    """
    num_chans = emg_stream.shape[-1]
    key_windows = np.zeros((emg_stream_step_idxs.size, window_pre + window_post, num_chans), dtype=dtype)

    for i, step in enumerate(emg_stream_step_idxs):
        key_windows[i] = emg_stream[int(step - window_pre):int(step + window_post), :]
    return key_windows


def augment_data(window: torch.tensor, subwindow_len: int, num_views: int, var_factor: int = 3, add_noise: bool = True,
                 time_shift: bool = True):
    window_len = window.size(dim=0)
    num_chans = window.size(dim=1)
    if not time_shift:
        start_idx = (window_len - subwindow_len) // 2
        window = window[start_idx:start_idx + subwindow_len, :]
        emg_windows = torch.unsqueeze(window, 0).repeat(num_views, 1, 1)
    elif window_len - subwindow_len < num_views:
        stride = 1
        emg_windows = window.unfold(0, subwindow_len, stride).swapaxes(1, 2)
        num_duplicates = int(np.ceil(num_views / (window_len - subwindow_len)))
        emg_windows = emg_windows.repeat(num_duplicates, 1, 1)[:num_views]
    else:
        stride = int((window_len - subwindow_len) // num_views)
        emg_windows = window.unfold(0, subwindow_len, stride).swapaxes(1, 2)[:num_views]  # TODO: Check this

    if not add_noise:
        return emg_windows
    else:
        channel_variances = torch.var(window, dim=0)
        white_noise = torch.randn((num_views, subwindow_len, num_chans)
                                  ) * torch.sqrt(channel_variances) / var_factor
        white_noise[num_views // 2, ...].fill_(
            0)  # The center view is the original window, no noise added. # TODO: Check this
        return emg_windows + white_noise


class KeyDataset(Dataset):
    def __init__(self, emg_stream: np.ndarray, key_data: np.ndarray, layout: list, window_pre: int, window_post: int,
                 subwindow_len: int,
                 start_channel: int = None,
                 end_channel: int = None,
                 bipolar: bool = True,
                 avg_ref: bool = True,
                 include_blanks: bool = True,
                 limit_blanks: bool = False,
                 binary: bool = False,
                 augment: bool = False,
                 filter_params: dict = None,
                 init: bool = True):
        """
        Dataset for creating windows of EMG data around keypresses + blanks.
        :param emg_stream: EMG data stream of shape (num_samples, num_channels).
        :param key_data: Key data of shape (num_keypresses, 5) containing keypress data.
        :param layout: Keyboard layout as a nested list.
        :param window_pre: Number of samples to include before the keypress.
        :param window_post: Number of samples to include after the keypress.
        :param subwindow_len: Length of the sliding subwindows across the original data (this will be the actual window
        length used for training).
        :param start_channel: Index of the first channel to include in the dataset.
        :param end_channel: Index of the last channel to include in the dataset.
        :param bipolar: Whether to convert the EMG data to a bipolar reference.
        :param binary: Whether to use binary labels (keypress vs. blank).
        :param augment: Whether to augment the data by adding noise.
        :param filter_params: Dictionary of filter parameters, containing details of notch and bandpass filters.
        :param init: Whether to conduct all the preprocessing in the constructor (unnecessary for dataset addition).
        """

        self.binary = binary
        self.augment = augment
        self.subwindow_len = subwindow_len

        if not init:
            self.X = None
            self.y = None
            self.num_chans = None
            self.full_windows = None
            return

        if filter_params is None:
            filter_params = {}
        fn = filter_params.get('fn', [60, 120, 180, 60, 300, 240])
        qs = filter_params.get('qs', [10, 5, 2, 4, 2, 2])
        fc = filter_params.get('fc', (10, 1000))
        fs = filter_params.get('fs', 4000)
        filt = Filter(fn, qs, fc=fc, butterworth_type='bandpass', order=4, sampling_freq=fs)

        if start_channel is None:
            start_channel = 0
        if end_channel is None:
            end_channel = emg_stream.shape[-1]

        emg_buffer = emg_stream[..., start_channel:end_channel]

        if avg_ref:
            emg_buffer = average_reference(emg_buffer)

        if bipolar:
            bp_emg_stream = bipolar_conversion(emg_buffer)
            filtered_emg_stream = filt.filter_data(bp_emg_stream)
        else:
            filtered_emg_stream = filt.filter_data(emg_buffer)

        self.num_chans = filtered_emg_stream.shape[-1]

        key_start_data = key_data[key_data['keybuffer'][:, 0] != 0]
        nonoverlapping_key_data = remove_overlaps(key_start_data, window_pre, window_post)
        selected_key_data = select_keys(nonoverlapping_key_data, layout)

        key_labels = selected_key_data['keybuffer'][..., 1]
        key_windows = create_windows(filtered_emg_stream, selected_key_data['emg_stream_step'], window_pre,
                                     window_post)

        if binary:
            if not include_blanks:
                raise ValueError("Cannot use binary labels without including blanks!")
            self.y = key_labels != 0
        else:
            self.y = key_labels

        if include_blanks:
            blank_idxs = get_blank_idxs(key_data, window_pre, window_post)
            if limit_blanks:
                np.random.shuffle(blank_idxs)
                classes, counts = np.unique(key_labels, return_counts=True)
                max_count = np.max(counts)
                blank_idxs = blank_idxs[:max_count]
            blank_windows = create_windows(filtered_emg_stream, blank_idxs, window_pre, window_post)
            blank_labels = np.zeros(blank_idxs.size)
            self.full_windows = np.concatenate((key_windows, blank_windows))
            self.y = np.concatenate((self.y, blank_labels))
        else:
            self.full_windows = key_windows

        start_idx = (self.full_windows.shape[1] - subwindow_len) // 2
        self.X = self.full_windows[:, start_idx:start_idx + subwindow_len, :]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.binary:
            return self.X[idx], self.y[idx] != 0
        return self.X[idx], self.y[idx]

    def __add__(self, other):
        if not isinstance(other, KeyDataset):
            raise TypeError("Can only concatenate KeyDatasets with other KeyDatasets!")
        if other.binary != self.binary:
            raise ValueError("Only one KeyDataset is set to binary labels!")
        if other.augment != self.augment:
            raise ValueError("Only one KeyDataset is set to augment data!")

        dataset_sum = KeyDataset(None, None, None, None, None, self.subwindow_len, init=False, binary=self.binary)
        dataset_sum.full_windows = np.concatenate((self.full_windows, other.full_windows))
        dataset_sum.X = np.concatenate((self.X, other.X))
        dataset_sum.y = np.concatenate((self.y, other.y))
        dataset_sum.num_chans = self.num_chans
        return dataset_sum

    def __radd__(self, other):
        if not isinstance(other, KeyDataset):
            raise TypeError("Can only concatenate KeyDatasets with other KeyDatasets!")
        if other.binary != self.binary:
            raise ValueError("Only one KeyDataset is set to binary labels!")
        if other.augment != self.augment:
            raise ValueError("Only one KeyDataset is set to augment data!")

        dataset_sum = KeyDataset(None, None, None, None, None, self.subwindow_len, init=False, binary=self.binary)
        dataset_sum.full_windows = np.concatenate((self.full_windows, other.full_windows))
        dataset_sum.X = np.concatenate((self.X, other.X))
        dataset_sum.y = np.concatenate((self.y, other.y))
        dataset_sum.num_chans = self.num_chans
        return dataset_sum

    def augment_class(self, class_label: int, target_num_samples: int = None, num_views: int = None,
                      var_factor: int = 3, time_shift: bool = True, add_noise: bool = True, in_place: bool = False):
        """
        Augment a class by adding noise to the data. Note that this will not remove original samples.
        :param class_label: Label of the class to augment.
        :param target_num_samples: Number of samples to have after augmentation.
        :param num_views: Number of views to take from each sample.
        :param var_factor: Scale of white noise to apply during augmentation.
        :param time_shift: Whether to shift the data in time during augmentation.
        :param add_noise: Whether to add white noise during augmentation.
        :param in_place: Whether to directly alter the instance data or return a new dataset for use with balancing.
        :return:
        """
        class_idxs = np.argwhere(self.y == class_label)

        num_samples = class_idxs.size

        if target_num_samples is not None:
            if num_samples > target_num_samples:
                print(f"Already have {num_samples}!")
                return self.X, self.y

        if num_views is not None and target_num_samples is not None:
            raise ValueError("Choose one of num_views or target_num_samples to set!")
        if num_views is None and target_num_samples is None:
            raise ValueError("Choose one of num_views or target_num_samples to set!")
        if target_num_samples is not None:
            num_views = int(np.ceil(target_num_samples / num_samples))
        else:
            target_num_samples = num_views * num_samples

        num_chans = self.full_windows.shape[-1]
        augmented_X = np.zeros((num_views * num_samples, self.subwindow_len, num_chans), dtype=np.float64)
        for i, sample in enumerate(self.full_windows[class_idxs]):
            augmented_X[i * num_views:(i + 1) * num_views] = augment_data(torch.squeeze(torch.tensor(sample)),
                                                                          self.subwindow_len,
                                                                          num_views,
                                                                          var_factor=var_factor,
                                                                          add_noise=add_noise,
                                                                          time_shift=time_shift)
        augmented_y = np.ones(num_views * num_samples) * class_label
        # Shuffle before truncating
        shuffle_idxs = np.random.permutation(augmented_y.size)
        augmented_X = augmented_X[shuffle_idxs]
        augmented_y = augmented_y[shuffle_idxs]
        if in_place:
            self.X = np.concatenate((self.X, augmented_X[:target_num_samples]))
            self.y = np.concatenate((self.y, augmented_y[:target_num_samples]))
        else:
            return augmented_X[:target_num_samples], augmented_y[:target_num_samples]

    def balance_classes(self, var_factor: int = 3, time_shift: bool = True, add_noise: bool = True, max_samples: int = None):
        classes, counts = np.unique(self.y, return_counts=True)
        max_count = np.max(counts)
        if max_samples is not None:
            max_count = max_samples
        augmented_data = []
        augmented_labels = []
        for class_label in classes:
            aug_X, aug_y = self.augment_class(class_label, target_num_samples=max_count, var_factor=var_factor,
                                              time_shift=time_shift, add_noise=add_noise)
            augmented_data.append(aug_X)
            augmented_labels.append(aug_y)
        self.X = np.concatenate(augmented_data)
        self.y = np.concatenate(augmented_labels)

    def print_counts(self):
        codes, count = np.unique(self.y, return_counts=True)
        for i, code in enumerate(codes):
            if code == 0:
                char = "NULL"
            else:
                char = chr(int(code))
            print(f"{code}, {char}: {count[i]}")

    def to_numpy(self, include_full_windows: bool = True):
        if include_full_windows:
            dtype = [('full_windows', self.full_windows.dtype, self.full_windows.shape[1:]),
                     ('emg_windows', self.X.dtype, self.X.shape[1:]),
                     ('key_label', self.y.dtype)]
        else:
            dtype = [('emg_windows', self.X.dtype, self.X.shape[1:]),
                     ('key_label', self.y.dtype)]

        data = np.zeros(self.X.shape[0], dtype=dtype)

        if include_full_windows:
            data['full_windows'] = self.full_windows
        data['emg_windows'] = self.X
        data['key_label'] = self.y
        return data


if __name__ == '__main__':
    left_layout = [['q', 'w', 'e', 'r', 't'],
                   ['a', 's', 'd', 'f', 'g'],
                   ['z', 'x', 'c', 'v', 'b']]
    right_layout = [['y', 'u', 'i', 'o', 'p'],
                    ['h', 'j', 'k', 'l'],
                    ['n', 'm']]
    window_pre = 160
    window_post = 160
    subwindow_len = 240
    left_start_chan = 33
    right_start_chan = 1

    data_dir = "/Users/johnzhou/research/rumi/data/"
    expt_names = ["John-Zhou_2023-07-17-1628_Open-Loop-Typing-Task",
                  "John-Zhou_2023-07-17-1702_Open-Loop-Typing-Task"]

    left_datasets = []
    right_datasets = []

    for expt_name in expt_names:
        key_fname = data_dir + expt_name + "/data_streams/key_stream.bin"
        emg_fname = data_dir + expt_name + "/data_streams/emg_stream.bin"

        key_data = load_data(key_fname)
        emg_data = load_data(emg_fname)

        emg_stream = emg_data['emg_stream']
        left_dataset = KeyDataset(emg_stream, key_data, left_layout, window_pre, window_post, subwindow_len,
                                  start_channel=left_start_chan,
                                  end_channel=left_start_chan + 32, bipolar=True, binary=True, filter_params=None)
        right_dataset = KeyDataset(emg_stream, key_data, right_layout, window_pre, window_post, subwindow_len,
                                   start_channel=right_start_chan,
                                   end_channel=right_start_chan + 32, bipolar=True, binary=True, filter_params=None)
        left_datasets.append(left_dataset)
        right_datasets.append(right_dataset)
