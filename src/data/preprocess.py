from typing import Union, Any

import numpy as np
from numpy import ndarray

from emg_decoder.src.data.utils import load_experiment, bipolar_conversion
from emg_decoder.src.data.filter import Filter


def generate_labels(key_log: dict,
                    num_timesteps: int,
                    downsample_factor: int,
                    blank_label: Union[int, str] = 0,
                    require_ascii: bool = False,
                    software: str = 'raspy',
                    binary: bool = False) -> (np.ndarray, list):
    """
    Generate labels for EMG data based on key log.
    :param key_log: Dictionary containing key log data, including keybuffer2 and emg_step.
    :param num_timesteps: Number of timesteps in EMG data, equivalent to emg_data.shape[0].
    :param downsample_factor: Downsample factor applied EMG data.
    :param blank_label: Which index to use for blank label, either an int or 'last', which sets it equal to num_keys
    (the last index in Python indexing). 0 by default.
    :param require_ascii: Whether to require that the key pressed is an ASCII character, or allow any key.
    :param software: Which software was used to generate the key log, either 'raspy' or 'rumi'.
    :param binary: Whether to use binary labels, where any key is either pressed or not pressed, or multiclass labels
    :return: Array of length emg_data_length containing labels for each timestep, where each label is the index
    corresponding to the sorted ASCII code of the key from press to release, with the blank label when there is no key
    being pressed.
    """
    if software == 'raspy':
        key_buffer = key_log['keybuffer2']
        key_emg_idxs = np.expand_dims(key_log['emg_step'] // downsample_factor, -1)
    elif software == 'rumi':
        key_buffer = key_log['key_stream']
        key_emg_idxs = np.expand_dims(key_log['emg_stream_step'] // downsample_factor, -1)
    else:
        raise ValueError('Software must be either "raspy" or "rumi"')

    key_buffer = np.hstack([key_buffer, key_emg_idxs])

    keypress_ascii = key_buffer[:, 2]
    key_list = np.sort(np.unique(keypress_ascii))

    if require_ascii:
        key_list = key_list[key_list < 128]

    key_list = key_list.astype(int).tolist()

    print("Key counts:", [chr(let) for let in key_list])

    if type(blank_label) == int:
        labels = np.full(num_timesteps, blank_label)
    elif blank_label == 'last':
        labels = np.full(num_timesteps, len(key_list))
    else:
        raise ValueError('blank_label must be an int or the string "last"')

    for idx, key in enumerate(key_list):
        specific_key_idxs = np.nonzero(keypress_ascii == key)[0]
        key_log_for_specific_key = key_buffer[specific_key_idxs, :]

        # Getting the start and end EMG indices where start is when key is pressed and end is when key is released
        key_log_spec_key_starts = key_log_for_specific_key[np.nonzero(key_log_for_specific_key[:, 0] == 0)[0], -1]
        # key_log_spec_key_ends = key_log_for_specific_key[np.nonzero(key_log_for_specific_key[:, 0] == 1)[0], -1]
        print("char:", chr(int(key)), "num:", key_log_spec_key_starts.shape[0])
        for i in range(key_log_spec_key_starts.shape[0]):
            if blank_label == 0:
                if binary:
                    labels[int(key_log_spec_key_starts[i])] = 1
                else:
                    labels[int(key_log_spec_key_starts[i])] = idx + 1
            elif blank_label == 'last':
                if binary:
                    labels[int(key_log_spec_key_starts[i])] = 0
                else:
                    labels[int(key_log_spec_key_starts[i])] = idx

    return labels, key_list


def generate_window_idxs(num_timesteps: int,
                         window_len: int,
                         window_stride: int,
                         sampling_freq: int,
                         downsample_factor: int) -> np.ndarray:
    """
    Generate indices for windowing EMG data, where each window is an input sample to the decoder.
    :param num_timesteps: Number of timesteps in EMG data, equivalent to emg_data.shape[0].
    :param window_len: Length of window in seconds.
    :param window_stride: Stride of window in seconds.
    :param sampling_freq: Sampling frequency of EMG data.
    :param downsample_factor: Downsample factor applied to EMG data.
    :return: Array of start indices for windows of EMG data.
    """
    effective_sampling_freq = sampling_freq / downsample_factor
    window_len = int(effective_sampling_freq * window_len)
    window_idxs = []
    for i in range(0, num_timesteps - window_len, int(effective_sampling_freq * window_stride)):
        window_idxs.append(i)

    return np.array(window_idxs)


def preprocess(config: dict, software: str = 'raspy') -> tuple[Any, ndarray, ndarray, ndarray | Any]:
    """
    Select channels, filter, and downsample EMG data and generate corresponding key labels and window indices.
    :param config: Configuration dictionary containing all preprocessing params, metadata, data hyperparams.
    :param software: The type of software used to record, should be either "raspy" or "rumi."
    :return: EMG data, key labels, and window indices for samples.
    """
    meta_config = config['metadata']
    pre_config = config['preprocessing']
    data_config = config['data']
    model_config = config['model']
    data_dir = pre_config['raw_data_dir']
    if software == 'raspy':
        data_dir += "/raspydata"
    elif software == 'rumi':
        data_dir += "/data_streams"
    start_channel = meta_config['start_channel']
    end_channel = meta_config['end_channel']

    if 'dead_channels' in meta_config:
        dead_channels = meta_config['dead_channels']
    else:
        dead_channels = None

    emg_data, key_log = load_experiment(data_dir, start_channel, end_channel,
                                        dead_channels=dead_channels,
                                        software=software)

    if meta_config['configuration'] == 'bipolar':
        if meta_config['append']:
            emg_data = np.concatenate((emg_data, bipolar_conversion(emg_data, meta_config['offset'])), axis=1)
        else:
            emg_data = bipolar_conversion(emg_data, meta_config['offset'])

    fn = pre_config['notch_filters']
    qs = pre_config['quality_factors']
    fc = (pre_config['bandpass_lower_bound'], pre_config['bandpass_upper_bound'])
    fs = meta_config['sampling_frequency']
    downsample_factor = pre_config['downsample_factor']

    emg_filter = Filter(fn, qs, fc=fc, butterworth_type='bandpass', order=4, sampling_freq=fs)
    emg_data = emg_filter.filter_data(emg_data)
    emg_data = emg_data[::downsample_factor]

    if meta_config['configuration'] == 'bipolar':
        pass
    elif pre_config['reference_mode'] == 'average':
        emg_data -= np.mean(emg_data, axis=1, keepdims=True)
    elif pre_config['reference_mode'] == 'common':
        pass
    else:
        raise NotImplementedError

    num_timesteps = emg_data.shape[0]
    blank_label = pre_config['blank_label']

    labels, key_list = generate_labels(key_log, num_timesteps, downsample_factor,
                                       blank_label=blank_label,
                                       require_ascii=pre_config['require_ascii'],
                                       binary=pre_config['binary_labels'],
                                       software=software)

    if model_config['type'] == 'EMGNet':
        interval_len = data_config['interval_len']
        interval_stride = data_config['interval_stride']
        windows = generate_window_idxs(num_timesteps,
                                       interval_len,
                                       interval_stride,
                                       fs,
                                       downsample_factor)
    elif model_config['type'] == 'TemporalNet':
        window_len = data_config['window_len']
        window_stride = data_config['window_stride']
        windows = generate_window_idxs(num_timesteps,
                                       window_len,
                                       window_stride,
                                       fs,
                                       downsample_factor)
    else:
        raise NotImplementedError(f"Model type {model_config['type']} not implemented.")

    return emg_data, labels, windows, key_list
