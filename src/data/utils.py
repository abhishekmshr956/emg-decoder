from typing import Union

import numpy as np
import ast


def load_data_deprecated(fname, encoding: str = 'utf-8'):
    """Read a structured Numpy array from a binary file in a cross-platform-compatible manner."""
    print("This function is used to load data of a deprecated format.")
    with open(fname, mode='rb') as f:
        header_len = int.from_bytes(f.read(2), 'little')
        stream_name, dtype = f.read(header_len).decode(encoding).split('\n')[:-1]
        data = np.fromfile(f, dtype=ast.literal_eval(dtype))
    return data


def load_data(filename: str, return_dict: bool = False, encoding: str = 'utf-8') -> np.ndarray or dict:
    """
    Legacy binary file data loading function from Raspy. Rumi integrates shapes and dtypes into the same line while
    maintaining compatibility.
    :param filename: Filename of the binary file to load.
    :param return_dict: Whether to return a dictionary or a structured Numpy array.
    :param encoding: Encoding to decode the header of the binary file.
    :return: Dictionary or numpy array of the data.
    """
    with open(filename, 'rb') as openfile:
        name = openfile.readline().decode(encoding).strip()
        labels = openfile.readline().decode(encoding).strip()
        dtypes = openfile.readline().decode(encoding).strip()
        shapes = None
        # shapes can be indicated with a $ to separate.
        if len(dtypes.split('$')) == 2:
            dtypes, shapes = dtypes.split('$')
            dtypes = dtypes.strip()
            shapes = ast.literal_eval(shapes.strip())

        labels = labels.split(',')
        dtypes = dtypes.split(',')
        if shapes is None:
            data = np.fromfile(openfile, dtype=list(zip(labels, dtypes)))
        else:
            data = np.fromfile(openfile, dtype=list(zip(labels, dtypes, shapes)))
        if not return_dict:
            return data
        data_dict = {label: data[label] for label in labels}
        data_dict['name'] = name
        data_dict['labels'] = labels
        data_dict['dtypes'] = dtypes
    return data_dict


def load_experiment(data_dir: str,
                    start_channel: int,
                    end_channel: int,
                    dead_channels: list = None,
                    software: str = 'raspy') -> tuple[np.ndarray, dict]:
    """
    Load an experiment from a directory.
    :param data_dir: Directory containing the experiment data.
    :param start_channel: Index of first channel used to record from.
    :param end_channel: Index of last channel used to record from.
    :param dead_channels: List of any dead channels to remove.
    :param software: Software used to record the experiment.
    :return: EMG data and key log.
    """
    if software == 'raspy':
        emg_data = load_data(f"{data_dir}/emg.bin")['emgbuffersignal'][:, start_channel:end_channel]
        key_log = load_data(f"{data_dir}/keylog.bin")
    elif software == 'rumi':
        emg_data = load_data(f"{data_dir}/emg_stream.bin")['emg_stream'][:, start_channel:end_channel]
        key_log = load_data(f"{data_dir}/key_stream.bin")
    else:
        raise ValueError(f"Unknown software: {software}")

    if dead_channels:
        print(f"Removing dead channels: {dead_channels}")
        dead_channel_idxs = [chan_idx - start_channel for chan_idx in dead_channels]
        emg_data = np.delete(emg_data, dead_channel_idxs, axis=1)

    return emg_data, key_log


def remove_adj_duplicates(target, blank: int = 0):
    """Remove adjacent duplicates from a numpy array.
    :param target: a numpy array
    :param blank: the value to treat as a blank
    :return: a numpy array with no adjacent duplicates and no blank characters
    """
    no_dup = target[np.insert(np.diff(target).astype(bool), 0, True)]
    return no_dup[no_dup != blank]


def channel_conversion_matrix(data: np.ndarray, grid_type) -> np.ndarray:
    """Generate a conversion matrix for converting between 32 unipolar channel numbers and position/naming on the
    textile HD-EMG grid. After conversion, column indices are in ascending order starting from 0 at the non-connector
    side of the grid, and electrodes in the same column are separated by num_rows units.
    :param data: an array of shape (num_timesteps, 32) containing the data to generate the conversion
    :param grid_type: type of textile HD-EMG grid used, either '4-8-L' or '8-8-L'
    :return: a numpy array of shape (num_timesteps, 4, 8) containing the data in the new channel order
    """
    if grid_type == '4-8-L':
        r1 = list(range(0, 8)[::-1])
        r2 = [13, 14, 15, 12, 11, 10, 9, 8]
        r3 = [18, 17, 16, 19, 20, 21, 22, 23]
        r4 = list(range(24, 32))
        idxs = r1 + r2 + r3 + r4

    elif grid_type == '8-8-L-1':
        # No change
        r1 = [16, 15, 14, 13, 12, 8, 4, 0]
        r2 = [21, 20, 19, 18, 17, 9, 5, 1]
        r3 = [26, 25, 24, 23, 22, 10, 6, 2]
        r4 = [31, 30, 29, 28, 27, 11, 7, 3]
        idxs = r1 + r2 + r3 + r4

    elif grid_type == '8-8-L-2':
        r1 = [0, 1, 2, 3, 4, 20, 24, 28]
        r2 = [5, 6, 7, 8, 9, 21, 25, 29]
        r3 = [10, 11, 12, 13, 14, 22, 26, 30]
        r4 = [15, 16, 17, 18, 19, 23, 27, 31]
        idxs = r1 + r2 + r3 + r4
    elif grid_type == '6-11-L-1':
        # No change
        r1 = [21,20,19,18,17,14,11,8,5,2]
        r2 = [26,25,24,23,22,15,12,9,6,3,0]
        r3 = [31,30,29,28,27,16,13,10,7,4,1]
        idxs = r1 + r2 + r3 

    elif grid_type == '6-11-L-2':
        r1 = [0,1,2,3,4,15,18,21,24,27,30]
        r2 = [5,6,7,8,9,16,19,22,25,28,31]
        r3 = [10,11,12,13,14,17,20,23,26,29]
        idxs = r1 + r2 + r3
    else:
        raise NotImplementedError(f"Unknown grid type: {grid_type}")

    #idxs = r1 + r2 + r3 + r4
    # print("Mapping")
    # for i, j in enumerate([idx + 1 for idx in idxs]):
    #     print(f"R{i//8 + 1}C{i % 8 + 1}: UNI {j}")
    # print(data[:, idxs].shape)
    return np.reshape(data[:, idxs], (-1, 4, 8))


def average_reference(data: np.ndarray) -> np.ndarray:
    _, num_chans = data.shape
    if num_chans % 32 != 0:
        raise ValueError("Number of channels should be a multiple of 32 for HD-EMG grids!")
    num_grids = num_chans // 32

    for i in range(num_grids):
        data[:, i * 32:(i + 1) * 32] = (data[:, i * 32:(i + 1) * 32].T - np.mean(data[:, i * 32:(i + 1) * 32], axis=1)).T
    
    return data


def bipolar_conversion(data: np.ndarray, offset: int, grid_type: Union[str, list[str]]) -> np.ndarray:
    """
    Convert unipolar data to bipolar data.
    :param data: an array of shape (num_timesteps, 32*n) containing the data to convert
    :param offset: offset between bipolar channel pairs
    :param grid_type: type of textile HD-EMG grid used, either '4-8-L' or '8-8-L'
    :return: an array of shape (num_timesteps, <32*n) containing the converted bipolar data
    """
    num_timesteps, num_chans = data.shape
    if isinstance(grid_type, str):
        grid_type = [grid_type]
    if num_chans % 32 != 0:
        raise ValueError("Number of channels should be a multiple of 32 for HD-EMG grids!")
    num_grids = num_chans // 32
    if len(grid_type) != num_grids:
        raise ValueError(f"Number of grid types ({len(grid_type)}) must match number of grids ({num_grids})!")
    grids = []

    for i in range(num_grids):
        grid_data = channel_conversion_matrix(data[:, i * 32:(i + 1) * 32], grid_type[i])
        num_timesteps, num_uni_rows, num_cols = grid_data.shape
        num_bp_rows = int(num_uni_rows - offset)
        bp_data = np.zeros((num_timesteps, num_bp_rows, num_cols))
        for j in range(num_bp_rows):
            bp_data[:, j, :] = grid_data[:, j, :] - grid_data[:, j + offset, :]
        bp_data = np.reshape(bp_data, (num_timesteps, num_bp_rows * num_cols))
        grids.append(bp_data)
    return np.hstack(grids)


def key_counts(key_data):
    key_buffer = key_data['key_stream']
    num_total_keypresses = np.where(key_buffer[:, 0] == 0)[0].size
    print("Total keypresses", num_total_keypresses)
    keypress_ascii = key_buffer[:, 2]
    key_list = np.sort(np.unique(keypress_ascii))
    key_list = key_list.astype(int).tolist()
    print(f"{len(key_list)} unique keys:", [chr(let) for let in key_list])

    for idx, key in enumerate(key_list):
        specific_key_idxs = np.nonzero(keypress_ascii == key)[0]
        key_log_for_specific_key = key_buffer[specific_key_idxs, :]

        # Getting the start and end EMG indices where start is when key is pressed and end is when key is released
        key_log_spec_key_starts = key_log_for_specific_key[np.nonzero(key_log_for_specific_key[:, 0] == 0)[0], -1]
        char_code = chr(key)
        print(f"char: '{char_code}', ord: {key}, num: {key_log_spec_key_starts.shape[0]}")


if __name__ == '__main__':
    data = np.expand_dims(np.arange(32), 0)
    print(data)
    bipolar_conversion(data, offset=2, grid_type='8-8-L-1')
