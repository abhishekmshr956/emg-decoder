from typing import Union

import numpy as np
import scipy.signal


class Filter:
    def __init__(self,
                 notch_freqs: list,
                 quality_factors: list,
                 fc: Union[tuple, int] = None,
                 butterworth_type: str = 'bandpass',
                 order: int = 1,
                 sampling_freq: float = 1000.0) -> None:
        """
        Class to apply notch and butterworth filters to data.
        :param notch_freqs: List of notch filter frequencies
        :param quality_factors: List of notch filter quality factors, should be same length as notch_freqs
        :param fc: Frequency cutoffs for butterworth filter, if None, no butterworth filter is applied. Single int for
        lowpass or highpass, tuple for bandpass.
        :param butterworth_type: String, 'lowpass', 'highpass', or 'bandpass'.
        :param order: Int, order of butterworth filter, or orders if separate highpass (first) and lowpass (second)
        :param sampling_freq: Sampling frequency of data
        """
        self.fn = notch_freqs
        self.q = quality_factors
        self.fc = fc
        self.btype = butterworth_type
        self.order = order
        self.fs = sampling_freq
        self.sosnotch = [np.concatenate(scipy.signal.iirnotch(fn_i, q_i, fs=sampling_freq)).reshape((1, 6)) for
                         fn_i, q_i in
                         zip(notch_freqs, quality_factors)]
        if fc is None:
            self.sosbutter = np.zeros((0, 6))
            self.sos = np.vstack([*self.sosnotch, self.sosbutter])
        elif (isinstance(fc, int) or isinstance(fc, float)) and butterworth_type != 'bandpass':
            wc = fc / sampling_freq * 2.0
            self.sosbutter = scipy.signal.butter(order, wc, btype=butterworth_type, output='sos')
            self.sos = np.vstack([*self.sosnotch, self.sosbutter])
        elif len(fc) == 2 and butterworth_type == 'bandpass':
            wc = [fc[0] / sampling_freq * 2.0, fc[1] / sampling_freq * 2.0]
            if isinstance(order, int):
                self.sosbutter = scipy.signal.butter(order, wc, btype=butterworth_type, output='sos')
                self.sos = np.vstack([*self.sosnotch, self.sosbutter])
            else:
                self.soshp = scipy.signal.butter(order[0], wc[0], btype='highpass', output='sos')
                self.soslp = scipy.signal.butter(order[1], wc[1], btype='lowpass', output='sos')
                self.sos = np.vstack([*self.sosnotch, self.soslp, self.soshp])
        else:
            raise ValueError('Frequency cutoffs must match Butterworth filter type!')

        self.zi0 = scipy.signal.sosfilt_zi(self.sos)

    def filter_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch and butterworth filters to data.
        :param data: Data to be filtered, should have shape (time, channels)
        :return: Filtered data.
        """
        if data.ndim == 1:
            zi = self.zi0 * data[0]
            out, zo = scipy.signal.sosfilt(self.sos, data, zi=zi)
        else:
            zi = (self.zi0[..., None] @ data[0].reshape((1, -1)))
            out, zo = scipy.signal.sosfilt(self.sos, data, axis=0, zi=zi)
        return out
