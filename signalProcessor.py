# Author: Muhtasim Redwan (Avionics, BSMRAAU)
# GitHub: https://github.com/redwine-1
# Time: July 2024

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal


class signalProcessor:
    @staticmethod
    def stereo_to_mono(stereo_signal: np.ndarray) -> np.ndarray:
        """
        Converts stereo signal to mono by taking the average of the two channels.

        Parameters:
            stereo_signal (np.ndarray): Stereo signal

        Returns:
            np.ndarray: Mono signal
        """
        if stereo_signal.ndim == 1:
            return stereo_signal
        elif stereo_signal.shape[1] == 2:
            return np.mean(stereo_signal, axis=1)
        else:
            raise ValueError("Unsupported signal format")
