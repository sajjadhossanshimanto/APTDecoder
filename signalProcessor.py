# Author: Muhtasim Redwan (Avionics, BSMRAAU)
# GitHub: https://github.com/redwine-1
# Time: July 2024

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

SAMPLE_RATE = 20800  # intermediate  sample rate
fc = 2400  # sub carrier frequency


class signalProcessor:
    @staticmethod
    def stereoToMono(stereo_signal: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def resampleSignal(
        input_signal: np.nonzero, input_rate: int, resample_rate: int
    ) -> np.ndarray:
        """
        Resamples the given audio file to the specified sample rate.

        Parameters:
            file_name (str): Path to the wav file.
            resample_rate (int): Desired sample rate of the output signal.

        Returns:
            np.ndarray: The resampled audio signal.
        """
        # Resample the signal if necessary
        if input_rate != resample_rate:
            resample_factor = resample_rate / input_rate
            number_of_output_samples = int(resample_factor * input_signal.size)
            input_signal = signal.resample(input_signal, number_of_output_samples)

        return input_signal

    @staticmethod
    def ampDemod(signal_data: np.ndarray) -> np.ndarray:
        """
        Demodulates the given AM-modulated signal.

        Parameters:
            signal_data (np.ndarray): The AM-modulated signal data.

        Returns:
            np.ndarray: The demodulated signal.
        """
        phi = 2 * np.pi * fc / SAMPLE_RATE
        signal_length = len(signal_data)
        signal_data_squared = np.square(signal_data)

        # Source: https://raw.githubusercontent.com/martinber/noaa-apt/master/extra/demodulation.pdf
        amplitude_signal = np.sqrt(
            signal_data_squared[1:signal_length]
            + signal_data_squared[0 : signal_length - 1]
            - 2
            * signal_data[1:signal_length]
            * signal_data[0 : signal_length - 1]
            * np.cos(phi)
        ) / np.sin(phi)

        # Insert a 0 at the beginning of the array
        amplitude_signal = np.insert(amplitude_signal, 0, 0)

        return amplitude_signal

    def bandpassFilter(signal_data: np.ndarray, lowpass: int, highpass: int):
        """
        Parameters:
            signal_data (np.ndarray): data which need to filtered
            lowpass (int): lowpass frequency
            highpass (int): highpass frequency

        Returns
            np.ndarray: Filtered signal after applying highpass and lowpass filter
        """
        sos = signal.butter(
            8, [highpass, lowpass], "band", fs=SAMPLE_RATE, output="sos"
        )
        filtered = signal.sosfilt(sos, signal_data)
        return filtered
