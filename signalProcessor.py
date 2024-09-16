# Author: Muhtasim Redwan (Avionics, BSMRAAU)
# GitHub: https://github.com/redwine-1
# Time: July 2024

import numpy as np
import scipy.signal as signal


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
            input_signal (np.nonzero): signal to be resampled
            input_rate (int): input signal sample rate
            resample_rate (int): desired sample rate

        Returns:
            np.ndarray: resampled signal
        """
        if input_rate == resample_rate:
            return input_signal

        resample_factor = resample_rate / input_rate
        number_of_output_samples = int(resample_factor * input_signal.size)
        output_signal = signal.resample(input_signal, number_of_output_samples)

        return output_signal

    @staticmethod
    def ampDemod(
        input_signal: np.ndarray, carrierFrequency: int, sampleFrequency: int
    ) -> np.ndarray:
        """
        Demodulates the given AM-modulated signal.

        Source: https://raw.githubusercontent.com/martinber/noaa-apt/master/extra/demodulation.pdf

        Parameters:
            input_signal (np.ndarray): input signal
            carrierFrequency (int): carrier frequency of the signal
            sampleFrequency (int): sample frequency of the input signal

        Returns:
            np.ndarray: Demodulated signal
        """
        phi = 2 * np.pi * carrierFrequency / sampleFrequency
        signal_length = len(input_signal)
        signal_data_squared = np.square(input_signal)

        amplitude_signal = np.sqrt(
            signal_data_squared[1:signal_length]
            + signal_data_squared[0 : signal_length - 1]
            - 2
            * input_signal[1:signal_length]
            * input_signal[0 : signal_length - 1]
            * np.cos(phi)
        ) / np.sin(phi)

        # Insert a 0 at the beginning of the array
        amplitude_signal = np.insert(amplitude_signal, 0, 0)

        return amplitude_signal

    @staticmethod
    def bandpassFilter(
        input_signal: np.ndarray,
        sampleFreq: int,
        lowpass: int,
        highpass: int,
        filterOrder: int = 8,
    ) -> np.ndarray:
        """applies bandpass filter to the input signal

        Parameters:
            input_signal (np.ndarray):
            sampleFreq (int): sample frequency of the signal
            lowpass (int): lower cutoff frequency
            highpass (int): upper cutoff frequency
            filterOrder (int, optional): order of the filter. Defaults to 8.

        Returns:
            np.ndarray: filtered signal
        """

        sos = signal.butter(
            filterOrder, [highpass, lowpass], "band", fs=sampleFreq, output="sos"
        )
        filtered = signal.sosfilt(sos, input_signal)
        return filtered
