# Author: Muhtasim Redwan (Avioncis, BSMRAAU)
# GitHub: https://github.com/redwine-1
# Time: January 2023

# imports
import numpy as np
from signalProcessor import signalProcessor
from imageProcessor import imageProcessor

# TODO: explain the reason of sampling at 20800
SAMPLE_RATE = 20800  # intermediate  sample rate
fc = 2400  # sub carrier frequency

# https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT)
# structure of one APT line
APT_STRUCTURE = {
    "pixel_per_row": 2080,
    "pixel_per_channel": 1040,
    "sync_A": (0, 39),
    "space_A": (39, 86),
    "image_A": (86, 995),
    "telemetry_A": (995, 1040),
    "sync_B": (1040, 1079),
    "space_B": (1079, 1126),
    "image_B": (1126, 2035),
    "telemetry_B": (2035, 2080),
}


class APTDecoder:

    def __init__(self) -> None:
        pass

    @staticmethod
    def rotate_image(self, matrix):
        """
        Rotates an image represented as a 2D NumPy array by rearranging the columns of the array.

        Parameters:
            matrix (np.ndarray): The 2D Numpy array with rows representing the rows of the image
            and columns representing the columns of the image.

        Returns:
            Rotated image represent by 2D Numpy array

        """
        # Get slices of the input matrix corresponding to different regions of the image
        sync_space_A = matrix[
            :, APT_STRUCTURE["sync_A"][0] : APT_STRUCTURE["space_A"][1]
        ]
        channel_A = matrix[:, APT_STRUCTURE["image_A"][0] : APT_STRUCTURE["image_A"][1]]
        tel_A_2_space_B = matrix[
            :, APT_STRUCTURE["telemetry_A"][0] : APT_STRUCTURE["space_B"][1]
        ]
        channel_B = matrix[:, APT_STRUCTURE["image_B"][0] : APT_STRUCTURE["image_B"][1]]
        telemetry_B = matrix[
            :, APT_STRUCTURE["telemetry_B"][0] : APT_STRUCTURE["telemetry_B"][1]
        ]

        # Flip the image channels
        channel_A = np.flip(channel_A)
        channel_B = np.flip(channel_B)

        # Concatenate the slices of the matrix to form the output matrix
        rotated_matrix = np.concatenate(
            (
                sync_space_A,
                channel_A,
                tel_A_2_space_B,
                channel_B,
                telemetry_B,
            ),
            axis=1,
        )
        return rotated_matrix

    def synchronize_apt_signal(
        self,
        remapped_signal,
    ):  # TODO: use two function one to synchronize another to convert 1D to 2D
        """
        Synchronizes the given signal by finding the sync frame and converting the 1D signal to a 2D image.
        The sync frame is found by looking for the maximum values of the cross correlation between the signal and a
        hardcoded syncA signal. The minimum distance between sync frames is set to 2000.

        Parameters:
            remapped_signal (np.ndarray): The demodulated signal, remapped to the range 0-255.

        Returns:
            np.ndarray: The resulting 2D image.
        """
        # hard coded syncA
        syncA = np.array([0, 0, 255, 255] * 7 + [0 * 7])
        syncA = [x - 128 for x in syncA]

        # list of maximum correlations found (index, corr)
        peaks = [(0, 0)]

        # using minimum distance as 2000
        min_distance = 2000
        shifted_signal = [x - 128 for x in remapped_signal]

        # finds the maximum value of correlation between syncA and signal_data
        for i in range(len(shifted_signal) - len(syncA)):
            corr = np.dot(syncA, shifted_signal[i : i + len(syncA)])
            if i - peaks[-1][0] > min_distance:
                peaks.append((i, corr))
            elif corr > peaks[-1][1]:
                peaks[-1] = (i, corr)

        matrix = []

        for i in range(len(peaks) - 1):
            matrix.append(remapped_signal[peaks[i][0] : peaks[i][0] + 2080])

        return np.array(matrix)

    def apt_signal_to_image(
        self, raw_signal: np.ndarray, signal_rate: int
    ) -> np.ndarray:
        """
        Decodes an encoded image file and saves the resulting image.

        Parameters:
            in_file (str): The path to the input file.
            out_file (str): The path to the output file.
            rotate (bool, optional): A flag indicating whether the image should be rotated. Default is False.

        Returns:
            np.ndarray: 2D Image array
        """
        # Convert stereo to mono audio
        raw_signal = signalProcessor.stereo_to_mono(raw_signal)

        # Resample the signal data at 20800 sample rate
        print(f"Resampling at {SAMPLE_RATE}hz")
        signal_data = signalProcessor.resample_signal(
            raw_signal, signal_rate, SAMPLE_RATE
        )

        # Truncate the signal data to an integer multiple of the sample rate
        truncate = SAMPLE_RATE * int(len(signal_data) // SAMPLE_RATE)
        signal_data = signal_data[: int(truncate)]

        # Apply a bandpass filter to the signal data
        print(
            "Applying bandpass filter 1000 highpass and 4000 lowpass "
        )  # TODO: change hardcoded numerical value
        signal_data_filtered = signalProcessor.bandpass_filter(
            signal_data, lowpass=4000, highpass=1000
        )

        # Demodulate the filtered signal data
        print("Demodulating signal")
        demodulated_signal = signalProcessor.ampDemod(signal_data_filtered)

        # Downsample the demodulated signal data to baud rate (4160 Hz)
        reshaped = demodulated_signal.reshape(len(demodulated_signal) // 5, 5)
        demodulated_signal = reshaped[:, 2]

        # Remap the values of the signal data to a range between 0 and 255
        print("Remapping signal values between 0 and 255")
        remapped = imageProcessor.remap_signal_value(demodulated_signal)

        # Create an image matrix from the signal data
        print("Creating image matrix")
        image_matrix = self.synchronize_apt_signal(remapped)

        # Perform histogram equalization on the image matrix
        image_matrix = imageProcessor.histogram_equalization(np.array(image_matrix))

        return image_matrix
