# Author: Muhtasim Redwan (Avioncis, BSMRAAU)
# GitHub: https://github.com/redwine-1
# Time: January 2023

# imports
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
from PIL import Image
import argparse
from signalProcessor import signalProcessor

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


def remap_signal_value(signal: np.ndarray) -> np.ndarray[np.uint8]:
    """
    Remaps the given signal values within the range 0 to 255.

    Parameters:
        signal (np.ndarray): signal which need to remap #TODO: use better docstring

    Returns:
        np.ndarray[np.uint8]: The remapped signal, with values in the range 0 to 255.
    """
    # Find the minimum and maximum values of the demodulated signal
    min_val = signal.min()
    max_val = signal.max()

    # Calculate the range of the demodulated signal
    delta = max_val - min_val

    # Remap the demodulated signal to the range 0 to 255
    remapped = np.round(255 * (signal - min_val) / delta)

    # Return the remapped signal as an array of unsigned 8-bit integers
    return remapped.astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Implements histogram equalization for an image represented as a NumPy array

    Parameters:
        image: Image represents as 2D Numpy array

    Returns:
        Histogram equalized image as 2D Numpy array.
    """

    # Calculate the histogram of the image
    hist, _ = np.histogram(image, range=(0, 255), bins=256)

    # Normalize the histogram
    hist = hist / image.size

    # Calculate the cumulative distribution function (CDF) of the normalized histogram
    cdf = hist.cumsum()

    # Map the intensity values of the pixels in the input image to new intensity values using the CDF
    image_equalized = np.interp(image, range(256), cdf)

    # Scale the output image to the range 0-255
    image_equalized = (image_equalized * 255).astype(np.uint8)

    return image_equalized


def rotate_image(matrix):
    """
    Rotates an image represented as a 2D NumPy array by rearranging the columns of the array.

    Parameters:
        matrix (np.ndarray): The 2D Numpy array with rows representing the rows of the image
        and columns representing the columns of the image.

    Returns:
        Rotated image represent by 2D Numpy array

    """
    # Get slices of the input matrix corresponding to different regions of the image
    sync_space_A = matrix[:, APT_STRUCTURE["sync_A"][0] : APT_STRUCTURE["space_A"][1]]
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


def apt_signal_to_image(raw_signal: np.ndarray, signal_rate: int) -> np.ndarray:
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
    signal_data = signalProcessor.resample_signal(raw_signal, signal_rate, SAMPLE_RATE)

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
    remapped = remap_signal_value(demodulated_signal)

    # Create an image matrix from the signal data
    print("Creating image matrix")
    image_matrix = synchronize_apt_signal(remapped)

    # Perform histogram equalization on the image matrix
    image_matrix = histogram_equalization(np.array(image_matrix))

    return image_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input Wav File", required=True)
    parser.add_argument("-o", "--output", help="Output File", required=True)
    parser.add_argument(
        "-r", "--rotate", help="Enables image rotation", action="store_true"
    )
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    rotate = args.rotate

    signal_rate, raw_signal = wav.read(input_file)

    image_matrix = apt_signal_to_image(raw_signal, signal_rate)

    # Rotate the image matrix
    if rotate:
        image_matrix = rotate_image(image_matrix)

    # Save the image matrix as an image file
    image = Image.fromarray(image_matrix)
    image.save(output_file)

    print(f"Decoded image saved as {output_file}")
