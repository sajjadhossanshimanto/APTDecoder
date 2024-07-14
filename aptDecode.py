# Author: Muhtasim Redwan (Avionics, BSMRAAU)
# GitHub: https://github.com/redwine-1
# Time: January 2023

# imports
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
from PIL import Image
import argparse

# Constants
SAMPLE_RATE = (
    20800  # Intermediate sample rate, chosen to balance quality and processing time
)
FC = 2400  # Subcarrier frequency

# APT Signal Structure (based on https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT))
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


def resample_signal(
    input_signal: np.ndarray, input_rate: int, resample_rate: int
) -> np.ndarray:
    """
    Resamples the given signal to the specified sample rate.

    Parameters:
        input_signal (np.ndarray): Input signal
        input_rate (int): Original sample rate of the input signal
        resample_rate (int): Desired sample rate of the output signal

    Returns:
        np.ndarray: Resampled signal
    """
    if input_rate != resample_rate:
        resample_factor = resample_rate / input_rate
        number_of_output_samples = int(resample_factor * len(input_signal))
        return signal.resample(input_signal, number_of_output_samples)
    return input_signal


def bandpass_filter(signal_data: np.ndarray, lowpass: int, highpass: int) -> np.ndarray:
    """
    Applies a bandpass filter to the signal data.

    Parameters:
        signal_data (np.ndarray): Signal to be filtered
        lowpass (int): Lowpass frequency
        highpass (int): Highpass frequency

    Returns:
        np.ndarray: Filtered signal
    """
    sos = signal.butter(8, [highpass, lowpass], "band", fs=SAMPLE_RATE, output="sos")
    return signal.sosfilt(sos, signal_data)


def remap_signal_value(signal_data: np.ndarray) -> np.ndarray:
    """
    Remaps the given signal values to the range 0 to 255.

    Parameters:
        signal_data (np.ndarray): Signal to be remapped

    Returns:
        np.ndarray: Remapped signal
    """
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    return np.round(255 * (signal_data - min_val) / (max_val - min_val)).astype(
        np.uint8
    )


def demodulate(signal_data: np.ndarray) -> np.ndarray:
    """
    Demodulates the given AM-modulated signal.

    Parameters:
        signal_data (np.ndarray): AM-modulated signal data

    Returns:
        np.ndarray: Demodulated signal
    """
    phi = 2 * np.pi * FC / SAMPLE_RATE
    signal_length = len(signal_data)
    signal_data_squared = np.square(signal_data)

    amplitude_signal = np.sqrt(
        signal_data_squared[1:signal_length]
        + signal_data_squared[0 : signal_length - 1]
        - 2
        * signal_data[1:signal_length]
        * signal_data[0 : signal_length - 1]
        * np.cos(phi)
    ) / np.sin(phi)

    return np.insert(amplitude_signal, 0, 0)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Applies histogram equalization to the given image.

    Parameters:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Histogram equalized image
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 255])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    image_equalized = np.interp(image.flatten(), range(256), cdf_normalized)
    return image_equalized.reshape(image.shape).astype(np.uint8)


def rotate_image(matrix: np.ndarray) -> np.ndarray:
    """
    Rotates an image by flipping its channels.

    Parameters:
        matrix (np.ndarray): 2D image matrix

    Returns:
        np.ndarray: Rotated image
    """
    sync_space_A = matrix[:, APT_STRUCTURE["sync_A"][0] : APT_STRUCTURE["space_A"][1]]
    channel_A = matrix[:, APT_STRUCTURE["image_A"][0] : APT_STRUCTURE["image_A"][1]]
    tel_A_2_space_B = matrix[
        :, APT_STRUCTURE["telemetry_A"][0] : APT_STRUCTURE["space_B"][1]
    ]
    channel_B = matrix[:, APT_STRUCTURE["image_B"][0] : APT_STRUCTURE["image_B"][1]]
    telemetry_B = matrix[
        :, APT_STRUCTURE["telemetry_B"][0] : APT_STRUCTURE["telemetry_B"][1]
    ]

    channel_A = np.flip(channel_A)
    channel_B = np.flip(channel_B)

    return np.concatenate(
        [sync_space_A, channel_A, tel_A_2_space_B, channel_B, telemetry_B], axis=1
    )


def synchronize_apt_signal(remapped_signal: np.ndarray) -> np.ndarray:
    """
    Synchronizes the signal and converts it to a 2D image.

    Parameters:
        remapped_signal (np.ndarray): Remapped signal

    Returns:
        np.ndarray: 2D image matrix
    """
    syncA = np.array([0, 0, 255, 255] * 7 + [0] * 7) - 128
    peaks = [(0, 0)]
    min_distance = 2000
    shifted_signal = remapped_signal - 128

    for i in range(len(shifted_signal) - len(syncA)):
        corr = np.dot(syncA, shifted_signal[i : i + len(syncA)])
        if i - peaks[-1][0] > min_distance:
            peaks.append((i, corr))
        elif corr > peaks[-1][1]:
            peaks[-1] = (i, corr)

    matrix = [
        remapped_signal[peaks[i][0] : peaks[i][0] + 2080] for i in range(len(peaks) - 1)
    ]
    return np.array(matrix)


def apt_signal_to_image(raw_signal: np.ndarray, signal_rate: int) -> np.ndarray:
    """
    Processes the raw APT signal and converts it to an image.

    Parameters:
        raw_signal (np.ndarray): Raw signal data
        signal_rate (int): Sample rate of the signal

    Returns:
        np.ndarray: 2D image matrix
    """
    raw_signal = stereo_to_mono(raw_signal)
    print(f"Resampling at {SAMPLE_RATE} Hz")
    signal_data = resample_signal(raw_signal, signal_rate, SAMPLE_RATE)
    truncate_length = SAMPLE_RATE * (len(signal_data) // SAMPLE_RATE)
    signal_data = signal_data[:truncate_length]

    print("Applying bandpass filter (1000 Hz - 4000 Hz)")
    signal_data_filtered = bandpass_filter(signal_data, lowpass=4000, highpass=1000)

    print("Demodulating signal")
    demodulated_signal = demodulate(signal_data_filtered)

    reshaped = demodulated_signal.reshape(len(demodulated_signal) // 5, 5)
    demodulated_signal = reshaped[:, 2]

    print("Remapping signal values between 0 and 255")
    remapped_signal = remap_signal_value(demodulated_signal)

    print("Creating image matrix")
    image_matrix = synchronize_apt_signal(remapped_signal)

    print("Applying histogram equalization")
    return histogram_equalization(image_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APT Signal Decoder")
    parser.add_argument("-i", "--input", help="Input WAV file", required=True)
    parser.add_argument("-o", "--output", help="Output image file", required=True)
    parser.add_argument("-r", "--rotate", help="Rotate the image", action="store_true")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    rotate = args.rotate

    signal_rate, raw_signal = wav.read(input_file)
    image_matrix = apt_signal_to_image(raw_signal, signal_rate)

    if rotate:
        image_matrix = rotate_image(image_matrix)

    image = Image.fromarray(image_matrix)
    image.save(output_file)

    print(f"Decoded image saved as {output_file}")
