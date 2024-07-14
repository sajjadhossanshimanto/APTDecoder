import numpy as np


class imageProcessor:

    @staticmethod
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

    @staticmethod
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
