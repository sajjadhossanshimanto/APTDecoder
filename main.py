import argparse
import scipy.io.wavfile as wav
from PIL import Image
from APTDecoder import APTDecoder


def parse_arguments():
    parser = argparse.ArgumentParser(description="APT Signal Decoder")
    parser.add_argument("-i", "--input", help="Input WAV file", required=True)
    parser.add_argument("-o", "--output", help="Output image file", required=True)
    parser.add_argument(
        "-r", "--rotate", help="Enables image rotation", action="store_true"
    )
    return parser.parse_args()


def read_input(input_file):
    return wav.read(input_file)


def save_image(image_matrix, output_file):
    image = Image.fromarray(image_matrix)
    image.save(output_file)
    print(f"Decoded image saved as {output_file}")


def main():
    args = parse_arguments()

    signal_rate, raw_signal = read_input(args.input)
    decoder = APTDecoder()
    image_matrix = decoder.apt_signal_to_image(raw_signal, signal_rate)

    if args.rotate:
        image_matrix = decoder.rotate_image(image_matrix)

    save_image(image_matrix, args.output)


if __name__ == "__main__":
    main()
