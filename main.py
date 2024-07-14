import argparse
import scipy.io.wavfile as wav
from PIL import Image
from APTDecoder import APTDecoder


def parseArguments():
    parser = argparse.ArgumentParser(description="APT Signal Decoder")
    parser.add_argument("-i", "--input", help="Input WAV file", required=True)
    parser.add_argument("-o", "--output", help="Output image file", required=True)
    parser.add_argument(
        "-r", "--rotate", help="Enables image rotation", action="store_true"
    )
    return parser.parse_args()


def readWavFile(input_file):
    return wav.read(input_file)


def saveImage(image_matrix, output_file):
    image = Image.fromarray(image_matrix)
    image.save(output_file)
    print(f"Decoded image saved as {output_file}")


def main():
    args = parseArguments()

    signalRate, rawSignal = readWavFile(args.input)
    decoder = APTDecoder(signalRate=signalRate)
    image_matrix = decoder.APTSignalToImage(rawSignal)

    if args.rotate:
        image_matrix = decoder.rotateImage(image_matrix)

    saveImage(image_matrix, args.output)


if __name__ == "__main__":
    main()
