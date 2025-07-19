import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm


def apply_frequency_filter(image, radius=30):
    """
    Applies a low-pass frequency filter to an image to remove high-frequency artifacts.
    This is done for each color channel individually.
    """
    # Split the image into its B, G, R channels
    channels = cv2.split(image)
    filtered_channels = []

    for channel in channels:
        # 1. Apply Fourier Transform and shift the zero-frequency component to the center
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # 2. Create a circular low-pass filter mask
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, [1, 1], -1)

        # 3. Apply the mask to the frequency spectrum
        fshift = dft_shift * mask

        # 4. Apply Inverse Fourier Transform
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)

        # 5. Restore the image to a viewable format
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize the image to the 0-255 range
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        filtered_channels.append(np.uint8(img_back))

    # Merge the processed channels back into a color image
    return cv2.merge(filtered_channels)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process morphs using frequency domain filtering."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory of morph images to process."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the processed images."
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=60,
        help="The radius of the low-pass filter. Smaller is stronger.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    image_files = [
        f
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Starting frequency filtering with radius = {args.radius}...")
    for filename in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        image = cv2.imread(input_path)
        if image is not None:
            # Apply the advanced processing
            processed_image = apply_frequency_filter(image, args.radius)

            # Save the result
            cv2.imwrite(output_path, processed_image)

    print("\nProcessing complete.")
    print(f"Processed images saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
