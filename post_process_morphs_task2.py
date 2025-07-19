import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import shutil


def apply_frequency_filter(image, radius=30):
    """
    Applies a low-pass frequency filter to an image to remove high-frequency artifacts.
    This is done for each color channel individually.
    """
    channels = cv2.split(image)
    filtered_channels = []

    for channel in channels:
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, [1, 1], -1)
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        filtered_channels.append(np.uint8(img_back))

    return cv2.merge(filtered_channels)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process morphs using frequency domain filtering and create a corresponding triplet file."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory of original morph images and their benign sources.",
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

    # --- Setup paths for output images and the new triplet file ---
    output_dataset_name = os.path.basename(args.output_dir)
    triplets_dir = "triplets"
    triplet_file_path = os.path.join(
        triplets_dir, f"SYN-MAD22_{output_dataset_name}_triples.txt"
    )

    # --- Create directories and clean up old triplet file if it exists ---
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(triplets_dir, exist_ok=True)
    if os.path.exists(triplet_file_path):
        os.remove(triplet_file_path)
        print(f"Removed old triplet file: {triplet_file_path}")

    image_files = [
        f
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Starting frequency filtering with radius = {args.radius}...")
    for filename in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        # --- Differentiate between morphed and benign images ---
        if "vs" in filename:
            # This is a morphed image, so we apply the filter
            image = cv2.imread(input_path)
            if image is not None:
                processed_image = apply_frequency_filter(image, args.radius)
                cv2.imwrite(output_path, processed_image)

                # And now, we create the triplet entry for this morph
                try:
                    parts = filename.split("-")
                    benign_img1 = parts[0] + ".jpg"
                    benign_img2 = parts[2].split(".")[0] + ".jpg"
                    with open(triplet_file_path, "a") as f:
                        f.write(f"{filename}\t{benign_img1}\t{benign_img2}\n")
                except IndexError:
                    print(
                        f"\nWarning: Could not parse triplet from filename: {filename}"
                    )
        else:
            # This is a benign (source) image, so we just copy it without modification
            shutil.copyfile(input_path, output_path)

    print("\nProcessing complete.")
    print(f"Processed images saved in: {args.output_dir}")
    print(f"New triplet file created at: {triplet_file_path}")


if __name__ == "__main__":
    main()
