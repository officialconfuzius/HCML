import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import shutil


def process_image(image_path):
    """
    Applies artifact removal to a single image using color-based masking and inpainting.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process morphed images to remove artifacts and create a triplet file."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory of original morphs (e.g., SYN-MAD22/FaceMorpher_aligned).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save processed morphs (e.g., SYN-MAD22/FaceMorpher_processed).",
    )
    args = parser.parse_args()

    # --- Setup paths ---
    output_dataset_name = os.path.basename(args.output_dir)
    triplet_file_path = f"triplets/SYN-MAD22_{output_dataset_name}_triples_selfmade.txt"

    # --- Create directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(triplet_file_path), exist_ok=True)

    # --- Clear previous triplet file if it exists ---
    if os.path.exists(triplet_file_path):
        os.remove(triplet_file_path)

    print(f"Starting post-processing from '{args.input_dir}' to '{args.output_dir}'...")

    image_files = [
        f
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for filename in tqdm(image_files):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        if "vs" in filename:
            processed_image = process_image(input_path)
            if processed_image is not None:
                cv2.imwrite(output_path, processed_image)

            try:
                benign_img1 = filename.split("-")[0] + ".jpg"
                benign_img2 = filename.split("-")[2].split(".")[0] + ".jpg"
                with open(triplet_file_path, "a") as f:
                    f.write(f"{filename}\t{benign_img1}\t{benign_img2}\n")
            except IndexError:
                print(f"\nWarning: Could not parse triplet from filename: {filename}")
        else:
            shutil.copyfile(input_path, output_path)

    print(f"Post-processing complete. Processed images saved to '{args.output_dir}'.")
    print(f"Triplet file created at '{triplet_file_path}'.")
