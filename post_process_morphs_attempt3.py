import cv2
import numpy as np
import os
from tqdm import tqdm
import shutil


def process_image(image_path):
    """
    Replaces gray background artifacts with a sampled background color,
    while protecting a central "safe zone".
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # --- Step 1: Sample the target background color from the top-left corner ---
    # Take a 10x10 patch and average the color to get a stable background color
    corner_patch = img[0:10, 0:10]
    background_color = np.mean(corner_patch, axis=(0, 1)).astype(int)

    # --- Step 2: Define the central "safe zone" to protect ---
    h, w, _ = img.shape
    start_x, end_x = int(w * 0.15), int(w * 0.85)
    start_y, end_y = int(h * 0.15), int(h * 0.85)

    # --- Step 3: Create a mask for the gray background artifacts ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 220])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # --- Step 4: Exclude the safe zone from the artifact mask ---
    safe_zone_mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(safe_zone_mask, (start_x, start_y), (end_x, end_y), 255, -1)
    protection_mask = cv2.bitwise_not(safe_zone_mask)
    final_mask = cv2.bitwise_and(gray_mask, protection_mask)

    # --- Step 5: Replace the artifact pixels with the background color ---
    # Create a copy of the original image to modify
    result_img = img.copy()
    # Where the final mask is white (255), set the pixels to our sampled color
    result_img[final_mask == 255] = background_color

    return result_img


if __name__ == "__main__":

    datasets = [
        "SYN-MAD22/FaceMorpher_aligned", "SYN-MAD22/MIPGAN_I_aligned", "SYN-MAD22/MIPGAN_II_aligned",
        "SYN-MAD22/Webmorph_aligned", "SYN-MAD22/MorDIFF_aligned", "SYN-MAD22/OpenCV_aligned"
    ]
    outputs = [
        "SYN-MAD22/FaceMorpher_processed_replace_background", "SYN-MAD22/MIPGAN_I_processed_replace_background", "SYN-MAD22/MIPGAN_II_processed_replace_background",
        "SYN-MAD22/Webmorph_processed_replace_background", "SYN-MAD22/MorDIFF_processed_replace_background", "SYN-MAD22/OpenCV_processed_replace_background"
    ]

    for input_dir, output_dir in zip(datasets, outputs):
        output_dataset_name = os.path.basename(output_dir)
        triplet_file_path = f"triplets/SYN-MAD22_{output_dataset_name}_triples.txt"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(triplet_file_path), exist_ok=True)

        if os.path.exists(triplet_file_path):
            os.remove(triplet_file_path)

        print(
            f"Starting post-processing from '{input_dir}' to '{output_dir}'...")

        image_files = [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for filename in tqdm(image_files, desc=f"Processing {output_dataset_name}"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

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
                    print(
                        f"\nWarning: Could not parse triplet from filename: {filename}")
            elif "and" in filename:
                processed_image = process_image(input_path)
                if processed_image is not None:
                    cv2.imwrite(output_path, processed_image)

                try:
                    benign_img1 = filename.split(
                        "_")[1] + "_" + filename.split("_")[2] + ".jpg"
                    benign_img2 = filename.split(
                        "_")[4] + "_" + filename.split("_")[5].split(".")[0] + ".jpg"
                    with open(triplet_file_path, "a") as f:
                        f.write(f"{filename}\t{benign_img1}\t{benign_img2}\n")
                except IndexError:
                    print(
                        f"\nWarning: Could not parse triplet from filename: {filename}")
            else:
                shutil.copyfile(input_path, output_path)

        print(
            f"Post-processing complete. Processed images saved to '{output_dir}'.")
        print(f"Triplet file created at '{triplet_file_path}'.")
