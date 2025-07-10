import cv2
import numpy as np
import os
from tqdm import tqdm
import shutil


def process_image(image_path):
    """
    Removes gray background artifacts while protecting a central "safe zone".
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # --- Step 1: Define the central "safe zone" to protect ---
    h, w, _ = img.shape
    # Define a rectangle from 15% to 85% of the image dimensions.
    # This covers the main face but excludes the outer edges.
    start_x, end_x = int(w * 0.15), int(w * 0.85)
    start_y, end_y = int(h * 0.15), int(h * 0.85)

    # --- Step 2: Create a mask for the gray background artifacts ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define a range for gray colors (low saturation, medium brightness)
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 220])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # --- Step 3: Exclude the safe zone from the artifact mask ---
    # Create a black mask and draw the white safe zone on it
    safe_zone_mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(safe_zone_mask, (start_x, start_y), (end_x, end_y), 255, -1)

    # Invert the safe zone mask (now the center is black)
    protection_mask = cv2.bitwise_not(safe_zone_mask)

    # Combine the masks: find gray pixels that are OUTSIDE the safe zone
    final_mask = cv2.bitwise_and(gray_mask, protection_mask)

    # --- Step 4: Inpaint using the final mask ---
    inpainted_img = cv2.inpaint(img, final_mask, 3, cv2.INPAINT_TELEA)

    return inpainted_img


if __name__ == "__main__":

    datasets = [
        "SYN-MAD22/FaceMorpher_aligned", "SYN-MAD22/MIPGAN_I_aligned", "SYN-MAD22/MIPGAN_II_aligned",
        "SYN-MAD22/Webmorph_aligned", "SYN-MAD22/MorDIFF_aligned", "SYN-MAD22/OpenCV_aligned"
    ]
    outputs = [
        "SYN-MAD22/FaceMorpher_processed_gray", "SYN-MAD22/MIPGAN_I_processed_gray", "SYN-MAD22/MIPGAN_II_processed_gray",
        "SYN-MAD22/Webmorph_processed_gray", "SYN-MAD22/MorDIFF_processed_gray", "SYN-MAD22/OpenCV_processed_gray"
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
