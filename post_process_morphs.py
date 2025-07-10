import cv2
import numpy as np
import os
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

    datasets = [
        "SYN-MAD22/FaceMorpher_aligned", "SYN-MAD22/MIPGAN_I_aligned", "SYN-MAD22/MIPGAN_II_aligned",
        "SYN-MAD22/Webmorph_aligned", "SYN-MAD22/MorDIFF_aligned", "SYN-MAD22/OpenCV_aligned"
    ]
    outputs = [
        "SYN-MAD22/FaceMorpher_processed_orange", "SYN-MAD22/MIPGAN_I_processed_orange", "SYN-MAD22/MIPGAN_II_processed_orange",
        "SYN-MAD22/Webmorph_processed_orange", "SYN-MAD22/MorDIFF_processed_orange", "SYN-MAD22/OpenCV_processed_orange"
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
