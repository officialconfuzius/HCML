import cv2
import os
from tqdm import tqdm
import shutil

# You will need to download this file and place it in your project folder.
# It's a pre-trained model for face detection.
# You can find it by searching for "haarcascade_frontalface_default.xml"
CASCADE_FILE = "haarcascade_frontalface_default.xml"


def process_image(image_path, face_cascade):
    """
    Detects a face, enlarges the bounding box by 40%, and crops the image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    original_height, original_width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # Assume the largest detected face is the primary one
        (x, y, w, h) = sorted(
            faces, key=lambda f: f[2] * f[3], reverse=True)[0]

        # --- Enlarge the bounding box by 40% (20% on each side) ---
        center_x, center_y = x + w // 2, y + h // 2
        new_w = int(w * 1.4)
        new_h = int(h * 1.4)

        # Calculate new top-left corner
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)

        # Ensure the new crop is within the image boundaries
        new_w = min(new_w, original_width - new_x)
        new_h = min(new_h, original_height - new_y)

        # Crop the image to the new enlarged bounding box
        cropped_face = img[new_y: new_y + new_h, new_x: new_x + new_w]

        # Resize the cropped face back to the original image size
        # This is crucial for the FR model to process it correctly.
        resized_img = cv2.resize(
            cropped_face,
            (original_width, original_height),
            interpolation=cv2.INTER_AREA,
        )

        return resized_img
    else:
        # If no face is detected, return the original image
        return img


if __name__ == "__main__":
    # --- Load the face detector ---
    if not os.path.exists(CASCADE_FILE):
        raise FileNotFoundError(
            f"'{CASCADE_FILE}' not found. Please download it and place it in the project directory."
        )
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

    datasets = [
        "SYN-MAD22/FaceMorpher_aligned", "SYN-MAD22/MIPGAN_I_aligned", "SYN-MAD22/MIPGAN_II_aligned",
        "SYN-MAD22/Webmorph_aligned", "SYN-MAD22/MorDIFF_aligned", "SYN-MAD22/OpenCV_aligned"
    ]
    outputs = [
        "SYN-MAD22/FaceMorpher_processed_improved_alignment", "SYN-MAD22/MIPGAN_I_processed_improved_alignment", "SYN-MAD22/MIPGAN_II_processed_improved_alignment",
        "SYN-MAD22/Webmorph_processed_improved_alignment", "SYN-MAD22/MorDIFF_processed_improved_alignment", "SYN-MAD22/OpenCV_processed_improved_alignment"
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
                processed_image = process_image(input_path, face_cascade)
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
                processed_image = process_image(input_path, face_cascade)
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
