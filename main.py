# Main
import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

from backbones.curr_resnet import curr_iresnet100
from backbones.elastic_resnet import elastic_iresnet100

# ==========================================================================================
# Original Helper Functions (No changes needed here)
# ==========================================================================================


def cos_sim(a, b):
    """Calculates cosine similarity between vector a and b."""
    a, b = a.reshape(-1), b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_model(model_name, model_path, device):
    """Load model from path_model."""
    if not os.path.exists(model_path):
        raise Exception(f"Model file not found at: {model_path}")
    if model_name in ["ElasticFaceArc", "ElasticFaceCos"]:
        backbone = elastic_iresnet100(num_features=512).to(device)
    elif model_name == "CurricularFace":
        backbone = curr_iresnet100().to(device)
    else:
        raise Exception("Cannot load unknown model:", model_name)

    backbone.load_state_dict(torch.load(model_path, map_location=device))
    backbone.eval()
    return backbone


def image_iter(path):
    """Return image paths and filenames in filepath."""
    image_paths = []
    file_names = []
    for path, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(path, name))
                file_names.append(name)
    image_paths.sort()
    file_names.sort()
    return image_paths, file_names


def load_and_prep_images(src_path, morphing_dataset_name):
    """Loads images, normalizes them, and generates a triplet file if applicable."""
    img_paths, file_names = image_iter(src_path)
    imgs = []
    triplet_file_name = (
        f"triplets/SYN-MAD22_{morphing_dataset_name}_triples_selfmade.txt"
    )

    # Create directory for triplets if it doesn't exist
    os.makedirs(os.path.dirname(triplet_file_name), exist_ok=True)

    # Clear previous triplet file if it exists, as we are regenerating embeddings
    if os.path.exists(triplet_file_name):
        os.remove(triplet_file_name)

    for p in tqdm(img_paths, desc=f"Loading images from {morphing_dataset_name}"):
        img_name = os.path.basename(p)
        if "vs" in img_name:
            morph_img = img_name
            try:
                benign_img1 = img_name.split("-")[0] + ".jpg"
                benign_img2 = img_name.split("-")[2].split(".")[0] + ".jpg"
                with open(triplet_file_name, "a") as f:
                    f.write(f"{morph_img}\t{benign_img1}\t{benign_img2}\n")
            except IndexError:
                print(f"\nWarning: Could not parse triplet from filename: {img_name}")

        img = cv2.imread(p)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = np.asarray([img], dtype="float32")
            img = ((img / 255) - 0.5) / 0.5
            img = torch.tensor(img)
            imgs.append(img)

    return file_names, torch.cat(imgs, dim=0) if imgs else ([], None)


def extract_embeddings(model_name, model_path, src_path, dest_path, device):
    """Extract embeddings from images using a specified model and save them."""
    print(f"\nExtracting embeddings for dataset: {os.path.basename(src_path)}")

    # Prepare inference
    batchsize = 32
    backbone = load_model(model_name, model_path, device)
    backbone.to(device)

    # Load Images and get filenames
    file_names, imgs_tensor = load_and_prep_images(src_path, os.path.basename(src_path))
    if imgs_tensor is None:
        print(f"No images found in {src_path}. Skipping embedding extraction.")
        return

    imgs_tensor = torch.split(imgs_tensor, batchsize)

    # Compute embeddings
    embs = []
    for batch in tqdm(imgs_tensor, desc="Computing embeddings"):
        with torch.no_grad():
            batch = batch.to(device)
            if model_name in ["CurricularFace", "AdaFace"]:
                _embedding = backbone(batch)[0]
            else:
                _embedding = backbone(batch)
            embs.extend(_embedding.cpu().numpy())

    # Save embeddings
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    np.save(dest_path + "_embs", np.array(embs))
    np.save(dest_path + "_ids", np.array(file_names))
    print("Embeddings saved successfully.")


def get_reference(rid, id_list):
    """Finds the index of a specific ID in an ID list."""
    for idx, i in enumerate(id_list):
        if i == rid:
            return idx
    return None


def eval_morph(triple, morphs, morph_id, ref, ref_ids, thresh=0):
    """Performs the MMPMR and IAPAR evaluation."""
    mated, non_mated, iapar_match, iapar_non_match = [], [], [], []

    for i, (mid, id1, id2) in enumerate(tqdm(triple, desc="Evaluating morphs")):
        morph_idx = get_reference(mid, morph_id)
        emb1_idx = get_reference(id1, ref_ids)
        emb2_idx = get_reference(id2, ref_ids)

        if morph_idx is None or emb1_idx is None or emb2_idx is None:
            non_mated.append(None)  # Can't evaluate this triplet
            continue

        morph = morphs[morph_idx]
        emb1 = ref[emb1_idx]
        emb2 = ref[emb2_idx]

        cos1 = cos_sim(emb1, morph)
        cos2 = cos_sim(emb2, morph)

        if cos1 > thresh and cos2 > thresh:
            mated.append(morph)
        else:
            non_mated.append(morph)

        if cos1 > thresh:
            iapar_match.append(cos1)
        else:
            iapar_non_match.append(cos1)
        if cos2 > thresh:
            iapar_match.append(cos2)
        else:
            iapar_non_match.append(cos2)

    total_evaluated = len(mated) + len(non_mated)
    mmpmr = len(mated) / total_evaluated if total_evaluated > 0 else 0

    total_iapar = len(iapar_match) + len(iapar_non_match)
    iapar = len(iapar_match) / total_iapar if total_iapar > 0 else 0

    return mmpmr, iapar


# ==========================================================================================
# New Main Execution Block
# ==========================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Morphing Embedding & Evaluation Toolkit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ElasticFaceArc",
        choices=["ElasticFaceArc", "ElasticFaceCos", "CurricularFace"],
        help="Select the face recognition model to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the morphing dataset folder in SYN-MAD22 (e.g., FaceMorpher_aligned).",
    )
    parser.add_argument(
        "--fmr",
        type=float,
        default=0.01,
        choices=[0.01, 0.001],
        help="FNMR@FMR threshold level (1% or 0.1%).",
    )
    args = parser.parse_args()

    # --- Setup constants and paths based on arguments ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    REF_DATASET = "BonaFide_aligned"

    model_path = f"models/{args.model}.pth"
    morph_dataset_path = f"SYN-MAD22/{args.dataset}/"
    ref_dataset_path = f"SYN-MAD22/{REF_DATASET}/"

    morph_emb_dest_path = f"embeddings/{args.model}_{args.dataset}"
    ref_emb_dest_path = f"embeddings/{args.model}_{REF_DATASET}"
    triplet_file_path = f"triplets/SYN-MAD22_{args.dataset}_triples_selfmade.txt"

    # --- Select threshold based on arguments ---
    thresholds = {
        "ElasticFaceArc": {0.01: 0.21402553, 0.001: 0.29908442},
        "ElasticFaceCos": {0.01: 0.18321043, 0.001: 0.26074028},
        "CurricularFace": {0.01: 0.1901376, 0.001: 0.26934636},
    }
    thresh = thresholds[args.model][args.fmr]

    # --- Intelligently generate missing files ---
    print("--- Checking for necessary files ---")
    # 1. Check for reference (Bona Fide) embeddings
    if not os.path.exists(ref_emb_dest_path + "_embs.npy"):
        extract_embeddings(
            args.model, model_path, ref_dataset_path, ref_emb_dest_path, device
        )

    # 2. Check for morphed image embeddings and triplet file
    if not os.path.exists(morph_emb_dest_path + "_embs.npy") or not os.path.exists(
        triplet_file_path
    ):
        extract_embeddings(
            args.model, model_path, morph_dataset_path, morph_emb_dest_path, device
        )

    print("\n--- All necessary files are present. Starting evaluation. ---")

    # --- Load all data for evaluation ---
    morphs = np.load(morph_emb_dest_path + "_embs.npy")
    morph_ids = np.load(morph_emb_dest_path + "_ids.npy")

    ref_embs = np.load(ref_emb_dest_path + "_embs.npy")
    ref_ids = np.load(ref_emb_dest_path + "_ids.npy")

    triplets = np.genfromtxt(triplet_file_path, delimiter="\t", dtype=str)

    # --- Perform the evaluation ---
    mmpmr, iapar = eval_morph(
        triple=triplets,
        morphs=morphs,
        morph_id=morph_ids,
        ref=ref_embs,
        ref_ids=ref_ids,
        thresh=thresh,
    )

    # --- Print final results ---
    print("\n" + "=" * 40)
    print("          EVALUATION RESULTS")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Morphing Dataset: {args.dataset}")
    print(f"Threshold (FNMR@FMR={args.fmr*100}%): {thresh}")
    print("-" * 40)
    print(f"MMPMR: {mmpmr:.4f}")
    print(f"IAPAR: {iapar:.4f}")
    print("=" * 40)
