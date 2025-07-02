import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
import re

# Assuming these backbones are in the specified path
from backbones.curr_resnet import curr_iresnet100
from backbones.elastic_resnet import elastic_iresnet100


def cos_sim(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_model(model_name, model_path, device):
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
    image_paths, file_names = [], []
    for path, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(path, name))
                file_names.append(name)
    image_paths.sort()
    file_names.sort()
    return image_paths, file_names


def load_images_for_embeddings(src_path):
    """Simplified function to only load and prepare images for tensor processing."""
    img_paths, file_names = image_iter(src_path)
    imgs = []

    # Get an informative name for the progress bar description
    dataset_name_desc = os.path.basename(os.path.normpath(src_path))

    for p in tqdm(img_paths, desc=f"Loading images from {dataset_name_desc}"):
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
    """Extracts and saves embeddings without creating triplet files."""
    print(f"\nExtracting embeddings for dataset: {os.path.basename(src_path)}")
    batchsize = 32
    backbone = load_model(model_name, model_path, device)

    file_names, imgs_tensor = load_images_for_embeddings(src_path)
    if imgs_tensor is None:
        print(f"No images found in {src_path}.")
        return

    imgs_tensor = torch.split(imgs_tensor, batchsize)
    embs = []
    for batch in tqdm(imgs_tensor, desc="Computing embeddings"):
        with torch.no_grad():
            batch = batch.to(device)
            _embedding = backbone(batch)
            if isinstance(_embedding, tuple):  # Handle models that return tuples
                _embedding = _embedding[0]
            embs.extend(_embedding.cpu().numpy())

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    np.save(dest_path + "_embs", np.array(embs))
    np.save(dest_path + "_ids", np.array(file_names))
    print("Embeddings saved successfully.")


def get_reference(rid, id_list):
    for idx, i in enumerate(id_list):
        if i == rid:
            return idx
    return None


def eval_morph(triple, morphs, morph_id, ref, ref_ids, thresh=0):
    mated, non_mated, iapar_match, iapar_non_match = [], [], [], []

    for i, (mid, id1, id2) in enumerate(tqdm(triple, desc="Evaluating morphs")):
        morph_idx = get_reference(mid, morph_id)
        emb1_idx = get_reference(id1, ref_ids)
        emb2_idx = get_reference(id2, ref_ids)

        if morph_idx is None or emb1_idx is None or emb2_idx is None:
            non_mated.append(None)
            continue

        morph, emb1, emb2 = morphs[morph_idx], ref[emb1_idx], ref[emb2_idx]
        cos1, cos2 = cos_sim(emb1, morph), cos_sim(emb2, morph)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Morphing Evaluation Toolkit using Official Triplet Files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ElasticFaceArc",
        choices=["ElasticFaceArc", "ElasticFaceCos", "CurricularFace"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the morphing dataset folder (e.g., OpenCV_aligned).",
    )
    parser.add_argument("--fmr", type=float, default=0.01, choices=[0.01, 0.001])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    REF_DATASET = "BonaFide_aligned"

    # --- Construct paths ---
    model_path = f"models/{args.model}.pth"
    morph_dataset_path = f"SYN-MAD22/{args.dataset}/"
    ref_dataset_path = f"SYN-MAD22/{REF_DATASET}/"

    morph_emb_dest_path = f"embeddings/{args.model}_{args.dataset}"
    ref_emb_dest_path = f"embeddings/{args.model}_{REF_DATASET}"

    # Use the official triplet file, removing '_aligned' from the dataset name to match file names
    triplet_dataset_name = args.dataset.replace("_aligned", "")
    triplet_file_path = f"triplets/SYN-MAD22_{triplet_dataset_name}_triples.txt"

    # --- Check for triplet file first ---
    if not os.path.exists(triplet_file_path):
        raise FileNotFoundError(
            f"Official triplet file not found! Please place it at: {triplet_file_path}"
        )

    # --- Select threshold ---
    thresholds = {
        "ElasticFaceArc": {0.01: 0.21402553, 0.001: 0.29908442},
        "ElasticFaceCos": {0.01: 0.18321043, 0.001: 0.26074028},
        "CurricularFace": {0.01: 0.1901376, 0.001: 0.26934636},
    }
    thresh = thresholds[args.model][args.fmr]

    # --- Generate missing embeddings ---
    print("--- Checking for necessary embedding files ---")
    if not os.path.exists(ref_emb_dest_path + "_embs.npy"):
        extract_embeddings(
            args.model, model_path, ref_dataset_path, ref_emb_dest_path, device
        )

    if not os.path.exists(morph_emb_dest_path + "_embs.npy"):
        extract_embeddings(
            args.model, model_path, morph_dataset_path, morph_emb_dest_path, device
        )

    # --- Load all data and evaluate ---
    print("\n--- All necessary files are present. Starting evaluation. ---")
    morphs, morph_ids = np.load(morph_emb_dest_path + "_embs.npy"), np.load(
        morph_emb_dest_path + "_ids.npy"
    )
    ref_embs, ref_ids = np.load(ref_emb_dest_path + "_embs.npy"), np.load(
        ref_emb_dest_path + "_ids.npy"
    )
    triplets = np.genfromtxt(triplet_file_path, delimiter="\t", dtype=str)

    mmpmr, iapar = eval_morph(triplets, morphs, morph_ids, ref_embs, ref_ids, thresh)

    print("\n" + "=" * 40 + "\n          EVALUATION RESULTS\n" + "=" * 40)
    print(f"Model: {args.model}\nMorphing Dataset: {args.dataset}")
    print(f"Threshold (FNMR@FMR={args.fmr*100}%): {thresh}")
    print("-" * 40 + f"\nMMPMR: {mmpmr:.4f}\nIAPAR: {iapar:.4f}\n" + "=" * 40)
