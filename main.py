# Main
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from backbones.curr_resnet import curr_iresnet100
from backbones.elastic_resnet import elastic_iresnet100

# SET THIS VARIABLE: 'model'
# Model selection
# Options: 'ElasticFaceArc', 'ElasticFaceCos', 'CurricularFace'
model = 'ElasticFaceCos'
model_path = f"models/{model}.pth"  # Path to the model .pth file

# SET THIS VARIABLE: 'morphing_dataset'
# Which morphing dataset (i.e. images) to use
# Options: 'BonaFide_aligned' (for benign samples), FaceMorpher_aligned and others in SYN-MAD directory!
morphing_dataset = 'FaceMorpher_aligned'
# Path to the morphing dataset
morphing_path = f"SYN-MAD22/{morphing_dataset}/"

# SET THIS VARIABLE: dest_path, you only need to change the 'embeddings' folder name
# in case your working directory is different
# Destination path for embeddings
dest_path = f"embeddings/{model}_{morphing_dataset}"


# Thresholds
thr_arc_base_100 = 0.21402553   # ElasticFace-Arc FNMR@FMR = 1%
thr_cos_base_100 = 0.18321043   # ElasticFace-Cos FNMR@FMR = 1%
thr_cur_base_100 = 0.1901376    # CurricularFace FNMR@FMR = 1%

thr_arc_base_1000 = 0.29908442  # ElasticFace-Arc FNMR@FMR = 0.1%
thr_cos_base_1000 = 0.26074028  # ElasticFace-Cos FNMR@FMR = 0.1%
thr_cur_base_1000 = 0.26934636  # CurricularFace FNMR@FMR = 0.1%

# SET THIS VARIABLE: fnmr_at_fmr
# Options: 0.01 (FNMR@FMR = 1%) or 0.001 (FNMR@FMR = 0.1%)
fnmr_at_fmr = 0.01
thresh = thr_arc_base_100 if model == 'ElasticFaceArc' and fnmr_at_fmr == 0.01 else \
    thr_cos_base_100 if model == 'ElasticFaceCos' and fnmr_at_fmr == 0.01 else \
    thr_cur_base_100 if model == 'CurricularFace' and fnmr_at_fmr == 0.01 else \
    thr_arc_base_1000 if model == 'ElasticFaceArc' and fnmr_at_fmr == 0.001 else \
    thr_cos_base_1000 if model == 'ElasticFaceCos' and fnmr_at_fmr == 0.001 else \
    thr_cur_base_1000 if model == 'CurricularFace' and fnmr_at_fmr == 0.001 else \
    None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Calculates cosine similarity between vector a and b


def cos_sim(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load FR model


def load_model(model, model_path, device):
    """Load model from path_model."""
    if not os.path.exists(model_path):
        raise Exception("Model file does not exist!", model_path)
    if model in ['ElasticFaceArc', 'ElasticFaceCos']:
        backbone = elastic_iresnet100(num_features=512).to(device)
        backbone.load_state_dict(torch.load(model_path, map_location=device))
    elif model == 'CurricularFace':
        backbone = curr_iresnet100().to(device)
        backbone.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise Exception("Cannot load unknown model:", model)

    backbone.eval()
    return backbone

# iterates over images


def image_iter(path):
    """Return image paths in filepath."""
    image_paths = []
    file_names = []

    for path, subdirs, files in os.walk(path):
        for name in files:
            image_paths.append(os.path.join(path, name))
            file_names.append(name)
    image_paths.sort()
    return image_paths, file_names

# Loads and normalizes images


def load_imgs(src_path, transfrom, tensor):
    img_paths, file_names = image_iter(src_path)

    imgs = []

    # Triplet file name
    triplet_file_name = f'triplets/SYN-MAD22_{morphing_dataset}_triples_selfmade.txt'

    for p in tqdm(img_paths, desc="Load images"):

        # Create triplets
        img_name = p.split("/")[-1]
        if "vs" in img_name:
            morph_img = img_name
            benign_img1 = img_name.split("-")[0] + ".jpg"
            benign_img2 = img_name.split("-")[2].split(".")[0] + ".jpg"
            # write this triplet to a file

            with open(triplet_file_name, 'a') as f:
                f.write(f"{morph_img}\t{benign_img1}\t{benign_img2}\n")
        img = cv2.imread(p)
        if img is not None:
            if transfrom:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))  # (channel, height, width)
                img = np.asarray([img], dtype="float32")
                img = ((img / 255) - 0.5) / 0.5
            if tensor:
                img = torch.tensor(img, device=device)
            imgs.append(img)

    if tensor:
        imgs = torch.cat(imgs, dim=0)
    return file_names, imgs


# Extract embeddings and save them to a file

def extract_embeddings(model_path="where_is_the_model_pth",
                       src_path="where_to_load_images_for_embeddings",
                       dest_path="where_to_save_embeddings"):
    """Extract embeddings from images using a specified model."""

    # prepare inference
    batchsize = 32
    backbone = load_model(model, model_path, device)
    backbone.to(device)

    # Load Images and get file_names
    img_paths, imgs_tensor = load_imgs(
        src_path, transfrom=True, tensor=True)
    imgs_tensor = torch.split(imgs_tensor, batchsize)

    # compute embeddings
    embs = []
    for batch in tqdm(imgs_tensor, desc="Compute embeddings"):
        batch.to(device)

        if model in ['CurricularFace', 'AdaFace']:
            _embedding = backbone(batch)[0]
        else:  # self.name in ['ElasticArcface', 'ElasticCosface']:
            _embedding = backbone(batch)

        _embedding = _embedding.detach().cpu().numpy()
        if _embedding.shape[0] == 1:
            embs.append(_embedding)
        else:
            embs.extend(_embedding)

    # save embeddings
    res = np.array(embs)
    np.save(dest_path+"_embs", res)
    np.save(dest_path+"_ids", img_paths)


# Example Evaluation

def get_reference(rid, id_list):
    for idx, i in enumerate(id_list):
        if i == rid:

            return idx


def eval_morph(triple, morphs, morph_id, ref, ref_ids, thresh=0):
    scores = []
    mated, non_mated, iapar_match, iapar_non_match = [], [], [], []

    for i, (mid, id1, id2) in enumerate(triple):
        morph = morphs[get_reference(mid, morph_id)]
        emb1 = ref[get_reference(id1, ref_ids)]
        emb2 = ref[get_reference(id2, ref_ids)]

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

        scores.append(cos1)
        scores.append(cos2)

    mmpmr = len(mated) / (len(mated) + len(non_mated))
    iapar = len(iapar_match) / (len(iapar_match) + len(iapar_non_match))
    return mmpmr, iapar


if __name__ == "__main__":
    triplets_file = f'triplets/SYN-MAD22_{morphing_dataset}_triples_selfmade.txt'
    # Extract embeddings only if they do not exist
    if not os.path.exists(dest_path + "_embs.npy") or not os.path.exists(dest_path + "_ids.npy") or not os.path.exists(triplets_file):
        extract_embeddings(model_path=model_path,
                           src_path=morphing_path,
                           dest_path=dest_path)

    # Load embeddings
    morphs = np.load(dest_path + "_embs.npy")
    morph_ids = np.load(dest_path + "_ids.npy")

    # Adjust this in case you want to use the triplet files that were provided
    triplets = np.genfromtxt(
        triplets_file,
        delimiter='\t',
        dtype=str)

    # Evaluate with a threshold
    mmpmr, iapar = eval_morph(
        triple=triplets, morphs=morphs, morph_id=morph_ids, ref=morphs, ref_ids=morph_ids, thresh=thresh)
    # Print results
    print(
        f"Model: {model}, Morphing Dataset: {morphing_dataset}, Threshold: {thresh}, FNMR@FMR: {fnmr_at_fmr}")
    print(f"MMPMR: {mmpmr}, IAPAR: {iapar}")
