import argparse
from tqdm import tqdm
import os
import torch
from main import image_iter
import numpy as np
import csv
import network
from sklearn.metrics import roc_curve
from dataset_gen import TestDataset
from torch.utils.data import DataLoader

# Code partially taken from https://github.com/meilfang/SPL-MAD/blob/main/


def load_detector_model(model_path):
    # Load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(
        network.AEMAD(in_channels=3, features_root=64))
    model.load_state_dict(torch.load(
        model_path, map_location=device))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def load_dataset(dataset_path):
    # Load dataset
    img_paths, file_names = image_iter(dataset_path)
    dataset = TestDataset(
        paths=img_paths,
    )
    return dataset, file_names


def get_eer_threhold(fpr, tpr, threshold):
    """
    Calculates the Equal Error Rate (EER) and its corresponding threshold.
    Handles cases where all values are NaN by returning default values.
    """
    differ_tpr_fpr_1 = tpr + fpr - 1.0

    if np.all(np.isnan(differ_tpr_fpr_1)):
        # Return default values if all are NaN
        return float('nan'), float('nan'), -1

    right_index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    eer = fpr[right_index]

    return eer, best_th, right_index


def get_performance(prediction_scores, gt_labels, pos_label=1, verbose=True):
    """
    Computes the EER and its threshold for the given prediction scores and ground truth labels.
    """
    fpr, tpr, threshold = roc_curve(
        gt_labels, prediction_scores, pos_label=pos_label)
    eer, eer_th, _ = get_eer_threhold(fpr, tpr, threshold)
    # test_auc = auc(fpr, tpr)

    if verbose is True:
        print(f'EER is {eer}, threshold is {eer_th}')

    return eer, eer_th


def test_dataset(dataset, model_path, output_path):
    # Load model
    detector = load_detector_model(model_path)

    test_loader = DataLoader(dataset, batch_size=32,
                             shuffle=False, num_workers=8, pin_memory=True)

    print('Number of test images:', len(test_loader.dataset))

    mse_criterion = torch.nn.MSELoss(reduction='none').cuda(
    ) if torch.cuda.is_available() else torch.nn.MSELoss(reduction='none').cpu()

    test_scores, gt_labels, test_scores_dict = [], [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            if torch.cuda.is_available():
                raw, labels, img_ids = data['images'].cuda(
                ), data['labels'], data['img_path']
            else:
                raw, labels, img_ids = data['images'].cpu(
                ), data['labels'], data['img_path']
            _, output_raw = detector(raw)

            scores = mse_criterion(output_raw, raw).cpu().data.numpy()
            scores = np.sum(np.sum(np.sum(scores, axis=3), axis=2), axis=1)
            test_scores.extend(scores)
            gt_labels.extend((1 - labels.data.numpy()))
            for j in range(labels.shape[0]):
                l = 'attack' if labels[j].detach().numpy() == 1 else 'bonafide'
                test_scores_dict.append(
                    {'img_path': img_ids[j], 'labels': l, 'prediction_score': float(scores[j])})

    eer, eer_th = get_performance(test_scores, gt_labels)
    print('Test EER:', eer*100)

    with open(output_path, mode='w') as csv_file:
        fieldnames = ['img_path', 'labels', 'prediction_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for d in test_scores_dict:
            writer.writerow(d)
        print('Prediction scores write done in', output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Face Morphing Detector"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=False,
        help="Configure your output path here."
    )

    args = parser.parse_args()
    output_dir = args.output_directory if args.output_directory else ""

    # Load the model and dataset
    model_path = "models/casia_smdd.pth"  # detector model path
    dataset = args.dataset

    output_path = os.path.join(
        output_dir, f"{dataset}_morphing_detection_results.csv")

    dataset_path = f"SYN-MAD22/{dataset}"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    # create the dataset
    dataset, file_name = load_dataset(dataset_path)

    test_dataset(dataset, model_path, output_path)
