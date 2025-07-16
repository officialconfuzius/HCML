# Code taken from https://github.com/meilfang/SPL-MAD/blob/main/

import cv2

import torch
from torch.utils.data import Dataset

import albumentations
from albumentations.pytorch import ToTensorV2

PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]


class TestDataset(Dataset):

    def __init__(self, paths, labels, input_shape=(112, 112)):
        # self.image_dir = image_dir
        self.paths = paths
        self.labels = labels
        self.composed_transformations = albumentations.Compose([
            albumentations.SmallestMaxSize(max_size=input_shape[0]),
            albumentations.CenterCrop(
                height=input_shape[0], width=input_shape[0]),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)

        image = self.composed_transformations(image=image)['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.float),
            "img_path": img_path
        }
