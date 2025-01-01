from PIL import Image
import math
import numpy as np
import random
import os
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class NaturalDisasterDataset(Dataset):
    def __init__(self, root_img, root_gt, mean, std, phase="train"):
        super().__init__()
        self.root_img = root_img  # Path to images
        self.root_gt = root_gt    # Path to ground truth masks
        self.phase = phase
        self.mean, self.std = mean, std

        if self.phase in ["train", "val"]:
            if not os.path.exists(self.root_img) or not os.path.exists(self.root_gt):
                raise FileNotFoundError(
                    f"Paths {self.root_img} or {self.root_gt} do not exist."
                )
            self.image_names = os.listdir(self.root_img)
        else:
            if not os.path.exists(self.root_img):
                raise FileNotFoundError(f"Root directory {self.root_img} does not exist.")
            self.image_names = os.listdir(self.root_img)

    def __getitem__(self, index):
        try:
            x_path = os.path.join(self.root_img, self.image_names[index])
            y_path = os.path.join(self.root_gt, self.image_names[index])
            x = Image.open(x_path)
            y = Image.open(y_path).convert("L")
        except Exception as e:
            print(f"Error loading files: {x_path}, {y_path}. Exception: {e}")
            raise e
        return self.apply_transforms(x, y)


    def __len__(self):
        return len(self.image_names)


    def apply_transforms(self, x, y=None):
        if self.phase == "train":
            x_transforms = transforms.Compose(
                [
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.GaussianBlur(3, sigma=(0.1, 10)),
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )

            y_transforms = transforms.Compose(
                [
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

        else:
            x_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            y_transforms = None

        # Fix for random seed range
        seed = np.random.randint(0, 2147483647)

        random.seed(seed)
        torch.manual_seed(seed)
        x = x_transforms(x)

        if self.phase in ["train", "val"]:
            if y_transforms:
                random.seed(seed)
                torch.manual_seed(seed)
                y = y_transforms(y)

            y = torch.Tensor(np.array(y)).squeeze().long()
            return x, y
        else:
            return x


def get_train_data_loaders(img_dir, gt_dir, validation_split, batch_size):
    """
    Returns training and validation data loaders.

    Args:
        img_dir (str): Path to the image directory.
        gt_dir (str): Path to the ground truth masks directory.
        validation_split (float): Fraction of data to use for validation.
        batch_size (int): Batch size.

    Returns:
        tuple: Training and validation data loaders.
    """
    # num_workers=os.cpu_count()
    num_workers=10

    images_ds = NaturalDisasterDataset(
        root_img=img_dir,
        root_gt=gt_dir,
        mean=[0.46077183, 0.45584197, 0.41929824],
        std=[0.18551224, 0.17078055, 0.17699541],
        phase="train",
    )

    ds_lengths = [
        math.floor((1 - validation_split) * len(images_ds)),
        math.ceil((validation_split * len(images_ds))),
    ]
    train_ds, val_ds = random_split(images_ds, ds_lengths)

    train_data_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, timeout=60, persistent_workers=False
    )
    val_data_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, timeout=60, persistent_workers=False
    )

    return train_data_loader, val_data_loader



def get_test_data_loaders(root_dir):
    images_ds = NaturalDisasterDataset(
        root=root_dir,
        mean=[0.46077183, 0.45584197, 0.41929824],
        std=[0.18551224, 0.17078055, 0.17699541],
        phase="test",
    )
    test_data_loader = DataLoader(images_ds, batch_size=1, shuffle=False, num_workers=1)
    return test_data_loader