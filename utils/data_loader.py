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
    def __init__(self, root, mean, std, phase="train"):
        super().__init__()

        self.root = root
        self.phase = phase

        self.mean, self.std = mean, std

        if self.phase in ["train", "val"]:
            self.image_names = os.listdir(os.path.join(self.root, "images"))
        else:
            self.image_names = os.listdir(os.path.join(self.root))

    def __getitem__(self, index):
        if self.phase in ["train", "val"]:
            x_path = os.path.join(self.root, "images", self.image_names[index])
            y_path = os.path.join(self.root, "masks", self.image_names[index])

            x = Image.open(x_path)
            y = Image.open(y_path).convert("L")

            return self.apply_transforms(x, y)

        else:
            x_path = os.path.join(self.root, self.image_names[index])
            x = Image.open(x_path)

            return self.apply_transforms(x)

    def __len__(self):
        if self.phase in ["train", "val"]:
            return len(os.listdir(os.path.join(self.root, "images")))
        else:
            return len(os.listdir(os.path.join(self.root)))

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

        seed = np.random.randint(23412304981723094)

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


def get_train_data_loaders(root_dir, validation_split, batch_size):
    images_ds = NaturalDisasterDataset(
        root=root_dir,
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
        train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
    )
    val_data_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()
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