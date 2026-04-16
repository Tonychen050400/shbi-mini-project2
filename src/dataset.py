import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import numpy as np


VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
NUM_CLASSES = len(VOC_CLASSES)

VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(augment=False, normalize=False, img_size=256):
    img_transforms = [transforms.Resize((img_size, img_size))]
    if augment:
        img_transforms += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ]
    img_transforms.append(transforms.ToTensor())
    if normalize:
        img_transforms.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    transform_target = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.PILToTensor(),
    ])
    return transforms.Compose(img_transforms), transform_target


class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set="train", augment=False, normalize=False, img_size=256):
        transform_img, transform_target = get_transforms(
            augment=False, normalize=normalize, img_size=img_size
        )
        self.dataset = VOCSegmentation(
            root=root, year="2007", image_set=image_set,
            download=False, transform=transform_img,
            target_transform=transform_target,
        )
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        mask = mask.squeeze(0).long()
        mask[mask > 20] = 255
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-1])
                mask = torch.flip(mask, [-1])
        return image, mask


def get_dataloaders(root, batch_size=4, augment=False, normalize=False,
                    img_size=256, num_workers=2):
    train_ds = VOCSegDataset(root, image_set="train", augment=augment,
                             normalize=normalize, img_size=img_size)
    val_ds = VOCSegDataset(root, image_set="val", augment=False,
                           normalize=normalize, img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
