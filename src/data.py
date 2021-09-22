import torch
import os
import zipfile
import urllib.request
import torchvision
from torchvision import datasets, transforms


def download_coco_dataset():
    """Download the COCO dataset"""

    if not os.path.exists("data/coco"):
        os.makedirs("data/coco")
    url = "http://images.cocodataset.org/zips/train2014.zip"
    print("Downloading " + url)
    urllib.request.urlretrieve(url, "data/coco/train2014.zip")
    url = "http://images.cocodataset.org/zips/val2014.zip"
    print("Downloading " + url)
    urllib.request.urlretrieve(url, "data/coco/val2014.zip")
    url = "http://images.cocodataset.org/zips/test2014.zip"
    print("Downloading " + url)
    urllib.request.urlretrieve(url, "data/coco/test2014.zip")
    print("Done")


def unzip_coco_dataset():
    """Unzip the COCO dataset"""
    # Check if the zip files exist
    if (
        not os.path.exists("data/coco/train2014.zip")
        or not os.path.exists("data/coco/val2014.zip")
        or not os.path.exists("data/coco/test2014.zip")
    ):
        print("COCO dataset not found, downloading...")
        download_coco_dataset()
    zip_ref = zipfile.ZipFile("data/coco/train2014.zip", "r")
    zip_ref.extractall("data/coco")
    zip_ref = zipfile.ZipFile("data/coco/val2014.zip", "r")
    zip_ref.extractall("data/coco")
    zip_ref = zipfile.ZipFile("data/coco/test2014.zip", "r")
    zip_ref.extractall("data/coco")
    print("Done")


def get_coco_dataloader(batch_size=32):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.ImageFolder("./data/coco/train2014/", transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return dataloader


def get_abstract_art_dataloader(batch_size=32):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.ImageFolder("./data/abstract_art", transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return dataloader
