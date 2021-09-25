import torch
import os
import shutil
import zipfile
import urllib.request
from torchvision import datasets, transforms
from kaggle.api.kaggle_api_extended import KaggleApi


def download_coco_dataset():
    """Download the COCO dataset"""
    # Create file structure
    os.makedirs(os.path.join("data", "coco", "train"), exist_ok=True)
    os.makedirs(os.path.join("data", "coco", "dev"), exist_ok=True)
    os.makedirs(os.path.join("data", "coco", "test"), exist_ok=True)
    # Download the train, dev and test datasets
    print("Downloading COCO dataset.")
    url = "http://images.cocodataset.org/zips/train2014.zip"
    print("Downloading " + url)
    urllib.request.urlretrieve(url, os.path.join("data", "coco", "train2014.zip"))
    url = "http://images.cocodataset.org/zips/val2014.zip"
    print("Downloading " + url)
    urllib.request.urlretrieve(url, os.path.join("data", "coco", "val2014.zip"))
    url = "http://images.cocodataset.org/zips/test2014.zip"
    print("Downloading " + url)
    urllib.request.urlretrieve(url, os.path.join("data", "coco", "test2014.zip"))
    print("Done downloading COCO dataset.")
    # Unzip the files
    print("Extracting COCO dataset.")
    # Extract Train dataset
    zip_ref = zipfile.ZipFile(os.path.join("data", "coco", "train2014.zip", "r"))
    zip_ref.extractall(os.path.join("data", "coco"))
    shutil.move(
        os.path.join("data", "coco", "train2014"),
        os.path.join("data", "coco", "train", "dummy"),
    )
    # Extract Validation dataset
    zip_ref = zipfile.ZipFile(os.path.join("data", "coco", "val2014.zip", "r"))
    zip_ref.extractall(os.path.join("data", "coco"))
    shutil.move(
        os.path.join("data", "coco", "val2014"),
        os.path.join("data", "coco", "dev", "dummy"),
    )
    # Extract Test dataset
    zip_ref = zipfile.ZipFile(os.path.join("data", "coco", "test2014.zip", "r"))
    zip_ref.extractall(os.path.join("data", "coco"))
    shutil.move(
        os.path.join("data", "coco", "test2014"),
        os.path.join("data", "coco", "test", "dummy"),
    )
    print("Done extracting COCO dataset.")


def download_wikiart_dataset():
    link = "http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip"
    print("Downloading WikiArt dataset (25GB).")
    urllib.request.urlretrieve(link, os.path.join("data", "wikiart.zip"))
    print("Done downloading Wikiart dataset.")
    print("Extracting Wikiart dataset.")
    zip_ref = zipfile.ZipFile(os.path.join("data", "wikiart.zip"))
    zip_ref.extractall(os.path.join("data"))
    print("Done extracting Wikiart dataset.")


def download_abstract_art_dataset():
    api = KaggleApi()
    api.authenticate()
    print("Downloading Abstract Art Gallery dataset (700MB).")
    dataset_path = os.path.join("data", "abstract_art")
    os.makedirs(dataset_path, exist_ok=True)
    api.dataset_download_files("bryanb/abstract-art-gallery", dataset_path)
    print("Done downloading Abstract Art Gallery dataset.")
    print("Extracting Abstract Art Gallery dataset.")
    zip_ref = zipfile.ZipFile(os.path.join(dataset_path, "abstract-art-gallery.zip"))
    zip_ref.extractall(dataset_path)
    print("Done extracting Abstract Art Gallery dataset.")


def get_coco_dataloader(batch_size=32):
    path = os.path.join("data", "coco", "train2014")
    return _get_directory_dataloader(path, batch_size)


def get_abstract_art_dataloader(batch_size=32):
    path = os.path.join("data", "abstract_art")
    return _get_directory_dataloader(path, batch_size)


def _get_directory_dataloader(
    directory, batch_size, resize=True, augment=True, drop_last=True, shuffle=True
):

    transformations = []
    if resize:
        transformations.extend(
            [transforms.Resize(255), transforms.RandomSizedCrop(224)]
        )
    if augment:
        transformations.extend([transforms.RandomHorizontalFlip()])
    transformations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform = transforms.Compose(transformations)

    dataset = datasets.ImageFolder(directory, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return dataloader
