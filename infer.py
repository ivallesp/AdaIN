import argparse
import itertools
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.data import _get_transformations, load_and_transform
from src.model import AdaInStyleTransfer


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--content-dir",
        required=True,
        type=str,
        help="Directory where the content images to use for inference are stored. They "
        "should be inside a 'dummy' folder that mocks a class, to be read by the "
        "Pytorch dataloader.",
    )
    argparser.add_argument(
        "--style-dir",
        required=True,
        type=str,
        help="Directory where the style images to use for inference are stored. They "
        "should be inside a 'dummy' folder that mocks a class, to be read by the "
        "Pytorch dataloader.",
    )
    argparser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where the output images will be stored.",
    )
    argparser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        help="Directory where the model checkpoints in format 'decoder_{epoch}.pt' "
        "are stored.",
    )
    argparser.add_argument(
        "--epoch",
        required=True,
        type=int,
        help="Epoch to use in the inference process",
    )
    argparser.add_argument(
        "--method",
        required=True,
        type=str,
        help="Either 'cartesian' or 'match'. The first one will transfer the style "
        "of all the images in the style-dir into all the images in the content dir. "
        "The second one will transfer the style one by one, matching the file names.",
    )
    argparser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Whether to use the GPU or not.",
        default=False,
    )

    return argparser.parse_args()


def denorm(pixvals, percentiles=[0, 100]):
    minval = np.percentile(pixvals, percentiles[0])
    maxval = np.percentile(pixvals, percentiles[1])
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = (pixvals - minval) / (maxval - minval)
    pixvals = np.clip(pixvals, 0, 1)
    return pixvals


def _find_pairs_of_matching_filenames(content_images, style_images):
    """
    Given a list of content images and a list of style images, find all the
    pairs of images that have the same name.
    """
    pairs = []
    for content_path in content_images:
        content_name = os.path.splitext(os.path.split(content_path)[-1])[0]
        for style_path in style_images:
            style_name = os.path.splitext(os.path.split(style_path)[-1])[0]
            if content_name == style_name:
                pairs.append((content_path, style_path))
    return pairs


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all jpgs and pngs recursively in args.content_dir
    content_images = glob(os.path.join(args.content_dir, "**/*.jpg"), recursive=True)
    content_images += glob(os.path.join(args.content_dir, "**/*.jpeg"), recursive=True)
    content_images += glob(os.path.join(args.content_dir, "**/*.png"), recursive=True)

    # Get all jpgs and pngs recursively in args.content_dir
    style_images = glob(os.path.join(args.style_dir, "**/*.jpg"), recursive=True)
    style_images += glob(os.path.join(args.style_dir, "**/*.jpeg"), recursive=True)
    style_images += glob(os.path.join(args.style_dir, "**/*.png"), recursive=True)

    transform = _get_transformations(resize=True, augment=False)

    device = torch.device("cuda" if args.use_gpu else "cpu")

    # Load the model
    model = AdaInStyleTransfer()
    model.decoder.load_state_dict(
        torch.load(
            os.path.join(args.model_dir, f"decoder_{args.epoch:03d}.pt"),
            map_location=device,
        )
    )
    model.vae.load_state_dict(
        torch.load(
            os.path.join(args.model_dir, f"vae_{args.epoch:03d}.pt"),
            map_location=device,
        )
    )

    # Move the model to the device
    model.encoder.to(device)
    model.decoder.to(device)

    # Build the data loaders
    if args.method == "cartesian":
        tuples = list(itertools.product(content_images, style_images))
    elif args.method == "match":
        tuples = _find_pairs_of_matching_filenames(content_images, style_images)
    else:
        raise NotImplementedError(
            f"Method {args.method} not implemented. "
            "Only 'cartesian' and 'match' are supported."
        )

    # Transfer the style of the required images
    for content_path, style_path in tqdm(tuples):
        output_filename = (
            os.path.splitext(os.path.split(content_path)[-1])[0]
            + "___"
            + os.path.splitext(os.path.split(style_path)[-1])[0]
            + ".jpg"
        )
        output_path = os.path.join(args.output_dir, output_filename)

        # Load the images
        content = load_and_transform(content_path, transform)[None].to(device)
        style = load_and_transform(style_path, transform)[None].to(device)

        # Run the model
        output, _, _, _, _, _ = model(content, style)

        # Postprocess the output
        output = output.squeeze(0).detach().cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = denorm(output, percentiles=[1, 98])

        # Save the image
        plt.imsave(output_path, output)


if __name__ == "__main__":
    main()
