import argparse
import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import get_dataloader
from src.model import AdaInStyleTransfer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-dataset", type=str, default="coco")
    parser.add_argument("--style-dataset", type=str, default="abstract")
    parser.add_argument("--alias", type=str, required=True, help="Alias for model")
    parser.add_argument("--n-epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--style_loss_weight", type=float, default=10.0, help="Weight for style loss"
    )
    return parser.parse_args()


def main():
    # Get input args
    args = parse_args()

    # Get data loaders
    content_dataloader = get_dataloader(
        args.content_dataset,
        batch_size=args.batch_size,
    )
    style_dataloader = get_dataloader(
        args.style_dataset,
        batch_size=args.batch_size,
    )

    # Create model and push to CUDA
    model = AdaInStyleTransfer()
    model.decoder = model.decoder.cuda()
    model.encoder = model.encoder.cuda()

    # Create output model directory
    model_dir = os.path.join("models", args.alias)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join("tb_logs", args.alias))

    # Deine optimizer
    optimizer = torch.optim.Adam(params=model.decoder.parameters(), lr=0.0001)

    # Train model
    style_iter = iter(style_dataloader)
    global_step = 0

    # Training loop
    for epoch in tqdm(range(1000)):
        for content, _ in content_dataloader:
            # Sample a style
            try:
                style, _ = next(style_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                style_iter = iter(style_dataloader)
                style, _ = next(style_iter)
            # Zero the gradients
            optimizer.zero_grad()

            # Get the loss
            content_loss, style_loss = model.calculate_losses(
                content.cuda(), style.cuda()
            )

            total_loss = (1.0 * content_loss + args.style_loss_weight * style_loss) / (
                1.0 + args.style_loss_weight
            )

            # Backward prop
            total_loss.backward()
            writer.add_scalar("Content Loss", content_loss, global_step)
            writer.add_scalar("Style Loss", style_loss, global_step)
            writer.add_scalar("Total Loss", total_loss, global_step)

            # Train step
            optimizer.step()
            global_step += 1

        # Push traning sample images to tensorboard
        samples = model.transfer_style(content.cuda(), style.cuda())[0].cpu().detach()

        content *= torch.tensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])
        content += torch.tensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
        writer.add_images("content", content, epoch)

        style *= torch.tensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])
        style += torch.tensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
        writer.add_images("style", style, epoch)

        samples *= torch.tensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])
        samples += torch.tensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
        writer.add_images("samples", samples, epoch)

        # Save decoder model
        torch.save(
            model.decoder.state_dict(),
            os.path.join(model_dir, f"decoder_{epoch:03}.pt"),
        )


if __name__ == "__main__":
    main()
