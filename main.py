import torch
import numpy as np
from src.model import AdaInStyleTransfer
from src.data import get_coco_dataloader, get_abstract_art_dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main():
    # Get data loaders
    content_dataloader = get_coco_dataloader()
    style_dataloader = get_abstract_art_dataloader()

    # Create model and push to CUDA
    model = AdaInStyleTransfer()
    model.decoder = model.decoder.cuda()
    model.encoder = model.encoder.cuda()

    # Create tensorboard writer
    writer = SummaryWriter(log_dir="tb_logs/test")

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

            total_loss = (1.0 * content_loss + 10.0 * style_loss) / 11.0

            # Backward prop
            total_loss.backward()
            writer.add_scalar("Content Loss", content_loss, global_step)
            writer.add_scalar("Style Loss", style_loss, global_step)
            writer.add_scalar("Total Loss", total_loss, global_step)

            # Train step
            optimizer.step()
            global_step += 1

        # Push sample images to tensorboard
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


if __name__ == "__main__":
    main()
