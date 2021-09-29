from torchvision import models
from torch import nn
import torch
import math

# First ReLU after MaxPooling
LAYERS_STYLE_LOSS = [1, 6, 11, 20]


class AdaInStyleTransfer:
    def __init__(self):
        self.encoder = build_encoder()
        self.decoder = build_decoder()
        self.vae = build_vae()
        self.optimizer = torch.optim.Adam(
            params=list(self.decoder.parameters()) + list(self.vae.parameters()),
            lr=0.0001,
        )
        self.global_step = 0

    def transfer_style(self, content, style, alpha=1.0):
        # Encode the content images using the pretrained model
        content_code, _ = self.forward_encoder(content)
        # Encode the style images using the pretrained model and Collect the outputs
        # of the intermediate layers for computing the style loss later

        style_code, style_loss_components_target = self.forward_encoder(style)

        # ADAIN: Normalize the content code to follow style stats
        adain_output = adain(content_code, style_code)

        # Newstyle code
        newstyle_code = (alpha) * adain_output + (1 - alpha) * content_code

        # Normalize code
        newstyle_code = (newstyle_code - newstyle_code.mean(dim=[1,2,3], keepdims=True)) / newstyle_code.std(dim=[1,2,3], keepdims=True)

        # VAE: Estimate a distribution for the newstyle code
        vae_projection = self.vae(newstyle_code)
        newstyle_mu, newstyle_logvar = torch.split(vae_projection, 512, dim=1)

        # VAE: Sample from the estimated distribution
        newstyle_z = newstyle_mu + torch.exp(newstyle_logvar) * torch.randn_like(
            newstyle_logvar
        ) + newstyle_code

        # Bring the normalized code to the image domain
        new_style = self.decoder(newstyle_z)
        return (
            new_style,
            style_loss_components_target,
            adain_output,
            newstyle_mu,
            newstyle_logvar,
            newstyle_z,
        )

    def calculate_losses(self, content, style):
        (
            new_style,
            style_loss_components_target,
            adain_output,
            newstyle_mu,
            newstyle_logvar,
            newstyle_z,
        ) = self.transfer_style(content, style)
        # Encode the generated image with the transferred style and  collect the
        # outputs of the intermediate layers for computint the style loss later
        newstyle_code, style_loss_components_gen = self.forward_encoder(new_style)

        # Compute the content loss
        content_loss = nn.MSELoss()(adain_output, newstyle_code)

        # Compute the style loss for the selected layers
        style_loss = 0
        for layer in range(len(LAYERS_STYLE_LOSS)):
            target_act = style_loss_components_target[layer]
            gen_act = style_loss_components_gen[layer]

            style_loss += nn.MSELoss()(
                gen_act.mean(axis=[-1, -2]), target_act.mean(axis=[-1, -2])
            )
            style_loss += nn.MSELoss()(
                gen_act.std(axis=[-1, -2]), target_act.std(axis=[-1, -2])
            )
        style_loss /= len(LAYERS_STYLE_LOSS)

        # VAE loss
        kld_loss = calculate_kld_loss(newstyle_mu, newstyle_logvar)

        return content_loss, style_loss, kld_loss

    def forward_encoder(self, x):
        intermediate_layers = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in LAYERS_STYLE_LOSS:
                intermediate_layers.append(x)
        return intermediate_layers[-1], intermediate_layers

    def opt(self, content, style, style_loss_weight):
        # Zero the gradients
        self.optimizer.zero_grad()

        # Get the loss
        content_loss, style_loss, kld_loss = self.calculate_losses(
            content.cuda(), style.cuda()
        )

        A = 500 # Point at which the weight starts growing linearly
        B = 40_000  # Point at which the weight becomes 1.0
        if self.global_step < A:
            kld_weight = 0.0
        elif self.global_step < B:
            # Linear increase from 0 to 1 between A and B
            # kld_weight = (self.global_step - A) / (B - A)
            # Cosine increase from 0 to 1 between A and B
            kld_weight = 0.5 * (1 - math.cos(math.pi * (self.global_step - A) / (B - A)))

        else:
            kld_weight = 1.0

        kld_weight *= 0.01
        total_loss = (
            1.0 * content_loss + style_loss_weight * style_loss + kld_loss * kld_weight
        ) / (1.0 + style_loss_weight + kld_weight)

        # Backward prop
        total_loss.backward()
        self.optimizer.step()
        self.global_step += 1

        return content_loss, style_loss, kld_loss, total_loss, kld_weight

    def __call__(self, content, style, alpha=1.0):
        return self.transfer_style(content, style, alpha)


def adain(content, style, eps=1e-5):
    mu_style = style.mean(axis=[-1, -2], keepdims=True)
    sigma_style = style.std(axis=[-1, -2], keepdims=True)
    mu_content = content.mean(axis=[-1, -2], keepdims=True)
    sigma_content = content.std(axis=[-1, -2], keepdims=True)
    renorm = ((content - mu_content) / (sigma_content + eps)) * sigma_style + mu_style
    return renorm


def build_encoder():
    model = models.vgg19(pretrained=True)
    encoder = model.features

    # Freeze the parameters of the model
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder


def build_decoder():
    decoder = nn.Sequential(
        nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    )
    return decoder


def build_vae():
    model = nn.Sequential(
        nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.Tanh(),
        nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    )
    return model


def calculate_kld_loss(mu, logvar):
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
    kld_loss = kld_loss.mean(axis=0) # Average batch values
    return kld_loss
