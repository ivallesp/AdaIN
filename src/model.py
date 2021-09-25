from torchvision import models
from torch import nn

# First ReLU after MaxPooling
LAYERS_STYLE_LOSS = [1, 6, 11, 20]


class AdaInStyleTransfer:
    def __init__(self):
        self.encoder = build_encoder()
        self.decoder = build_decoder()

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

        # Bring the normalized code to the image domain
        new_style = self.decoder(newstyle_code)
        return new_style, style_loss_components_target, adain_output

    def calculate_losses(self, content, style):
        new_style, style_loss_components_target, adain_output = self.transfer_style(
            content, style
        )
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

        return content_loss, style_loss

    def forward_encoder(self, x):
        intermediate_layers = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in LAYERS_STYLE_LOSS:
                intermediate_layers.append(x)
        return intermediate_layers[-1], intermediate_layers

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
