from torchvision import models
from torch import nn

LAYERS_STYLE_LOSS = [0, 10, 19, 28]


class AdaInStyleTransfer:
    def __init__(self):
        self.encoder, self.encoder_hook = build_encoder()
        self.decoder = build_decoder()

    def transfer_style(self, content, style):
        # Encode the content images using the pretrained model
        content_code = self.encoder(content)
        # Encode the style images using the pretrained model
        style_code = self.encoder(style)
        # Collect the outputs of the intermediate layers for computing the style loss later
        style_loss_components_target = self.encoder_hook.copy()

        # ADAIN: Normalize the content code to follow style stats
        adain_output = adain(content_code, style_code)

        # Bring the normalized code to the image domain
        new_style = self.decoder(adain_output)
        return new_style, style_loss_components_target, adain_output

    def calculate_losses(self, content, style):
        new_style, style_loss_components_target, adain_output = self.transfer_style(
            content, style
        )

        # Encode the generated image with the transferred style
        newstyle_code = self.encoder(new_style)
        # Collect the outputs of the intermediate layers for computint the style loss later
        style_loss_components_gen = self.encoder_hook.copy()

        # Compute the content loss
        content_loss = nn.MSELoss()(adain_output, newstyle_code)

        # Compute the style loss for the selected layers
        style_loss = 0
        for layer in LAYERS_STYLE_LOSS:
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



def adain(content, style, eps=1e-5):
    mu_style = style.mean(axis=[-1, -2], keepdims=True)
    sigma_style = style.std(axis=[-1, -2], keepdims=True)
    mu_content = content.mean(axis=[-1, -2], keepdims=True)
    sigma_content = content.std(axis=[-1, -2], keepdims=True)
    renorm = ((content - mu_content) / (sigma_content + eps)) * sigma_style + mu_style
    return renorm


def get_activation(name, dictionary):
    def hook(model, input, output):
        dictionary[name] = output.detach()

    return hook


def build_encoder():
    model = models.vgg19(pretrained=True)

    conv_blocks = list(model.children())[0]

    encoder_blocks = list(conv_blocks.children())[:-8]

    encoder = nn.Sequential(*encoder_blocks)

    # Create the forward hooks to collect the outputs of some intermediate layers,
    # used to compute the style loss
    activations_style_loss = {}
    for l in LAYERS_STYLE_LOSS:
        encoder.register_forward_hook(get_activation(l, activations_style_loss))

    # Freeze the parameters of the model
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder, activations_style_loss


def build_decoder():
    decoder = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
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
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    )
    return decoder
