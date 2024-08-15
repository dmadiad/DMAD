import timm  # noqa
import torch
import torchvision.models as models  # noqa

def load_ref_wrn50():
    import resnet
    return resnet.wide_resnet50_2(True)

_BACKBONES = {
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "dino_vits8": 'torch.hub.load("facebookresearch/dino:main", "dino_vits8")',
    "dino_vits16": 'torch.hub.load("facebookresearch/dino:main", "dino_vits16")',
    "dinov2_vits14": 'torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")'
}


def load(name):
    return eval(_BACKBONES[name])
