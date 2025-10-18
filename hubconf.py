dependencies = ['torch', 'timm']

import torch
from gazelle.model import get_gazelle_model

def gazelle_dinov2_vitb14(pretrained=True):
    model, transform = get_gazelle_model('gazelle_dinov2_vitb14')
    if pretrained:
        ckpt_path = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_hub.pt"
        model.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(ckpt_path))
    return model, transform

def gazelle_dinov2_vitl14(pretrained=True):
    model, transform = get_gazelle_model('gazelle_dinov2_vitl14')
    if pretrained:
        ckpt_path = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14.pt"
        model.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(ckpt_path))
    return model, transform

def gazelle_dinov2_vitb14_inout(pretrained=True):
    model, transform = get_gazelle_model('gazelle_dinov2_vitb14_inout')
    if pretrained:
        ckpt_path = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt"
        model.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(ckpt_path))
    return model, transform

def gazelle_dinov2_vitl14_inout(pretrained=True):
    model, transform = get_gazelle_model('gazelle_dinov2_vitl14_inout')
    if pretrained:
        ckpt_path = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14_inout.pt"
        model.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(ckpt_path))
    return model, transform

__all__ = [
    'gazelle_dinov2_vitb14',
    'gazelle_dinov2_vitl14',
    'gazelle_dinov2_vitb14_inout',
    'gazelle_dinov2_vitl14_inout',
]