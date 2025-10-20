import warnings
warnings.simplefilter('ignore')
import os
import torch
from gazelle.model import get_gazelle_model
import onnx
from onnxsim import simplify

"""
"gazelle_dinov2_vitb14": gazelle_dinov2_vitb14, gazelle_dinov2_vitb14.pt
"gazelle_dinov2_vitl14": gazelle_dinov2_vitl14, gazelle_dinov2_vitl14.pt
"gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout, gazelle_dinov2_vitb14_inout.pt
"gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout, gazelle_dinov2_vitl14_inout.pt
"gazelle_dinov3_vit_tiny": gazelle_dinov3_vit_tiny, gazelle_dinov3_vit_tiny.pt
"gazelle_dinov3_vit_tinyplus": gazelle_dinov3_vit_tinyplus, gazelle_dinov3_vit_tinyplus.pt
"gazelle_dinov3_vits16": gazelle_dinov3_vits16, gazelle_dinov3_vits16.pt
"gazelle_dinov3_vits16plus": gazelle_dinov3_vits16plus, gazelle_dinov3_vits16plus.pt
"gazelle_dinov3_vitb16": gazelle_dinov3_vitb16, gazelle_dinov3_vitb16.pt

    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            transforms.Resize(in_size),
        ])
"""

models = {
    # DINOv2
    # "gazelle_dinov2_vitb14": ["gazelle_dinov2_vitb14.pt", False],
    # "gazelle_dinov2_vitl14": ["gazelle_dinov2_vitl14.pt", False],
    # "gazelle_dinov2_vitb14_inout": ["gazelle_dinov2_vitb14_inout.pt", True],
    # "gazelle_dinov2_vitl14_inout": ["gazelle_dinov2_vitl14_inout.pt", True],
    # DINOv3
    "gazelle_dinov3_vit_tiny": ["gazelle_dinov3_vit_tiny.pt", False, 640, 640],
    # "gazelle_dinov3_vit_tinyplus": ["gazelle_dinov3_vit_tinyplus.pt", False, 640, 640],
    # "gazelle_dinov3_vits16": ["gazelle_dinov3_vits16.pt", False, 640, 640],
    # "gazelle_dinov3_vits16plus": ["gazelle_dinov3_vits16plus.pt", False, 640, 640],
    # "gazelle_dinov3_vitb16": ["gazelle_dinov3_vitb16.pt", False, 640, 640],
}

for m, params in models.items():
    model, transform = get_gazelle_model(model_name=m, onnx_export=True)
    ckpt_raw = torch.load(params[0], weights_only=False)
    ckpt = ckpt_raw["model"] if isinstance(ckpt_raw, dict) and "model" in ckpt_raw else ckpt_raw
    has_backbone = any(k.startswith("backbone") for k in ckpt.keys())
    model.load_gazelle_state_dict(ckpt, include_backbone=True)
    if not has_backbone:
        print(f"WARNING: Checkpoint {params[0]} does not contain backbone weights; exporting with backbone defaults.")
    model.eval()
    model.cpu()

    num_heads = 1
    filename_wo_ext = os.path.splitext(os.path.basename(params[0]))[0]
    h = int(params[2])
    w = int(params[3])
    oh = str(model.out_size[0])
    ow = str(model.out_size[1])
    onnx_file = f"{filename_wo_ext}_1x3x{h}x{w}_1x{num_heads}x4.onnx"
    images = torch.randn(1, 3, h, w).cpu()
    bboxes = torch.randn(1, num_heads, 4).cpu()
    if not params[1]:
        outputs = [
            'heatmap',
        ]
        dynamic_axes = {
            'bboxes_x1y1x2y2' : {1: 'heads'},
            'heatmap': {0: 'heads', 1: oh, 2: ow},
        }
    else:
        outputs = [
            'heatmap',
            'inout',
        ]
        dynamic_axes = {
            'bboxes_x1y1x2y2' : {1: 'heads'},
            'heatmap': {0: 'heads', 1: oh, 2: ow},
            'inout': {0: 'heads'},
        }

    torch.onnx.export(
        model,
        args=(images, bboxes),
        f=onnx_file,
        opset_version=17,
        input_names=[
            'image_bgr',
            'bboxes_x1y1x2y2',
        ],
        output_names=outputs,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)


    onnx_file = f"{filename_wo_ext}_1x3x{h}x{w}_1xNx4.onnx"
    images = torch.randn(1, 3, h, w).cpu()
    bboxes = torch.randn(1, num_heads, 4).cpu()
    torch.onnx.export(
        model,
        args=(images, bboxes),
        f=onnx_file,
        opset_version=17,
        input_names=[
            'image_bgr',
            'bboxes_x1y1x2y2',
        ],
        output_names=outputs,
        dynamic_axes=dynamic_axes,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
