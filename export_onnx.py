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
    "gazelle_dinov2_vitb14": ["gazelle_dinov2_vitb14.pt", False],
    "gazelle_dinov2_vitl14": ["gazelle_dinov2_vitl14.pt", False],
    "gazelle_dinov2_vitb14_inout": ["gazelle_dinov2_vitb14_inout.pt", True],
    "gazelle_dinov2_vitl14_inout": ["gazelle_dinov2_vitl14_inout.pt", True],
}

for m, params in models.items():
    model, transform = get_gazelle_model(model_name=m, onnx_export=True)
    model.load_gazelle_state_dict(torch.load(params[0], weights_only=True))
    model.eval()
    model.cpu()

    num_heads = 1
    filename_wo_ext = os.path.splitext(os.path.basename(params[0]))[0]
    onnx_file = f"{filename_wo_ext}_1x3x448x448_1x{num_heads}x4.onnx"
    images = torch.randn(1, 3, 448, 448).cpu()
    bboxes = torch.randn(1, num_heads, 4).cpu()
    if not params[1]:
        outputs = [
            'heatmap',
        ]
        dynamic_axes = {
            'bboxes_x1y1x2y2' : {1: 'heads'},
            'heatmap': {0: 'heads'}
        }
    else:
        outputs = [
            'heatmap',
            'inout',
        ]
        dynamic_axes = {
            'bboxes_x1y1x2y2' : {1: 'heads'},
            'heatmap': {0: 'heads'},
            'inout': {0: 'heads'},
        }

    torch.onnx.export(
        model,
        args=(images, bboxes),
        f=onnx_file,
        opset_version=14,
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


    onnx_file = f"{filename_wo_ext}_1x3x448x448_1xNx4.onnx"
    images = torch.randn(1, 3, 448, 448).cpu()
    bboxes = torch.randn(1, num_heads, 4).cpu()
    torch.onnx.export(
        model,
        args=(images, bboxes),
        f=onnx_file,
        opset_version=14,
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