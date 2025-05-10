import torch

from model_jit.SemLA import SemLA

device = torch.device("cuda")
fpMode = torch.float32


matcher = SemLA(device=device, fp=fpMode)
matcher.load_state_dict(torch.load(f"./reg.ckpt"), strict=False)

matcher = matcher.eval().to(device, dtype=fpMode)

width = 320
height = 256

torch_input_1 = torch.randn(1, 1, height, width).to(device)
torch_input_2 = torch.randn(1, 1, height, width).to(device)

if fpMode == torch.float16:
    torch_input_1 = torch_input_1.half()
    torch_input_2 = torch_input_2.half()

torch.onnx.export(
    matcher,
    (torch_input_1, torch_input_2),
    f"/home/hugo/example/SemLA_cpp_onnx/SemLA_onnx_{device}_{width}x{height}.onnx",
    verbose=True,
    opset_version=12,
    input_names=["vi_img", "ir_img"],
    output_names=["mkpt0", "mkpt1", "feat_sa_vi", "feat_sa_ir", "sa_ir"],
    dynamic_axes={
        "vi_img": [2, 3],
        "ir_img": [2, 3],
        "mkpt0": [0],
        "mkpt1": [0],
        "feat_sa_vi": [2, 3],
        "feat_sa_ir": [2, 3],
        "sa_ir": [2, 3],
    },
)

# if want to convert onnx model to tensorrt, please run export_to_onnx_tensorrt.py
