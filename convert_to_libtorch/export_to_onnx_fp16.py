import onnx
import torch

from model_jit.SemLA import SemLA
from onnxconverter_common import float16

device = torch.device("cuda")

matcher = SemLA(device=device)
matcher.load_state_dict(torch.load(f"./reg.ckpt"), strict=False)

matcher = matcher.eval().to(device)

width = 320
height = 240

torch_input_1 = torch.randn(1, 1, height, width).to(device)
torch_input_2 = torch.randn(1, 1, height, width).to(device)

torch.onnx.export(
    matcher,
    (torch_input_1, torch_input_2),
    # f"/home/hugo/example/SemLA_cpp_onnx/SemLA_onnx_{device}.onnx",
    f"SemLA_onnx_{device}_{width}x{height}.onnx",
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

# convert to fp16
model = onnx.load(f"SemLA_onnx_{device}_{width}x{height}.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, f"SemLA_onnx_{device}_{width}x{height}_fp16.onnx")

model_fp16 = onnx.load(f"SemLA_onnx_{device}_{width}x{height}_fp16.onnx")
try:
    onnx.checker.check_model(model_fp16)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")
