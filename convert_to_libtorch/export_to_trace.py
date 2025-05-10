import os
import torch

from model_jit.SemLA import SemLA

width = 320
height = 240
mode = torch.float32
device = torch.device("cuda")

if device == torch.device("cpu"):
    mode = torch.float32

matcher = SemLA(device, mode)
matcher.load_state_dict(torch.load(f"./reg.ckpt"), strict=False)
matcher = matcher.eval().to(device)
if mode == torch.float16:
    matcher = matcher.half()

input1 = torch.randn(1, 1, 240, 320).to(device)
input2 = torch.randn(1, 1, 240, 320).to(device)
if mode == torch.float16:
    input1 = input1.half()
    input2 = input2.half()

traced = torch.jit.trace(matcher, (input1, input2))

if mode == torch.float16:
    torch.jit.save(traced, f"SemLA_trace_{device}_fp16.zip")
else:
    torch.jit.save(traced, f"SemLA_trace_{device}.zip")

