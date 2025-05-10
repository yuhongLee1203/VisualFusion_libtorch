import os
import torch

from model_jit.SemLA import SemLA

mode = torch.float16
device = torch.device("cuda")

matcher = SemLA(device, mode)
matcher.load_state_dict(torch.load(f"./reg.ckpt"), strict=False)

matcher = matcher.eval()

matcher = torch.jit.script(matcher)

torch.jit.save(matcher, f"SemLA_jit_{device}_fp16.zip")
