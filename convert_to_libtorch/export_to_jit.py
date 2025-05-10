import os
import torch

from model_jit.SemLA import SemLA

device = torch.device("cuda")

matcher = SemLA(device=device)
matcher.load_state_dict(torch.load(f"./reg.ckpt"), strict=False)

matcher = matcher.eval()

matcher = torch.jit.script(matcher)

torch.jit.save(matcher, f"SemLA_jit_{device}.zip")
