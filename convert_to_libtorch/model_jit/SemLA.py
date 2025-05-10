import torch
import torch.nn as nn
import torch.nn.functional as F

from .reg import SemLA_Reg
from .utils import n_c_h_w_2_n_hw_c


class SemLA(nn.Module):

    def __init__(self, device, fp=torch.float32):
        super().__init__()
        self.backbone = SemLA_Reg(device, fp)

    def forward(self, img_vi, img_ir):
        # Select 'scene' mode when no semantic objects exist in the image
        thr = 0
        feat_reg_vi_final, feat_reg_ir_final, feat_sa_vi, feat_sa_ir = self.backbone(
            torch.cat((img_vi, img_ir), dim=0)
        )

        sa_vi, sa_ir = feat_sa_vi.reshape(-1), feat_sa_ir.reshape(-1)
        sa_vi, sa_ir = torch.where(sa_vi > thr)[0], torch.where(sa_ir > thr)[0]

        # feat_reg_vi = rearrange(feat_reg_vi_final, "n c h w -> n (h w) c")
        feat_reg_vi = n_c_h_w_2_n_hw_c(feat_reg_vi_final)
        # feat_reg_ir = rearrange(feat_reg_ir_final, "n c h w -> n (h w) c")
        feat_reg_ir = n_c_h_w_2_n_hw_c(feat_reg_ir_final)

        feat_reg_vi, feat_reg_ir = feat_reg_vi[:, sa_vi], feat_reg_ir[:, sa_ir]
        feat_reg_vi = feat_reg_vi / feat_reg_vi.shape[-1] ** 0.5
        feat_reg_ir = feat_reg_ir / feat_reg_ir.shape[-1] ** 0.5

        conf = torch.einsum("nlc,nsc->nls", feat_reg_vi, feat_reg_ir) / 0.1

        ones = (
            torch.ones_like(conf)
            * (conf == conf.max(dim=2, keepdim=True)[0])
            * (conf == conf.max(dim=1, keepdim=True)[0])
        )
        zeros = torch.zeros_like(conf)
        mask = torch.where(conf > 0, ones, zeros)

        # mask = conf > 0.0
        # mask = (
        #     mask
        #     * (conf == conf.max(dim=2, keepdim=True)[0])
        #     * (conf == conf.max(dim=1, keepdim=True)[0])
        # )

        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        i_ids = sa_vi[i_ids]
        j_ids = sa_ir[j_ids]

        mkpts0 = (
            torch.stack(
                [
                    i_ids % feat_sa_vi.shape[3],
                    torch.div(i_ids, feat_sa_vi.shape[3], rounding_mode="trunc"),
                ],
                dim=1,
                # [i_ids % feat_sa_vi.shape[3], i_ids // feat_sa_vi.shape[3]], dim=1
            )
            * 8
        )
        mkpts1 = (
            torch.stack(
                [
                    j_ids % feat_sa_vi.shape[3],
                    torch.div(j_ids, feat_sa_vi.shape[3], rounding_mode="trunc"),
                ],
                dim=1,
                # [j_ids % feat_sa_vi.shape[3], j_ids // feat_sa_vi.shape[3]], dim=1
            )
            * 8
        )

        sa_ir = F.interpolate(
            feat_sa_ir, scale_factor=8.0, mode="bilinear", align_corners=True
        )

        score = conf[b_ids, i_ids, j_ids]

        return mkpts0, mkpts1, feat_sa_vi, feat_sa_ir, sa_ir, score
