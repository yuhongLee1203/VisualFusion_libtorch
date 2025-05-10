import pywt
import torch
import matplotlib
import matplotlib.pyplot as plt

from torch import nn
from einops.einops import _prepare_transformation_recipe, _apply_recipe


def conv1x1(in_channels, out_channels, stride=1):
    """1 x 1 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
    )


def conv3x3(in_channels, out_channels, stride=1):
    """3 x 3 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class CBR(nn.Module):
    """3 x 3 convolution block

    Args:
        'x': (torch.Tensor): (N, C, H, W)
    """

    def __init__(self, in_channels, planes, stride=1):
        super().__init__()
        self.conv = conv3x3(in_channels, planes, stride)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """DepthWise convolution block

    Args:
        'x': (torch.Tensor): (N, C, H, W)
    """

    def __init__(self, out_channels):
        super().__init__()
        self.group_conv3x3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class MLP(nn.Module):
    """MLP Layer

    Args:
        'x': (torch.Tensor): (N, C, H, W)
    """

    def __init__(self, out_channels, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


# This class is implemented by [Wave-ViT](https://github.com/YehLi/ImageNetModel/blob/main/classification/torch_wavelets.py).
class DWT_2D(nn.Module):
    """Discrete Wavelet Transform for feature maps downsampling

    Args:
        'x': (torch.Tensor): (N, C, H, W)
    """

    def __init__(self, wave, fp=torch.float32):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer("w_ll", w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_lh", w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hl", w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hh", w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=fp)
        self.w_lh = self.w_lh.to(dtype=fp)
        self.w_hl = self.w_hl.to(dtype=fp)
        self.w_hh = self.w_hh.to(dtype=fp)

    def forward(self, x):
        x = x.contiguous()

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(
            x, self.w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim
        )
        x_lh = torch.nn.functional.conv2d(
            x, self.w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim
        )
        x_hl = torch.nn.functional.conv2d(
            x, self.w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim
        )
        x_hh = torch.nn.functional.conv2d(
            x, self.w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim
        )
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

        return x


class MLP2(nn.Module):
    def __init__(self, in_features, mlp_ratio=4):
        super(MLP2, self).__init__()
        hidden_features = in_features * mlp_ratio

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_features),
        )

    def forward(self, x):
        return self.fc(x)


class StructureAttention(nn.Module):
    # This class is implemented by [LoFTR](https://github.com/zju3dv/LoFTR).
    def __init__(self, d_model, nhead, fp=torch.float32):
        super(StructureAttention, self).__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention(fp=fp)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 1, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 1, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
        """
        bs = x.size(0)
        query, key, value = x, x, x

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6, fp=torch.float32):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps
        self.fp = fp

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries).to(dtype=self.fp)
        K = self.feature_map(keys).to(dtype=self.fp)
        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)

        Z = Z.to(dtype=self.fp)
        KV = KV.to(dtype=self.fp)

        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class MLP2(nn.Module):
    def __init__(self, in_features, mlp_ratio=4):
        super(MLP2, self).__init__()
        hidden_features = in_features * mlp_ratio

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_features),
        )

    def forward(self, x):
        return self.fc(x)


def make_matching_figure(
    pred, img0, img1, mkpts0, mkpts1, kpts0=None, kpts1=None, dpi=140, path=None
):
    # draw image pair
    assert (
        mkpts0.shape[0] == mkpts1.shape[0]
    ), f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=dpi)
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    axes[2].imshow(pred, cmap="gray")
    for i in range(3):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                transform=fig.transFigure,
                c=(124 / 255, 252 / 255, 0),
                linewidth=1,
            )
            for i in range(len(mkpts0))
        ]

    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig


def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0).detach()
    Cb = Cb.clamp(0.0, 1.0).detach()
    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)

    temp = (im_flat + bias).mm(mat)

    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0, 1.0)
    return out


# rearrange(tensor, "n c h w -> n (h w) c")
def n_c_h_w_2_n_hw_c(tensor):
    tensor = tensor.permute(0, 2, 3, 1)
    return tensor.contiguous().view(tensor.size(0), -1, tensor.size(-1))


# rearrange(tensor, "n h w c -> n c h w")
def n_h_w_c_2_n_c_h_w(tensor):
    return tensor.permute(0, 3, 1, 2)


# rearrange(tensor, "n c h w -> n c (h w)")
def n_c_h_w_2_n_c_hw(tensor):
    return tensor.view(tensor.size(0), tensor.size(1), -1)


# rearrange(tensor, "n (h w) c -> n c h w", h=h, w=w)
def n_hw_c_2_n_c_h_w(tensor, h: int, w: int):
    n, _, c = tensor.size()
    tensor = tensor.view(n, h, w, c)
    return tensor.permute(0, 3, 1, 2)
