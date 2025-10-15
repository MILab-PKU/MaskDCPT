# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

"""
NAFSSR: Stereo Image Super-Resolution Using NAFNet
@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


# form: https://github.com/keyu-tian/SparK/blob/main/pretrain/encoder.py#L112
def _get_active_ex_or_ii(H, W, _cur_active, returning_active_ex=True):
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(
        w_repeat, dim=3
    )
    return (
        active_ex
        if returning_active_ex
        else active_ex.squeeze(1).nonzero(as_tuple=True)
    )  # ii: bi, hi, wi


class MaskLayerNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__(normalized_shape, eps, elementwise_affine=True)

    def forward(self, x, mask):
        ii = _get_active_ex_or_ii(
            H=x.shape[2], W=x.shape[3], _cur_active=mask, returning_active_ex=False
        )
        bhwc = x.permute(0, 2, 3, 1).contiguous()
        nc = bhwc[ii]
        nc = super(MaskLayerNorm2d, self).forward(nc)

        x = torch.zeros_like(bhwc)
        x[ii] = nc
        return x.permute(0, 3, 1, 2).contiguous()


class NAFSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MaskNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = MaskLayerNorm2d(c)
        self.norm2 = MaskLayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, mask):
        x = inp

        x = self.norm1(x, mask)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x *= _get_active_ex_or_ii(
            H=x.shape[2], W=x.shape[3], _cur_active=mask, returning_active_ex=True
        )

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y, mask))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class MaskNAFBlocks(nn.Module):
    def __init__(self, chan, num):
        super().__init__()
        self.blocks = nn.ModuleList([MaskNAFBlock(chan) for _ in range(num)])

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return x


@ARCH_REGISTRY.register()
class MaskNAFNetBaseline(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        window_size=8,
    ):
        super().__init__()
        self.mask_patch_size = window_size
        self.mask_ratio = 0.5

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        # self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(MaskNAFBlocks(chan, num))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = MaskNAFBlocks(chan, middle_blk_num)

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            setattr(
                self,
                f"decoder{i}",
                MaskNAFBlocks(chan, num),
            )

    def mask(self, B, h, w, device, generator=None):
        len_keep = round(h * w * (1 - self.mask_ratio))
        idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :len_keep].to(device)  # (B, len_keep)
        return (
            torch.zeros(B, h * w, dtype=torch.bool, device=device)
            .scatter_(dim=1, index=idx, value=True)
            .view(B, 1, h, w)
        )

    def forward(self, inp):
        B, C, H, W = inp.shape
        # generate mask
        mask = self.mask(
            B, H // self.mask_patch_size, W // self.mask_patch_size, inp.device
        )

        # mask input
        inp = inp * mask.repeat_interleave(self.mask_patch_size, 2).repeat_interleave(
            self.mask_patch_size, 3
        )

        x = self.intro(inp) * mask.repeat_interleave(
            self.mask_patch_size, 2
        ).repeat_interleave(self.mask_patch_size, 3)

        encs = []

        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x = encoder(
                x,
                mask.repeat_interleave(
                    self.mask_patch_size // (2**i), 2
                ).repeat_interleave(self.mask_patch_size // (2**i), 3),
            )
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x, mask)

        for i, (up, enc_skip) in enumerate(zip(self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            decoder = getattr(self, f"decoder{i}")
            x = decoder(
                x,
                mask.repeat_interleave(
                    self.mask_patch_size // (2 ** (len(self.encoders) - 1)), 2
                ).repeat_interleave(
                    self.mask_patch_size // (2 ** (len(self.encoders) - 1)), 3
                ),
            )
        
        # for knn
        mask = mask.repeat_interleave(8, 2).repeat_interleave(8, 3).expand(-1, 32, -1, -1)
        x = x[mask].flatten()

        # for output
        # x = self.ending(x)

        return x
