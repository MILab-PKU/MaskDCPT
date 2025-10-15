## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import trunc_normal_


##########################################################################
## Layer Norm


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


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=False,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x, mask):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x *= _get_active_ex_or_ii(
            H=x.shape[2], W=x.shape[3], _cur_active=mask, returning_active_ex=True
        )
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=False,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x, mask):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv *= _get_active_ex_or_ii(
            H=x.shape[2], W=x.shape[3], _cur_active=mask, returning_active_ex=True
        )

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # attn = F.relu(attn)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = MaskLayerNorm2d(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = MaskLayerNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x, mask), mask)
        x = x + self.ffn(self.norm2(x, mask), mask)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.conv = nn.Conv2d(
            n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.downsample = nn.PixelUnshuffle(2)

    def forward(self, x, mask):
        x = self.conv(x)
        x *= _get_active_ex_or_ii(
            H=x.shape[2], W=x.shape[3], _cur_active=mask, returning_active_ex=True
        )
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.conv = nn.Conv2d(
            n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x, mask):
        x = self.conv(x)
        x *= _get_active_ex_or_ii(
            H=x.shape[2], W=x.shape[3], _cur_active=mask, returning_active_ex=True
        )
        return self.upsample(x)


class MaskRestormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        head,
        num_block,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_block)
            ]
        )

    def forward(self, x, mask):
        for block in self.blocks:
            # use checkpoint
            # x = checkpoint.checkpoint(block, x)
            x = block(x, mask)
        return x


@ARCH_REGISTRY.register()
class MaskRestormer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",  ## Other option 'WithBias'
        dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        scale=1,  ## True for super-resolution task only. Also set scale.
        window_size=8,
    ):

        super(MaskRestormer, self).__init__()

        self.mask_patch_size = window_size
        self.mask_ratio = 0.5

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = MaskRestormerBlock(
            dim, heads[0], num_blocks[0], ffn_expansion_factor, bias, LayerNorm_type
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = MaskRestormerBlock(
            int(dim * 2**1),
            heads[1],
            num_blocks[1],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = MaskRestormerBlock(
            int(dim * 2**2),
            heads[2],
            num_blocks[2],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = MaskRestormerBlock(
            int(dim * 2**3),
            heads[3],
            num_blocks[3],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
        )

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = MaskRestormerBlock(
            int(dim * 2**2),
            heads[2],
            num_blocks[2],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = MaskRestormerBlock(
            int(dim * 2**1),
            heads[1],
            num_blocks[1],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = MaskRestormerBlock(
            int(dim * 2**1),
            heads[0],
            num_blocks[0],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
        )

        self.refinement = MaskRestormerBlock(
            int(dim * 2**1),
            heads[0],
            num_refinement_blocks,
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
        )

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)
        ###########################

        self.scale = scale
        if scale > 1:
            self.output = nn.ModuleList(
                [
                    nn.Conv2d(
                        int(dim * 2**1),
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                    )
                    for _ in range(2**scale)
                ]
            )
        else:
            self.output = nn.Conv2d(
                int(dim * 2**1),
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def mask(self, B, h, w, device, generator=None):
        len_keep = round(h * w * (1 - self.mask_ratio))
        idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :len_keep].to(device)  # (B, len_keep)
        return (
            torch.zeros(B, h * w, dtype=torch.bool, device=device)
            .scatter_(dim=1, index=idx, value=True)
            .view(B, 1, h, w)
        )

    def forward(self, inp_img, hook=None):
        B, C, H, W = inp_img.shape
        # generate mask
        mask = self.mask(
            B, H // self.mask_patch_size, W // self.mask_patch_size, inp_img.device
        )

        # mask input
        inp_img = inp_img * mask.repeat_interleave(
            self.mask_patch_size, 2
        ).repeat_interleave(self.mask_patch_size, 3)

        inp_enc_level1 = self.patch_embed(inp_img) * mask.repeat_interleave(
            self.mask_patch_size, 2
        ).repeat_interleave(self.mask_patch_size, 3)

        out_enc_level1 = self.encoder_level1(inp_enc_level1, mask)

        inp_enc_level2 = self.down1_2(out_enc_level1, mask)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, mask)

        inp_enc_level3 = self.down2_3(out_enc_level2, mask)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, mask)

        inp_enc_level4 = self.down3_4(out_enc_level3, mask)
        latent = self.latent(inp_enc_level4, mask)

        inp_dec_level3 = self.up4_3(latent, mask)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, mask)

        inp_dec_level2 = self.up3_2(out_dec_level3, mask)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, mask)

        inp_dec_level1 = self.up2_1(out_dec_level2, mask)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, mask)

        out_dec_level1 = self.refinement(out_dec_level1, mask)

        # for knn
        mask = mask.repeat_interleave(8, 2).repeat_interleave(8, 3).expand(-1, 96, -1, -1)
        out_dec_level1 = out_dec_level1[mask].flatten()

        # # for output
        # #### For Dual-Pixel Defocus Deblurring Task ####
        # if self.dual_pixel_task:
        #     out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        #     out_dec_level1 = self.output(out_dec_level1)
        # ###########################
        # else:
        #     out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1
