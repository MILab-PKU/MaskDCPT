## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY

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


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x *= _get_active_ex_or_ii(
            H=x.shape[2], W=x.shape[3], _cur_active=mask, returning_active_ex=True
        )
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

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

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


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


##########################################################################
## Transformer Block
class MaskTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type=None):
        super(MaskTransformerBlock, self).__init__()

        self.norm1 = MaskLayerNorm2d(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = MaskLayerNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x, mask), mask)
        x = x + self.ffn(self.norm2(x, mask), mask)

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size),
            requires_grad=True,
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x, mask):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1
        ) * self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt *= _get_active_ex_or_ii(
            H=x.shape[2], W=x.shape[3], _cur_active=mask, returning_active_ex=True
        )
        prompt = self.conv3x3(prompt)
        prompt *= _get_active_ex_or_ii(
            H=x.shape[2], W=x.shape[3], _cur_active=mask, returning_active_ex=True
        )
        return prompt


class MaskPromptIRBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, num_blocks):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                MaskTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return x


##########################################################################
##---------- PromptIR -----------------------
@ARCH_REGISTRY.register()
class MaskPromptIR(nn.Module):
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
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
        decoder=True,
        window_size=8,
    ):

        super(MaskPromptIR, self).__init__()

        self.mask_patch_size = window_size
        self.mask_ratio = 0.5

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.decoder = decoder

        if self.decoder:
            self.prompt1 = PromptGenBlock(
                prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96
            )
            self.prompt2 = PromptGenBlock(
                prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192
            )
            self.prompt3 = PromptGenBlock(
                prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384
            )

        self.encoder_level1 = MaskPromptIRBlock(
            dim, heads[0], ffn_expansion_factor, bias, num_blocks[0]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.encoder_level2 = MaskPromptIRBlock(
            int(dim * 2**1), heads[1], ffn_expansion_factor, bias, num_blocks[1]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3

        self.encoder_level3 = MaskPromptIRBlock(
            int(dim * 2**2), heads[2], ffn_expansion_factor, bias, num_blocks[2]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = MaskPromptIRBlock(
            int(dim * 2**3), heads[3], ffn_expansion_factor, bias, num_blocks[3]
        )

        self.up4_3 = Upsample(int(dim * 2**2))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**1) + 192, int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.noise_level3 = MaskTransformerBlock(
            dim=int(dim * 2**2) + 512,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level3 = nn.Conv2d(
            int(dim * 2**2) + 512, int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.decoder_level3 = MaskPromptIRBlock(
            int(dim * 2**2), heads[2], ffn_expansion_factor, bias, num_blocks[2]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.noise_level2 = MaskTransformerBlock(
            dim=int(dim * 2**1) + 224,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level2 = nn.Conv2d(
            int(dim * 2**1) + 224, int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.decoder_level2 = MaskPromptIRBlock(
            int(dim * 2**1), heads[1], ffn_expansion_factor, bias, num_blocks[1]
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = MaskTransformerBlock(
            dim=int(dim * 2**1) + 64,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level1 = nn.Conv2d(
            int(dim * 2**1) + 64, int(dim * 2**1), kernel_size=1, bias=bias
        )

        self.decoder_level1 = MaskPromptIRBlock(
            int(dim * 2**1), heads[0], ffn_expansion_factor, bias, num_blocks[0]
        )

        self.refinement = MaskPromptIRBlock(
            int(dim * 2**1), heads[0], ffn_expansion_factor, bias, num_refinement_blocks
        )

        self.output = nn.Conv2d(
            int(dim * 2**1),
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
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

    def forward(self, inp_img):
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
        if self.decoder:
            dec3_param = self.prompt3(latent, mask)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent, mask)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent, mask)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3, mask)
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3, mask)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3, mask)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3, mask)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2, mask)
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2, mask)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2, mask)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2, mask)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1, mask)

        out_dec_level1 = self.refinement(out_dec_level1, mask)

        # for knn
        mask = mask.repeat_interleave(8, 2).repeat_interleave(8, 3).expand(-1, 96, -1, -1)
        out_dec_level1 = out_dec_level1[mask].flatten()

        # # for output
        # out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1
