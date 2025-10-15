import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY

from ..metrics.psnr_ssim import calculate_ssim_pt
from .loss_util import weighted_loss

_reduction_modes = ["none", "mean", "sum"]


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)


@weighted_loss
def huber_loss(pred, target, delta=0.01):
    abs_error = torch.abs(pred - target)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    losses = 0.5 * torch.pow(quadratic, 2) + linear
    return losses


@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean") -> None:
        super().__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        return self.loss_weight * F.cross_entropy(
            pred, target, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class ColorHistLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(ColorHistLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        b, c, _, _ = pred.shape
        # Compute histograms for each channel
        hist1 = []
        hist2 = []
        for c in range(c):  # Loop over Lab channels
            h1 = []
            h2 = []
            for n in range(b):  # Compute histograms per image in batch
                h1.append(torch.histc(pred[n, c], bins=255, min=0, max=1))
                h2.append(torch.histc(target[n, c], bins=255, min=0, max=1))
            hist1.append(torch.stack(h1))
            hist2.append(torch.stack(h2))

        # Combine histograms into tensors of shape (N, C, bins)
        hist1 = torch.stack(hist1, dim=1)  # Shape: (N, C, bins)
        hist2 = torch.stack(hist2, dim=1)  # Shape: (N, C, bins)

        # Normalize histograms
        hist1 = hist1 / hist1.sum(dim=-1, keepdim=True)
        hist2 = hist2 / hist2.sum(dim=-1, keepdim=True)

        # Compute cosine similarity for each channel and batch
        cos_sim = F.cosine_similarity(hist1, hist2, dim=-1)  # Shape: (N, C)
        loss = (1 - cos_sim).mean()  # Average over batch and channels

        return loss


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class SmoothL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(SmoothL1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * F.smooth_l1_loss(
            pred, target, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class HuberLoss(nn.Module):
    """Huber (Smooth L1) loss.

    Args:
        loss_weight (float): Loss weight for Huber loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, delta=0.01, reduction="mean"):
        super(HuberLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * huber_loss(
            pred, target, weight, delta=self.delta, reduction=self.reduction
        )


def rfft(x):
    x_fft = torch.fft.rfft2(x)
    return torch.stack([x_fft.real, x_fft.imag], dim=-1)


@LOSS_REGISTRY.register()
class MultiStageDeblurLoss(nn.Module):
    def __init__(self, loss_weight=1.0, lambda_fft=0.1, reduction="mean"):
        super(MultiStageDeblurLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.lambda_fft = lambda_fft
        self.reduction = reduction
        self.criterion = nn.L1Loss()

    def forward(self, pred, target, weight=None, **kwargs):
        target_x2 = F.interpolate(target, scale_factor=0.5, mode="bilinear")
        target_x4 = F.interpolate(target, scale_factor=0.25, mode="bilinear")

        l_x1 = self.criterion(pred[0], target)
        l_x2 = self.criterion(pred[1], target_x2)
        l_x4 = self.criterion(pred[2], target_x4)

        target_x1_fft = rfft(target)
        pred_x1_fft = rfft(pred[0])
        target_x2_fft = rfft(target_x2)
        pred_x2_fft = rfft(pred[1])
        target_x4_fft = rfft(target_x4)
        pred_x4_fft = rfft(pred[2])

        l_fft_x1 = self.criterion(pred_x1_fft, target_x1_fft)
        l_fft_x2 = self.criterion(pred_x2_fft, target_x2_fft)
        l_fft_x4 = self.criterion(pred_x4_fft, target_x4_fft)

        loss = l_x1 + l_x2 + l_x4 + self.lambda_fft * (l_fft_x1 + l_fft_x2 + l_fft_x4)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """SSIM (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for SSIM loss. Default: 1.0.
    """

    def __init__(
        self,
        ssim_weight=0.1,
        loss_weight=1.0,
        crop_border=0,
        reduction="mean",
        test_y_channel=False,
    ):
        super(SSIMLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = loss_weight
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return self.ssim_weight * (
            1
            - calculate_ssim_pt(
                pred,
                target,
                crop_border=self.crop_border,
                test_y_channel=self.test_y_channel,
                image_range=1,
            )[0].mean()
        ) + self.mse_weight * huber_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class SSIMMSELoss(nn.Module):
    """SSIM (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for SSIM loss. Default: 1.0.
    """

    def __init__(
        self,
        ssim_weight=0.1,
        mse_weight=1.0,
        crop_border=0,
        reduction="mean",
        test_y_channel=False,
    ):
        super(SSIMMSELoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return self.ssim_weight * (
            1
            - calculate_ssim_pt(
                pred,
                target,
                crop_border=self.crop_border,
                test_y_channel=self.test_y_channel,
                image_range=1,
                float64=False,
            ).mean()
        ) + self.mse_weight * mse_loss(pred, target, None, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction="mean", eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: mean | sum"
            )
        super(WeightedTVLoss, self).__init__(
            loss_weight=loss_weight, reduction=reduction
        )

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target, weight=None, **kwargs):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(
        self,
        layer_weights,
        vgg_type="vgg19",
        use_input_norm=True,
        range_norm=False,
        perceptual_weight=1.0,
        style_weight=0.0,
        criterion="l1",
    ):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == "for-norm":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "for-norm":
                    percep_loss += (
                        torch.norm(x_features[k] - gt_features[k], p="for-norm")
                        * self.layer_weights[k]
                    )
                else:
                    percep_loss += (
                        self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "for-norm":
                    style_loss += (
                        torch.norm(
                            self._gram_mat(x_features[k])
                            - self._gram_mat(gt_features[k]),
                            p="for-norm",
                        )
                        * self.layer_weights[k]
                    )
                else:
                    style_loss += (
                        self.criterion(
                            self._gram_mat(x_features[k]),
                            self._gram_mat(gt_features[k]),
                        )
                        * self.layer_weights[k]
                    )
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class PatchesKernel(nn.Module):
    def __init__(self, kernelsize, kernelstride, kernelpadding=0):
        super(PatchesKernel, self).__init__()
        kernel = torch.eye(kernelsize**2).view(
            kernelsize**2, 1, kernelsize, kernelsize
        )
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(kernelsize**2), requires_grad=False)
        self.kernelsize = kernelsize
        self.stride = kernelstride
        self.padding = kernelpadding

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, -1, self.kernelsize**2)
        return x


@LOSS_REGISTRY.register()
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


@LOSS_REGISTRY.register()
class PatchLoss(nn.Module):
    """Define patch loss
    Args:
        kernel_sizes (list): add (x, y) in the list.
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, kernel_sizes=[2, 4], loss_weight=1.0):
        super(PatchLoss, self).__init__()
        self.kernels = kernel_sizes
        self.loss_weight = loss_weight

    def forward(self, preds, labels):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            labels (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        (
            n,
            c,
            h,
            w,
        ) = preds.shape
        if c == 3:
            preds = (
                16.0
                + (
                    65.481 * preds[:, 0, :, :]
                    + 128.553 * preds[:, 1, :, :]
                    + 24.966 * preds[:, 2, :, :]
                )
            ) / 255.0
            labels = (
                16.0
                + (
                    65.481 * labels[:, 0, :, :]
                    + 128.553 * labels[:, 1, :, :]
                    + 24.966 * labels[:, 2, :, :]
                )
            ) / 255.0
            preds = preds.unsqueeze(1)
            labels = labels.unsqueeze(1)
        loss = 0.0

        for _kernel in self.kernels:
            _patchkernel = PatchesKernel(
                _kernel, _kernel // 2 + 1
            ).cuda()  # create instance
            preds_trans = _patchkernel(preds)  # [N, patch_num, patch_len ** 2]
            labels_trans = _patchkernel(labels)  # [N, patch_num, patch_len ** 2]
            preds_trans = preds_trans.reshape(
                -1, preds_trans.shape[-1]
            )  # [N * patch_num, patch_len ** 2]
            labels_trans = labels_trans.reshape(
                -1, labels_trans.shape[-1]
            )  # [N * patch_num, patch_len ** 2]
            x = torch.clamp(preds_trans, 0.000001, 0.999999)
            y = torch.clamp(labels_trans, 0.000001, 0.999999)
            dot_x_y = torch.einsum("ik,ik->i", x, y)  # [N * patch_num]
            pearson_x_y = torch.mean(
                torch.div(
                    torch.div(dot_x_y, torch.sqrt(torch.sum(x**2, dim=1))),
                    torch.sqrt(torch.sum(y**2, dim=1)),
                )
            )
            loss = loss + torch.exp(-pearson_x_y)  # y = e^(-x)
            # loss = loss - pearson_x_y # y = - x

        return loss * self.loss_weight


class PatchesKernel3D(nn.Module):
    def __init__(self, kernelsize, kernelstride, kernelpadding=0):
        super(PatchesKernel3D, self).__init__()
        kernel = torch.eye(kernelsize**2).view(
            kernelsize**2, 1, kernelsize, kernelsize
        )
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(kernelsize**2), requires_grad=False)
        self.kernelsize = kernelsize
        self.stride = kernelstride
        self.padding = kernelpadding

    def forward(self, x):
        batchsize = x.shape[0]
        channels = x.shape[1]
        x = x.reshape(batchsize * channels, x.shape[-2], x.shape[-1]).unsqueeze(1)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        x = (
            x.permute(0, 2, 3, 1)
            .reshape(batchsize, channels, -1, self.kernelsize**2)
            .permute(0, 2, 1, 3)
        )
        return x


class GradNet(nn.Module):
    def __init__(self):
        super(GradNet, self).__init__()
        kernel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        self.register_parameter(
            "kernel_x",
            nn.Parameter(
                torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0),
                requires_grad=False,
            ),
        )

        kernel_y = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
        self.register_parameter(
            "kernel_y",
            nn.Parameter(
                torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0),
                requires_grad=False,
            ),
        )

    def forward(self, x):
        grad_x = F.conv2d(x, self.kernel_x, stride=1, padding=1)
        grad_y = F.conv2d(x, self.kernel_y, stride=1, padding=1)
        return grad_x.abs() + grad_y.abs()


@LOSS_REGISTRY.register()
class GradStdLoss(nn.Module):
    def __init__(self, patch_sizes=[1, 2, 4, 8], loss_weight=1.0) -> None:
        super().__init__()
        self.get_grad = GradNet()
        self.patch_sizes = patch_sizes
        self.loss_weight = loss_weight

    def forward(self, preds, labels):
        grad_preds = self.get_grad(preds)
        grad_labels = self.get_grad(labels)
        loss = 0.0
        for patch_size in self.patch_sizes:
            loss = loss + l1_loss(
                F.unfold(grad_preds, kernel_size=patch_size, stride=patch_size).std(
                    dim=1
                ),
                F.unfold(grad_labels, kernel_size=patch_size, stride=patch_size).std(
                    dim=1
                ),
            )
        return l1_loss(grad_preds, grad_labels) * self.loss_weight


@LOSS_REGISTRY.register()
class PatchLoss3DXD(nn.Module):
    """Define patch loss
    Args:
        kernel_sizes (list): add (x, y) in the list.
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(
        self,
        kernel_sizes=[2, 4],
        loss_weight=1.0,
        grad_weight=1.0,
        use_std_to_force=True,
    ):
        super(PatchLoss3DXD, self).__init__()
        self.kernels = kernel_sizes
        self.loss_weight = loss_weight
        self.grad_weight = grad_weight
        self.use_std_to_force = use_std_to_force
        self.get_grad = GradNet()

    def forward(self, preds, labels):
        if self.grad_weight != 0:
            grad_preds = self.get_grad(preds)
            grad_labels = self.get_grad(labels)
            loss = self._forward_impl(grad_preds, grad_labels) * self.grad_weight
        else:
            loss = 0.0
        loss = loss + self._forward_impl(preds, labels)
        return loss * self.loss_weight

    def _forward_impl(self, preds, labels):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            labels (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        loss = 0.0
        for _kernel in self.kernels:
            _patchkernel = PatchesKernel3D(
                _kernel, _kernel // 2 + 1
            ).cuda()  # create instance
            preds_trans = _patchkernel(
                preds
            )  # [N, patch_num, channels, patch_len ** 2]
            labels_trans = _patchkernel(
                labels
            )  # [N, patch_num, channels, patch_len ** 2]
            preds_trans = preds_trans.reshape(
                -1, preds_trans.shape[-1]
            )  # [N * patch_num * channels, patch_len ** 2]
            labels_trans = labels_trans.reshape(
                -1, labels_trans.shape[-1]
            )  # [N * patch_num * channels, patch_len ** 2]
            x = torch.clamp(preds_trans, 0.000001, 0.999999)
            y = torch.clamp(labels_trans, 0.000001, 0.999999)
            dot_x_y = torch.einsum("ik,ik->i", x, y)
            if self.use_std_to_force == False:
                cosine0_x_y = torch.div(
                    torch.div(dot_x_y, torch.sqrt(torch.sum(x**2, dim=1))),
                    torch.sqrt(torch.sum(y**2, dim=1)),
                )
                loss = loss + torch.mean((1 - cosine0_x_y))  # y = 1-x
            else:
                dy = torch.std(labels_trans * 10, dim=1)
                cosine_x_y = torch.div(
                    torch.div(dot_x_y, torch.sqrt(torch.sum(x**2, dim=1))),
                    torch.sqrt(torch.sum(y**2, dim=1)),
                )
                cosine_x_y_d = torch.mul((1 - cosine_x_y), dy)  # y = (1-x) dy
                loss = loss + torch.mean(cosine_x_y_d)
        return loss
