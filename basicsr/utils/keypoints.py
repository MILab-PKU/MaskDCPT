# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger

from types import SimpleNamespace
from typing import List

import torch

# from kornia.feature import BlobDoG, LAFOrienter, ScaleSpaceDetector, get_laf_center
# from kornia.geometry import ConvQuadInterp3d, ScalePyramid
from torch import nn


def kpts2heatmap(kpts, image_height, image_width, sigma=1.0):
    """
    :param kpt_pos: bs x 2 x kps_num, [0, 1]
    :return: kpts_heatmap
    """
    y_coords = (
        2.0
        * torch.arange(image_height, dtype=kpts.dtype, device=kpts.device)
        .unsqueeze(1)
        .expand(image_height, image_width)
        / (image_height - 1.0)
        - 1.0
    )
    x_coords = (
        2.0
        * torch.arange(image_width, dtype=kpts.dtype, device=kpts.device)
        .unsqueeze(0)
        .expand(image_height, image_width)
        / (image_width - 1.0)
        - 1.0
    )
    coords = torch.stack((y_coords, x_coords), dim=0)
    coords = torch.unsqueeze(coords, dim=0)  # 1 x 2 x h x w

    H = torch.exp(
        -torch.square(
            torch.unsqueeze(coords, dim=2) - kpts.unsqueeze(3).unsqueeze(3)
        ).sum(dim=1)
        / sigma
    )  # bs x kps_num x h x w
    return H


# class TorchSIFT(torch.nn.Module):
#     def __init__(
#         self,
#         num_features: int = 64,
#         patch_size: int = 41,
#     ) -> None:
#         super().__init__()
#         self.patch_size = patch_size
#         self.detector = ScaleSpaceDetector(
#             num_features,
#             resp_module=BlobDoG(),
#             scale_space_response=True,  # We need that, because DoG operates on scale-space
#             nms_module=ConvQuadInterp3d(10),
#             scale_pyr_module=ScalePyramid(3, 1.6, patch_size, double_image=True),
#             ori_module=LAFOrienter(19),
#             mr_size=6.0,
#             minima_are_also_good=True,
#         )

#     def detect(self, x: torch.Tensor) -> torch.Tensor:
#         self.detector.to(x.device)
#         with torch.no_grad():
#             lafs, _ = self.detector(x.contiguous())
#         return lafs

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         lafs = self.detect(x)
#         kpts = get_laf_center(lafs)
#         return kpts


def max_pool(x: torch.Tensor, nms_radius: int):
    return torch.nn.functional.max_pool2d(
        x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
    )


@torch.jit.script
def simple_nms(scores: torch.Tensor, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    max_mask = scores == max_pool(scores, nms_radius)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float(), nms_radius) > 0
        supp_scores = torch.where(supp_mask, 0, scores)
        new_max_mask = supp_scores == max_pool(supp_scores, nms_radius)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, 0)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


@torch.jit.script
def refine_scores(scores: List[torch.Tensor], max_length: int):
    for i, score in enumerate(scores):
        score_shape = score.shape
        # assert score_shape[0] != 0
        if max_length > score_shape[0]:
            scores[i] = torch.cat(
                [
                    score,
                    torch.zeros((max_length - score_shape[0]), device=score.device),
                ]
            )
    return torch.stack(scores, 0)


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    preprocess_conf = {
        "resize": 1024,
    }

    required_data_keys = ["image"]

    def __init__(self, conf, pretrain=False):
        super().__init__()  # Update with default configuration.
        self.conf = SimpleNamespace(**conf)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        # self.convDb = nn.Conv2d(
        #     c5, self.conf.descriptor_dim, kernel_size=1, stride=1, padding=0
        # )
        if pretrain:
            # url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"  # noqa
            # self.load_state_dict(torch.hub.load_state_dict_from_url(url), strict=False)
            self.load_state_dict(
                torch.load("/home/hjk/sr/superpoint_v1.pth", map_location="cpu"),
                strict=False,
            )

        if self.conf.max_num_keypoints is not None and self.conf.max_num_keypoints <= 0:
            raise ValueError("max_num_keypoints must be positive or None")

    def forward(self, image, return_scores: bool = False):
        # TODO: 跨尺度、防噪声的keypoints提取
        """Compute keypoints, scores, descriptors for image"""
        # if image.shape[1] == 3:
        #     image = rgb_to_grayscale(image)

        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        if return_scores:
            return scores
        # scores_label = scores
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b = scores.shape[0]
        scores = torch.nn.functional.pixel_shuffle(scores, 8).squeeze(1)
        scores = simple_nms(scores, self.conf.nms_radius)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1
        # scores_label = scores

        # Extract keypoints
        best_kp = torch.where(scores > self.conf.detection_threshold)
        scores = scores[best_kp]

        # Separate into batches
        keypoints = [
            torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)
        ]
        scores = [scores[best_kp[0] == i] for i in range(b)]

        # Keep the k keypoints with highest score
        if self.conf.max_num_keypoints is not None:
            keypoints, scores = list(
                zip(
                    *[
                        top_k_keypoints(k, s, self.conf.max_num_keypoints)
                        for k, s in zip(keypoints, scores)
                    ]
                )
            )

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        max_shape = []
        for score in scores:
            max_shape.append(score.shape[0])
        max_shape = 1 if max(max_shape) == 0 else max(max_shape)

        return {
            "x": x,
            "keypoints": keypoints,
            "scores": refine_scores(scores, max_shape),
            "max_length": max_shape,
        }
