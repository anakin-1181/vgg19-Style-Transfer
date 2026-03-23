"""Loss functions used by the style transfer pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.image_utils import get_vgg_gram_matrix


class StyleLoss(nn.Module):
    """Track style loss against a target Gram matrix."""

    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.loss = 0.0
        self.target = target_feature.detach()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = get_vgg_gram_matrix(input_tensor)
        self.loss = F.mse_loss(output, self.target)
        return input_tensor


class ContentLoss(nn.Module):
    """Track content loss against a target feature map."""

    def __init__(self, feature_map: torch.Tensor):
        super().__init__()
        self.loss = 0.0
        self.target = feature_map.detach()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(input_tensor, self.target)
        return input_tensor
