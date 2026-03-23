"""Model and feature-extraction components for style transfer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
from torchvision import models

from src.image_utils import get_vgg_gram_matrix

DEFAULT_STYLE_LAYER_IDXS = [1, 6, 11, 20]
DEFAULT_CONTENT_LAYER_IDXS = [22]
CNN_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
CNN_NORMALIZATION_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    """Prefer MPS, then CUDA, then CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class StyleTransferRuntime:
    """Shared runtime objects used across requests."""

    device: torch.device
    cnn: nn.Sequential
    normalization_mean: torch.Tensor
    normalization_std: torch.Tensor


@lru_cache(maxsize=1)
def get_style_transfer_runtime() -> StyleTransferRuntime:
    """Build the frozen VGG19 backbone once per process."""

    device = get_device()
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)

    normalization_mean = torch.tensor(CNN_NORMALIZATION_MEAN, device=device)
    normalization_std = torch.tensor(CNN_NORMALIZATION_STD, device=device)
    return StyleTransferRuntime(
        device=device,
        cnn=cnn,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )


class Normalization(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


class StyleFeaturesExtractor:
    """Build the truncated VGG backbone and cache style targets."""

    def __init__(
        self,
        cnn: nn.Sequential,
        style_img: torch.Tensor,
        normalization_mean: torch.Tensor,
        normalization_std: torch.Tensor,
        style_layer_idxs: list[int] | None = None,
        content_layer_idxs: list[int] | None = None,
        device: torch.device | None = None,
    ):
        self.cnn = cnn
        self.style_img = style_img
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.style_layer_idxs = style_layer_idxs or DEFAULT_STYLE_LAYER_IDXS
        self.content_layer_idxs = content_layer_idxs or DEFAULT_CONTENT_LAYER_IDXS
        self.device = device or style_img.device
        self.model: nn.Sequential | None = None
        self.style_features: dict[int, torch.Tensor] = {}
        self.build_model_and_extract_features()

    def build_model_and_extract_features(self) -> None:
        normalization = Normalization(
            self.normalization_mean,
            self.normalization_std,
        ).to(self.device)

        max_required_idx = max(max(self.style_layer_idxs), max(self.content_layer_idxs))
        model = nn.Sequential()
        model.add_module("-1", normalization)

        style_features: dict[int, torch.Tensor] = {}
        current = self.style_img

        for idx, layer in enumerate(self.cnn.children()):
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)

            model.add_module(str(idx), layer)
            current = layer(current)

            if idx in self.style_layer_idxs:
                style_features[idx] = get_vgg_gram_matrix(current).detach()

            if idx >= max_required_idx:
                break

        self.model = model
        self.style_features = style_features


class ContentFeaturesExtractor:
    """Reuse the shared truncated model and cache content targets."""

    def __init__(
        self,
        model: nn.Sequential,
        content_img: torch.Tensor,
        content_layer_idxs: list[int] | None = None,
    ):
        self.model = model
        self.content_img = content_img
        self.content_layer_idxs = content_layer_idxs or DEFAULT_CONTENT_LAYER_IDXS
        self.feature_maps: dict[int, torch.Tensor] = {}
        self.extract_feature()

    def extract_feature(self) -> None:
        feature_maps: dict[int, torch.Tensor] = {}
        current = self.content_img

        for name, layer in self.model.named_children():
            current = layer(current)

            if name == "-1":
                continue

            idx = int(name)
            if idx in self.content_layer_idxs:
                feature_maps[idx] = current.detach()

        self.feature_maps = feature_maps

    def get_fm(self) -> dict[int, torch.Tensor]:
        return self.feature_maps
