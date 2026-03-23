"""Image loading and tensor conversion utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

IMAGE_SIZE = 256
UNLOADER = transforms.ToPILImage()


def get_image_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Center-crop and resize images for a consistent demo input size."""

    return transforms.Compose(
        [
            transforms.Lambda(
                lambda image: ImageOps.fit(
                    image,
                    (image_size, image_size),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
            ),
            transforms.ToTensor(),
        ]
    )


def _open_image(image_source: str | Path | Image.Image) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    return Image.open(image_source).convert("RGB")


def load_image(
    image_source: str | Path | Image.Image,
    transform: transforms.Compose | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load an image into a 4D tensor on the requested device."""

    image = _open_image(image_source)
    prepared = (transform or get_image_transform())(image).unsqueeze(0)
    if device is not None:
        return prepared.to(device)
    return prepared


def prepare_display_image(
    image_source: str | Path | Image.Image,
    image_size: int = IMAGE_SIZE,
) -> Image.Image:
    """Apply the demo crop/resize pipeline and return a PIL image for display."""

    tensor = load_image(
        image_source=image_source,
        transform=get_image_transform(image_size),
        device=None,
    )
    return unload_image(tensor)


def unload_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a BCHW tensor back into a PIL image."""

    image = tensor.detach().cpu().clamp(0, 1).clone().squeeze(0)
    return UNLOADER(image)


def get_vgg_gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    """Compute the normalized Gram matrix for a VGG feature map tensor."""

    batch, channels, height, width = input_tensor.shape
    features = input_tensor.view(batch, channels, height * width)
    return torch.bmm(features, features.transpose(1, 2)) / (channels * height * width)
