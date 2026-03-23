"""Optimization routines for VGG19 neural style transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from src.image_utils import unload_image
from src.losses import ContentLoss, StyleLoss
from src.model import ContentFeaturesExtractor, StyleFeaturesExtractor


@dataclass
class TransferUpdate:
    """A streamed snapshot from optimization."""

    iteration: int
    image: Image.Image
    tensor: torch.Tensor
    style_loss: float
    normalized_style_loss: float
    content_loss: float


class TransferCancelled(RuntimeError):
    """Raised when a newer generation request supersedes the current one."""


def get_input_optimizer(input_img: torch.Tensor) -> optim.Optimizer:
    return optim.LBFGS([input_img.requires_grad_()])


def get_new_model_and_losses(
    style_features_extractor: StyleFeaturesExtractor,
    content_features_extractor: ContentFeaturesExtractor,
) -> tuple[nn.Sequential, list[StyleLoss], ContentLoss]:
    """Rebuild the optimization model and insert the loss modules."""

    model = style_features_extractor.model
    if model is None:
        raise ValueError("Style feature extractor did not build a model.")

    new_model = nn.Sequential()
    style_losses: list[StyleLoss] = []
    content_loss_module: ContentLoss | None = None

    style_idxs = set(style_features_extractor.style_features.keys())
    content_idxs = list(content_features_extractor.feature_maps.keys())
    if not content_idxs:
        raise ValueError("No content feature maps were extracted.")

    last_content_idx = max(content_idxs)

    for name, layer in model.named_children():
        if isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False)

        new_model.add_module(name, layer)

        if name == "-1":
            continue

        idx = int(name)

        if idx in style_idxs:
            target_gram = style_features_extractor.style_features[idx]
            style_loss = StyleLoss(target_gram)
            new_model.add_module(f"style_loss_{idx}", style_loss)
            style_losses.append(style_loss)

        if idx == last_content_idx:
            target_feature = content_features_extractor.feature_maps[idx]
            content_loss = ContentLoss(target_feature)
            new_model.add_module(f"content_loss_{idx}", content_loss)
            content_loss_module = content_loss

    if content_loss_module is None:
        raise ValueError("No content loss module was inserted into the model.")

    return new_model, style_losses, content_loss_module


def run_style_transfer(
    style_features_extractor: StyleFeaturesExtractor,
    content_features_extractor: ContentFeaturesExtractor,
    style_weight: float,
    content_weight: float,
    num_steps: int = 2000,
) -> torch.Tensor:
    """Notebook-compatible final-only optimization loop."""

    model, style_losses, content_loss_module = get_new_model_and_losses(
        style_features_extractor=style_features_extractor,
        content_features_extractor=content_features_extractor,
    )
    content_img = content_features_extractor.content_img
    generated_img = content_img.clone().to(content_img.device)

    with torch.no_grad():
        generated_img.clamp_(0, 1)

    optimizer = get_input_optimizer(generated_img)
    model(generated_img)
    initial_style_loss = sum(style_loss.loss for style_loss in style_losses).item()
    eps = 1e-8
    run = [0]

    while run[0] <= num_steps:
        def closure() -> torch.Tensor:
            with torch.no_grad():
                generated_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(generated_img)
            style_loss = sum(loss.loss for loss in style_losses) / len(style_losses)
            normalized_style_loss = style_loss / (initial_style_loss + eps)
            content_loss = content_loss_module.loss
            total = style_weight * normalized_style_loss + content_weight * content_loss
            total.backward()

            run[0] += 1
            return total

        optimizer.step(closure)

        with torch.no_grad():
            generated_img.clamp_(0, 1)

    return generated_img


def _iter_style_transfer_inter(
    style_features_extractor: StyleFeaturesExtractor,
    content_features_extractor: ContentFeaturesExtractor,
    style_weight: float,
    content_weight: float,
    num_steps: int,
    show_every: int,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[TransferUpdate]:
    model, style_losses, content_loss_module = get_new_model_and_losses(
        style_features_extractor=style_features_extractor,
        content_features_extractor=content_features_extractor,
    )

    content_img = content_features_extractor.content_img
    generated_img = content_img.clone().to(content_img.device)
    generated_img = generated_img + 0.1 * torch.randn_like(content_img).to(content_img.device)

    with torch.no_grad():
        generated_img.clamp_(0, 1)

    optimizer = get_input_optimizer(generated_img)
    model(generated_img)
    initial_style_loss = sum(style_loss.loss for style_loss in style_losses).item()
    eps = 1e-8
    run = [0]

    def is_cancelled() -> bool:
        return should_stop is not None and should_stop()

    try:
        while run[0] < num_steps:
            if is_cancelled():
                raise TransferCancelled

            pending_snapshots: list[TransferUpdate] = []

            def closure() -> torch.Tensor:
                if is_cancelled():
                    raise TransferCancelled

                with torch.no_grad():
                    generated_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(generated_img)

                style_loss = sum(loss.loss for loss in style_losses) / len(style_losses)
                normalized_style_loss = style_loss / (initial_style_loss + eps)
                content_loss = content_loss_module.loss
                total = style_weight * normalized_style_loss + content_weight * content_loss
                total.backward()

                run[0] += 1
                if run[0] % show_every == 0 or run[0] == 1 or run[0] == num_steps:
                    pending_snapshots.append(
                        TransferUpdate(
                            iteration=run[0],
                            image=unload_image(generated_img).copy(),
                            tensor=generated_img.detach().cpu().clone(),
                            style_loss=style_loss.item(),
                            normalized_style_loss=normalized_style_loss.item(),
                            content_loss=content_loss.item(),
                        )
                    )
                return total

            optimizer.step(closure)

            if is_cancelled():
                raise TransferCancelled

            with torch.no_grad():
                generated_img.clamp_(0, 1)

            for snapshot in pending_snapshots:
                if is_cancelled():
                    raise TransferCancelled
                yield snapshot
    except TransferCancelled:
        return


def run_style_transfer_inter(
    style_features_extractor: StyleFeaturesExtractor,
    content_features_extractor: ContentFeaturesExtractor,
    style_weight: float,
    content_weight: float,
    num_steps: int = 2000,
    show_every: int = 100,
    display_intermediate: bool = False,
    save_intermediate: bool = True,
    stream: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[TransferUpdate] | tuple[torch.Tensor, list[tuple[int, Image.Image]]]:
    """Notebook-derived intermediate loop with optional streaming mode."""

    updates = _iter_style_transfer_inter(
        style_features_extractor=style_features_extractor,
        content_features_extractor=content_features_extractor,
        style_weight=style_weight,
        content_weight=content_weight,
        num_steps=num_steps,
        show_every=show_every,
        should_stop=should_stop,
    )
    if stream:
        return updates

    snapshots: list[tuple[int, Image.Image]] = []
    final_tensor: torch.Tensor | None = None
    cancelled = False

    for update in updates:
        final_tensor = update.tensor.to(content_features_extractor.content_img.device)

        if save_intermediate:
            snapshots.append((update.iteration, update.image.copy()))

        if display_intermediate:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(5, 5))
            plt.imshow(update.image)
            plt.title(f"Iteration {update.iteration}")
            plt.axis("off")
            plt.show()

    if final_tensor is None:
        cancelled = should_stop is not None and should_stop()
        if cancelled:
            final_tensor = content_features_extractor.content_img.detach().clone()
        else:
            raise ValueError("Style transfer produced no updates. Check num_steps and show_every.")

    return final_tensor, snapshots
