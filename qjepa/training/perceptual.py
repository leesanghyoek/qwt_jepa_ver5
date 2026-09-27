"""VGG16 feature loss (Johnson et al., 2016) for phase 2.

Per-pixel losses answer uncertain texture with its average: a smooth patch.
Comparing the restored and clean frames in the feature space of an ImageNet
VGG16 -- edges and blobs at relu1_2, corners and short strokes at relu2_2,
textures at relu3_3 -- penalises that smoothness, because smooth and textured
patches look different there even when their per-pixel errors are similar.

The trade-off, stated plainly: where the input really lost the detail, this term
rewards *plausible* texture, not necessarily the true one. It is not adversarial.

Each layer's L1 is divided by the mean |activation| of the clean frame at that
layer, so layers with large activations do not dominate and the weight reads as a
relative error. The network is frozen, always in eval mode, and kept out of the
restoration system: checkpoints stay the size they were.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Indices in torchvision's vgg16().features after which relu1_2, relu2_2, relu3_3 fire.
LAYERS = (3, 8, 15)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_vgg16_features(pretrained: bool = True) -> nn.Sequential:
    """VGG16 up to relu3_3. The ImageNet weights (528 MB file) download once per machine."""
    from torchvision.models import VGG16_Weights, vgg16

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    return model.features[: LAYERS[-1] + 1]


class PerceptualLoss(nn.Module):
    def __init__(self, features: nn.Sequential | None = None) -> None:
        super().__init__()
        self.features = features if features is not None else load_vgg16_features()
        self.features.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.train(False)

    def train(self, mode: bool = True):
        return super().train(False)

    def _activations(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = (image - self.mean) / self.std
        taken = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in LAYERS:
                taken.append(x)
        return taken

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            wanted = self._activations(target.float())
        got = self._activations(predicted.float())
        terms = [(a - b).abs().mean() / b.abs().mean().clamp_min(1e-6) for a, b in zip(got, wanted)]
        return torch.stack(terms).mean()
