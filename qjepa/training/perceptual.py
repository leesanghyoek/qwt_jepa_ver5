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
    """``crop`` > 0 scores one random ``crop`` x ``crop`` window per call (the same
    window in prediction and target) instead of the whole frame: VGG then costs
    (crop / side)^2 of the full frame, 1/4 for 128 of 256."""

    def __init__(self, features: nn.Sequential | None = None, crop: int = 0) -> None:
        super().__init__()
        self.crop = int(crop)
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
        height, width = predicted.shape[-2:]
        if self.crop and (height > self.crop or width > self.crop):
            # Global RNG: its state is in every checkpoint, so resume draws the same windows.
            top = int(torch.randint(0, height - self.crop + 1, (1,)))
            left = int(torch.randint(0, width - self.crop + 1, (1,)))
            predicted = predicted[..., top:top + self.crop, left:left + self.crop]
            target = target[..., top:top + self.crop, left:left + self.crop]
        with torch.no_grad():
            wanted = self._activations(target.float())
        got = self._activations(predicted.float())
        # The ratio in fp32 even when the activations came out of fp16 autocast.
        terms = [(a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
                 for a, b in zip(got, wanted)]
        return torch.stack(terms).mean()
