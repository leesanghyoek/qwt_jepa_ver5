from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.color_edge import chroma, color_error, downsample, excess_roughness, luminance


def _ssim(restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Channel-wise local SSIM with data_range=1."""
    size = min(11, restored.shape[-2], restored.shape[-1])
    if size % 2 == 0:
        size -= 1
    coordinate = torch.arange(size, dtype=restored.dtype, device=restored.device) - size // 2
    sigma = max(0.5, 1.5 * size / 11.0)
    one_dimensional = torch.exp(-coordinate.square() / (2.0 * sigma * sigma))
    one_dimensional /= one_dimensional.sum()
    kernel = torch.outer(one_dimensional, one_dimensional)
    kernel = kernel.expand(restored.shape[1], 1, size, size)
    padding = size // 2
    mean_x = F.conv2d(restored, kernel, padding=padding, groups=restored.shape[1])
    mean_y = F.conv2d(clean, kernel, padding=padding, groups=clean.shape[1])
    variance_x = F.conv2d(restored.square(), kernel, padding=padding, groups=restored.shape[1]) - mean_x.square()
    variance_y = F.conv2d(clean.square(), kernel, padding=padding, groups=clean.shape[1]) - mean_y.square()
    covariance = F.conv2d(restored * clean, kernel, padding=padding, groups=restored.shape[1]) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    )
    return score.mean()


def _luma_spectrum(image: torch.Tensor) -> torch.Tensor:
    """Windowed complex luma spectrum per image, [B, H, W]."""
    luma = image[:, 0] * 0.299 + image[:, 1] * 0.587 + image[:, 2] * 0.114
    height, width = luma.shape[-2:]
    window = torch.outer(
        torch.hann_window(height, periodic=False, dtype=luma.dtype, device=luma.device),
        torch.hann_window(width, periodic=False, dtype=luma.dtype, device=luma.device),
    )
    luma = (luma - luma.mean(dim=(-2, -1), keepdim=True)) * window
    return torch.fft.fft2(luma)


def spectral_ratios(restored: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    """Edge-band and stripe power of ``restored``, each relative to ``clean``, and
    how much of the clean detail sits in place.

    PSNR/SSIM cannot tell real sharpening from a fixed high-frequency pattern.
    ``image_edge_power`` is the power at periods of 4-16 px, where edges and
    texture live: 1.0 means as much as the clean frame. ``image_stripe_power`` is
    the power on the Nyquist lines (period 2 px), where a smooth offset in a QWT
    detail band lands in full: LH -> horizontal stripes, HL -> vertical, HH ->
    checkerboard. It is floored at 0.1% of the clean frame's power so smooth
    frames do not divide by ~0. Well above 1 means stripes the clean frame lacks.

    Power counts noise and invented texture as detail. The ``_in_place`` ratios do
    not: Re sum(F_restored * conj(F_clean)) / sum(|F_clean|^2) over a band -- the
    part of the clean frame's detail the output reproduces with the right phase,
    i.e. at the right place. Noise and made-up texture are uncorrelated with the
    clean frame and add ~0. 1.0 = all of it, 0 = none. ``image_fine_detail_in_place``
    covers periods of 2-4 px: small and distant objects. Same 0.1% floor.
    """
    got_spectrum, want_spectrum = _luma_spectrum(restored), _luma_spectrum(clean)
    got, want = got_spectrum.abs().square().sum(0), want_spectrum.abs().square().sum(0)
    shared = (got_spectrum * want_spectrum.conj()).real.sum(0)
    height, width = got.shape
    fy = torch.fft.fftfreq(height, device=got.device).abs()[:, None]
    fx = torch.fft.fftfreq(width, device=got.device).abs()[None, :]
    band = torch.maximum(fy, fx)
    edge = (band >= 1 / 16) & (band < 1 / 4)
    fine = band >= 1 / 4
    stripe = (fy >= 0.5 - 2 / height) | (fx >= 0.5 - 2 / width)
    floor = 1e-3 * want.sum()
    return {
        "image_edge_power": float(got[edge].sum() / want[edge].sum().clamp_min(1e-12)),
        "image_stripe_power": float(got[stripe].sum() / (want[stripe].sum() + floor).clamp_min(1e-12)),
        "image_edge_in_place": float(shared[edge].sum() / (want[edge].sum() + floor).clamp_min(1e-12)),
        "image_fine_detail_in_place": float(shared[fine].sum() / (want[fine].sum() + floor).clamp_min(1e-12)),
    }


def color_vividness(restored: torch.Tensor, clean: torch.Tensor, scale: int = 4) -> dict[str, float]:
    """How vivid and how contrasty ``restored`` is, with the clean frame's values.

    ``image_saturation`` is the mean chroma magnitude sqrt(Cb^2 + Cr^2) and
    ``image_contrast`` the standard deviation of luminance, both after averaging
    ``scale`` x ``scale`` blocks so colour noise and fine texture do not count.
    The ``_clean`` keys hold the clean frame's values; the report divides the
    means, so 1.00 = as vivid / as contrasty as clean. L1 on chroma pulls the hue
    towards grey wherever it is uncertain, and the colour error cannot tell that
    washed-out look from a colour cast -- these can.
    """
    values = {}
    for name, image in (("", restored), ("_clean", clean)):
        small = downsample(image, scale)
        values[f"image_saturation{name}"] = float(chroma(small).square().sum(1).sqrt().mean())
        values[f"image_contrast{name}"] = float(luminance(small).flatten(1).std(dim=1).mean())
    return values


def image_metrics(restored: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    restored = restored.clamp(0.0, 1.0)
    clean = clean.clamp(0.0, 1.0)
    error = restored - clean
    mae = error.abs().mean()
    mse = error.square().mean()
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {
        "image_mae": float(mae),
        "image_psnr_db": float(psnr),
        "image_ssim": float(_ssim(restored, clean)),
        **spectral_ratios(restored, clean),
        # Chroma after 4x4 averaging: colour cast and colour noise, not sharpness.
        "image_color_error": float(color_error(restored, clean)),
        **color_vividness(restored, clean),
        # Grain, ringing and blur spill beyond the clean frame's own gradients, /255.
        # 0 = as smooth as clean where clean is flat; lower is smoother.
        "image_excess_roughness": float(excess_roughness(luminance(restored), luminance(clean))) * 255.0,
    }


def imu_metrics(
    restored_phys: torch.Tensor,
    clean_phys: torch.Tensor,
    timestamps: torch.Tensor | None = None,
) -> dict[str, float | list[float]]:
    error = restored_phys - clean_phys
    rmse_axis = error.square().mean(dim=(0, 2)).sqrt()
    mae_axis = error.abs().mean(dim=(0, 2))
    bias_axis = error.mean(dim=(0, 2))
    metrics: dict[str, float | list[float]] = {
        "imu_rmse_axis": rmse_axis.detach().cpu().tolist(),
        "imu_mae_axis": mae_axis.detach().cpu().tolist(),
        "imu_bias_axis": bias_axis.detach().cpu().tolist(),
        "accel_rmse": float(error[:, :3].square().mean().sqrt()),
        "gyro_rmse": float(error[:, 3:].square().mean().sqrt()),
    }
    if timestamps is not None:
        dt = torch.diff(timestamps, dim=-1)
        if (dt <= 0).any():
            raise ValueError("Timestamps must increase for IMU variation metrics")
        restored_rate = torch.diff(restored_phys, dim=-1) / dt[:, None, :]
        clean_rate = torch.diff(clean_phys, dim=-1) / dt[:, None, :]
        rate_error = restored_rate - clean_rate
        metrics["accel_variation_rmse"] = float(rate_error[:, :3].square().mean().sqrt())
        metrics["gyro_variation_rmse"] = float(rate_error[:, 3:].square().mean().sqrt())
    return metrics


def pooled_effective_rank(feature: torch.Tensor) -> float:
    if feature.ndim == 4:
        pooled = feature.mean(dim=(-2, -1))
    elif feature.ndim == 3:
        pooled = feature.mean(dim=-1)
    else:
        raise ValueError("Expected a dense image or IMU feature")
    pooled = pooled.float() - pooled.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(pooled)
    if singular.sum() <= 1e-12:
        return 0.0
    probability = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy))


def latent_diagnostics(feature: torch.Tensor) -> dict[str, float]:
    tokens = feature.flatten(2).transpose(1, 2)
    same_position_std = tokens.std(dim=0, unbiased=True).mean()
    normalized = F.layer_norm(tokens.float(), (tokens.shape[-1],))
    normalized_feature = normalized.transpose(1, 2).reshape_as(feature)
    return {
        "raw_rms": float(feature.float().square().mean().sqrt()),
        "same_position_std": float(same_position_std),
        "pooled_effective_rank": pooled_effective_rank(feature),
        "normalized_same_position_std": float(normalized.std(dim=0, unbiased=True).mean()),
        "normalized_pooled_effective_rank": pooled_effective_rank(normalized_feature),
    }
