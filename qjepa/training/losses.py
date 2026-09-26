"""Losses for the strictly separated latent and restoration phases."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.color_edge import chroma, luminance_gradient_l1, split_targets
from ..models.predictors import image_tokens, imu_tokens


def layer_norm_no_affine(tokens: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.layer_norm(tokens, (tokens.shape[-1],), weight=None, bias=None, eps=eps)


def jepa_latent_loss(
    predicted_image: torch.Tensor,
    predicted_imu: torch.Tensor,
    target_image_dense: torch.Tensor,
    target_imu_dense: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_image = image_tokens(target_image_dense).detach()
    target_imu = imu_tokens(target_imu_dense).detach()
    image_term = F.smooth_l1_loss(
        layer_norm_no_affine(predicted_image), layer_norm_no_affine(target_image), beta=beta
    )
    imu_term = F.smooth_l1_loss(
        layer_norm_no_affine(predicted_imu), layer_norm_no_affine(target_imu), beta=beta
    )
    return 0.5 * (image_term + imu_term), image_term, imu_term


def _dense_tokens(dense: torch.Tensor) -> torch.Tensor:
    return dense.flatten(2).transpose(1, 2)


@torch.no_grad()
def jepa_diagnostics(
    predicted: torch.Tensor,
    target_dense: torch.Tensor,
    mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """What the raw JEPA number cannot say on its own.

    * ``normalized`` -- JEPA / the JEPA of one constant token (the mean target
      token): what a collapsed predictor scores. 1 = no better than collapse,
      0 = perfect. A latent that carries more gets harder targets and a higher
      raw JEPA; this ratio does not reward shedding information, so it is the one
      to compare across runs.
    * ``cosine`` -- mean cosine between LayerNorm'd prediction and target.
    * ``visible`` / ``masked`` -- JEPA over the tokens the predictor saw and the
      ones it had to infer, when a mask is given.
    """
    target = layer_norm_no_affine(_dense_tokens(target_dense).float())
    prediction = layer_norm_no_affine(predicted.float())
    per_token = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none").mean(-1)
    constant = layer_norm_no_affine(target.mean(dim=(0, 1), keepdim=True)).expand_as(target)
    collapse = F.smooth_l1_loss(constant, target, beta=beta)
    result = {
        "normalized": per_token.mean() / collapse.clamp_min(1e-12),
        "cosine": F.cosine_similarity(prediction, target, dim=-1).mean(),
    }
    if mask is not None:
        result["visible"] = per_token[~mask].mean()
        result["masked"] = per_token[mask].mean()
    return result


def jepa_fine_loss(predicted_dense: torch.Tensor, target_dense: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """JEPA against the teacher's previous stage: twice the grid's resolution, so
    each 16x16-pixel token must also say what its four 8x8 quarters hold."""
    return F.smooth_l1_loss(
        layer_norm_no_affine(_dense_tokens(predicted_dense)),
        layer_norm_no_affine(_dense_tokens(target_dense).detach()),
        beta=beta,
    )


def jepa_coarse_loss(
    predicted_tokens: torch.Tensor, target_dense: torch.Tensor, pool: int, beta: float = 1.0
) -> torch.Tensor:
    """JEPA after averaging ``pool`` x ``pool`` tokens of prediction and target.

    Token errors that cancel over a region leave it; errors shared across the
    region -- wrong layout, wrong regional brightness -- stay. It weights exactly
    what a masked block has to be inferred from.
    """
    predicted_dense = predicted_tokens.transpose(1, 2).reshape(target_dense.shape)
    return F.smooth_l1_loss(
        layer_norm_no_affine(_dense_tokens(F.avg_pool2d(predicted_dense, pool))),
        layer_norm_no_affine(_dense_tokens(F.avg_pool2d(target_dense.detach(), pool))),
        beta=beta,
    )


def dense_positions(feature: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
    """Convert channel-first dense features to aligned [B,K,D] positions."""
    if feature.ndim == 4:
        tokens = image_tokens(feature)
    elif feature.ndim == 3:
        tokens = imu_tokens(feature)
    else:
        raise ValueError(f"Expected [B,D,H,W] or [B,D,L], got {tuple(feature.shape)}")
    return tokens if indices is None else tokens.index_select(1, indices.to(tokens.device))


def variance_covariance_loss(
    positions: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
    pooled_covariance: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style statistics over B at the same spatial/temporal position.

    Each position is centred over the batch. The covariance term then either
    estimates one D x D matrix per position from the B samples (``pooled_covariance``
    False, the original form) or one matrix from all B*K centred samples.

    The per-position form is rank <= B-1: with B = 8 and D = 128 it is almost all
    sampling noise. Measured on perfectly uncorrelated features with std 0.82,
    it reads 8.24 where the true value is 0, and truly correlated features (half
    the variance shared) read only 18.0 -- it barely tells them apart. Its noise
    floor also grows with std^4, so its gradient mostly shrinks every feature,
    fighting the variance term. Pooling the centred samples of K = 16 positions
    reads 0.40 and 8.67 on the same two cases (K = 64: 0.13 on the uncorrelated
    one): it measures correlation, not noise.
    """
    if positions.ndim != 3 or positions.shape[0] < 2:
        raise ValueError("Expected H[B,K,D] with B >= 2")
    values = positions.float()
    centred = values - values.mean(dim=0, keepdim=True)
    variance = centred.square().sum(dim=0) / (values.shape[0] - 1)
    variance_loss = F.relu(gamma - torch.sqrt(variance + eps)).mean()
    dim = values.shape[-1]
    if pooled_covariance:
        samples = centred.reshape(-1, dim)
        # K per-position means were removed: K * (B - 1) degrees of freedom.
        covariance = samples.T @ samples / (values.shape[1] * (values.shape[0] - 1))
        off_diagonal = covariance.masked_fill(torch.eye(dim, dtype=torch.bool, device=values.device), 0.0)
        return variance_loss, off_diagonal.square().sum() / dim
    covariance = torch.einsum("bkd,bke->kde", centred, centred) / (values.shape[0] - 1)
    diagonal = torch.eye(dim, dtype=torch.bool, device=values.device)[None]
    off_diagonal = covariance.masked_fill(diagonal, 0.0)
    covariance_loss = off_diagonal.square().sum(dim=(-2, -1)).mean() / dim
    return variance_loss, covariance_loss


def detail_bands(coefficients: torch.Tensor, dim: int) -> torch.Tensor:
    """Tach rieng phan chi tiet — tuc duong net.

    QWT xep he so thanh [mau, bang, thanh phan] roi dep thanh 48 kenh; bang 0 la
    LL con bang 1..3 la LH/HL/HH. Haar cho IMU xep approx truoc, detail sau.
    Tra ve [B, bang, ...] de goi ben ngoai gop theo bang khi can.
    """
    if dim != 2:
        half = coefficients.shape[1] // 2
        return coefficients[:, half:]
    batch, channels, height, width = coefficients.shape
    if channels % 16:
        raise ValueError(f"Expected 16 bands*components per colour, got {channels}")
    shape = (batch, channels // 16, 4, 4, height, width)
    # [B, mau, bang, thanh phan, H, W] -> dua truc BANG len truoc de gop rieng.
    return coefficients.reshape(shape)[:, :, 1:].transpose(1, 2)


def detail_band_l1(predicted: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"Coefficients {tuple(predicted.shape)} != {tuple(target.shape)}")
    return F.l1_loss(detail_bands(predicted, dim), detail_bands(target, dim))


IMAGE_DETAIL_LOSSES = ("coefficient", "modulus")


def detail_modulus(coefficients: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Quaternion modulus of the image detail bands, ``[B, colour, band, H, W]``.

    ``|q| = sqrt(real^2 + i^2 + j^2 + k^2)`` over the four dual-tree components. For
    a genuine Hilbert pair this is the local edge strength and it barely moves when
    the edge moves by a fraction of a pixel -- the property the per-coefficient
    loss lacks. ``eps`` keeps the gradient finite in flat regions, where |q| -> 0.
    """
    batch, channels, height, width = coefficients.shape
    if channels % 16:
        raise ValueError(f"Expected 16 bands*components per colour, got {channels}")
    packed = coefficients.reshape(batch, channels // 16, 4, 4, height, width)[:, :, 1:]
    return (packed.square().sum(dim=3) + eps).sqrt()


def detail_modulus_l1(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between detail-band quaternion moduli -- the image detail term that does
    not reward blur.

    Where an edge's exact position is uncertain, the clean detail COEFFICIENT
    could be positive, negative or anything between, so an L1 on coefficients is
    minimised by shrinking toward zero: measured on real frames with edges
    uncertain by +-0.75 px, it prefers detail at 0.5x true strength, and at 0.25x
    on the actual corrupted input. The modulus of that same edge is nearly the
    same wherever it sits, so its conditional median is the true edge strength:
    1.0x and 1.25x on the same two measurements. Blurring stops being the safe
    answer. Phase is left unconstrained here on purpose; the pixel L1 and the
    residual on the input coefficients already pin polarity and placement.
    """
    if predicted.shape != target.shape:
        raise ValueError(f"Coefficients {tuple(predicted.shape)} != {tuple(target.shape)}")
    return F.l1_loss(detail_modulus(predicted), detail_modulus(target))


IMAGE_DETAIL_SOURCES = ("decoder_coefficients", "restored_image")


def invisible_detail_fraction(coefficients: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Share of the detail-band energy in ``coefficients`` the image does not show.

    The QWT keeps four trees and synthesis averages them, so coefficients are 4x
    redundant: whatever lies in the null space of synthesis leaves the image
    untouched. ``visible`` is ``analysis(synthesis(coefficients))``, the part that
    reaches the pixels. Input coefficients sit at ~0; a decoder whose detail terms
    are scored on its own output can drive this up, lowering those terms without
    changing the image.
    """
    hidden = detail_bands(coefficients - visible, dim=2).square().sum()
    return hidden / detail_bands(coefficients, dim=2).square().sum().clamp_min(1e-12)


def detail_energy_gap(predicted: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    """Lech nang luong duong net, tinh theo tung bang.

    L1 tren tung he so noi "moi he so phai gan dung", va trung vi co dieu kien cua
    no la 0 — nen khi duoc phep cham vao bang chi tiet, model chon CO NHO he so:
    o cho co the co canh, co ve 0 giam sai so chac chan, giu canh thi rui ro. Do
    dung la hanh vi delta_report do duoc (LH/HL te di sau khi model bat dau cham
    vao chung).

    So hang nay noi mot dieu khac han: TONG nang luong duong net phai bang anh
    sach. Co nho vi pham truc tiep, bat ke tung he so dung hay sai. No la khop
    mo-men chu khong phai doi khang — khong can mang thu hai, khong co rui ro mat
    can bang.

    Khong dam bao net DUNG CHO. No chi cam loi thoat "lam phang cho an toan".
    """
    if predicted.shape != target.shape:
        raise ValueError(f"Coefficients {tuple(predicted.shape)} != {tuple(target.shape)}")
    got, want = detail_bands(predicted, dim), detail_bands(target, dim)
    # Gop moi truc tru batch va bang: moi bang co mot nang luong rieng, va LH/HL
    # hanh xu khac HH nen khong duoc tron chung.
    axes = tuple(range(2, got.ndim))
    return (got.square().mean(axes).sqrt() - want.square().mean(axes).sqrt()).abs().mean()


def color_edge_split_loss(
    base: torch.Tensor,
    light: torch.Tensor,
    detail: torch.Tensor,
    restored: torch.Tensor,
    clean: torch.Tensor,
    *,
    color_scale: int,
    illumination_scale: int,
    color_weight: float,
    edge_weight: float,
    gradient_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Separate targets for the two branches of the split decoder.

    Colour: chroma of the base and the illumination against the same pieces of
    the clean frame -- L1, whose mean answer is the right colour. Edges: the
    luminance detail against the clean frame's. Optionally the luminance
    gradient of the whole output, which scores edge slopes directly.
    """
    with torch.no_grad():
        base_t, light_t, detail_t = split_targets(clean, color_scale, illumination_scale)
    color = F.l1_loss(chroma(base), chroma(base_t)) + F.l1_loss(light, light_t)
    edge = F.l1_loss(detail, detail_t)
    total = color_weight * color + edge_weight * edge
    parts = {"image_color_l1": color, "image_edge_detail_l1": edge}
    if gradient_weight > 0:
        gradient = luminance_gradient_l1(restored, clean)
        total = total + gradient_weight * gradient
        parts["image_edge_gradient_l1"] = gradient
    return total, parts


def first_difference_l1(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 tren sai phan bac mot doc truc thoi gian — tuc do rung.

    Cac so hang con lai cham diem tung mau doc lap, nen mot du doan bam dung bien
    do van co the giat tung mau mot ma khong bi phat gi. Day chinh la dai luong
    ma evaluate goi la variation_rmse: truoc day no duoc DO nhung khong duoc
    TOI UU, va do lieu cho thay no gan nhu khong nhuc nhich (-0,7% so voi input).
    """
    if predicted.shape != target.shape:
        raise ValueError(f"IMU {tuple(predicted.shape)} != {tuple(target.shape)}")
    if predicted.shape[-1] < 2:
        raise ValueError("Need at least two samples along time for a first difference")
    return F.l1_loss(predicted.diff(dim=-1), target.diff(dim=-1))


def phase2_reconstruction_loss(
    image_restored: torch.Tensor,
    image_clean: torch.Tensor,
    imu_restored_normalized: torch.Tensor,
    imu_clean_normalized: torch.Tensor,
    beta: float = 1.0,
    image_coefficients: torch.Tensor | None = None,
    image_coefficient_target: torch.Tensor | None = None,
    imu_coefficients: torch.Tensor | None = None,
    imu_coefficient_target: torch.Tensor | None = None,
    detail_weight: float = 0.0,
    variation_weight: float = 0.0,
    detail_energy_weight: float = 0.0,
    image_detail_loss: str = "coefficient",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if image_detail_loss not in IMAGE_DETAIL_LOSSES:
        raise ValueError(f"image_detail_loss must be one of {IMAGE_DETAIL_LOSSES}")
    image = F.l1_loss(image_restored, image_clean)
    accel = F.smooth_l1_loss(
        imu_restored_normalized[:, :3], imu_clean_normalized[:, :3], beta=beta
    )
    gyro = F.smooth_l1_loss(
        imu_restored_normalized[:, 3:], imu_clean_normalized[:, 3:], beta=beta
    )
    total = image + 0.5 * (accel + gyro)
    parts = {"image_l1": image, "imu_accel_smooth_l1": accel, "imu_gyro_smooth_l1": gyro}
    if detail_weight > 0 and image_coefficients is not None:
        # L1 tren pixel toi uu ve trung vi co dieu kien, ma nghiem do chinh la
        # anh mo. So hang nay phat rieng phan duong net bi mat.
        imu_detail = detail_band_l1(imu_coefficients, imu_coefficient_target, dim=1)
        if image_detail_loss == "modulus":
            image_detail = detail_modulus_l1(image_coefficients, image_coefficient_target)
            parts["image_detail_modulus_l1"] = image_detail
            # The coefficient L1 is still logged, outside the graph, so runs with
            # either term stay comparable on the number every earlier run reports.
            with torch.no_grad():
                parts["image_detail_l1"] = detail_band_l1(
                    image_coefficients, image_coefficient_target, dim=2)
        else:
            image_detail = detail_band_l1(image_coefficients, image_coefficient_target, dim=2)
            parts["image_detail_l1"] = image_detail
        total = total + detail_weight * (image_detail + 0.5 * imu_detail)
        parts["imu_detail_l1"] = imu_detail
    if detail_energy_weight > 0 and image_coefficients is not None:
        image_energy = detail_energy_gap(image_coefficients, image_coefficient_target, dim=2)
        imu_energy = detail_energy_gap(imu_coefficients, imu_coefficient_target, dim=1)
        total = total + detail_energy_weight * (image_energy + 0.5 * imu_energy)
        parts["image_detail_energy"] = image_energy
        parts["imu_detail_energy"] = imu_energy
    if variation_weight > 0:
        accel_variation = first_difference_l1(
            imu_restored_normalized[:, :3], imu_clean_normalized[:, :3]
        )
        gyro_variation = first_difference_l1(
            imu_restored_normalized[:, 3:], imu_clean_normalized[:, 3:]
        )
        total = total + variation_weight * (accel_variation + gyro_variation)
        parts["imu_accel_variation_l1"] = accel_variation
        parts["imu_gyro_variation_l1"] = gyro_variation
    return total, parts



def phase1_reconstruction_loss(
    image_predicted: torch.Tensor,
    image_target: torch.Tensor,
    imu_predicted: torch.Tensor,
    imu_target: torch.Tensor,
    detail_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Neo latent vao he so clean that, voi trong so them cho bang chi tiet.

    Day la so hang duy nhat trong phase 1 khong tu quy chieu: JEPA so bieu dien
    hoc duoc voi teacher EMA cua chinh no, con variance/covariance chi la thong
    ke cua bieu dien. Khong co so hang nay, ca he co the troi ve mot bieu dien
    tho ma loss van giam.

    QWT xep he so thanh [mau, bang, thanh phan] roi dep thanh 48 kenh; bang 0 la
    LL con bang 1..3 la LH/HL/HH, tuc duong net. Haar cho IMU xep approx truoc,
    detail sau. Ca hai deu duoc can them o phan chi tiet.
    """
    if image_predicted.shape != image_target.shape:
        raise ValueError(f"Image coefficients {tuple(image_predicted.shape)} != {tuple(image_target.shape)}")
    if imu_predicted.shape != imu_target.shape:
        raise ValueError(f"IMU coefficients {tuple(imu_predicted.shape)} != {tuple(imu_target.shape)}")
    image = F.l1_loss(image_predicted, image_target)
    image_detail = detail_band_l1(image_predicted, image_target, dim=2)
    imu = F.l1_loss(imu_predicted, imu_target)
    imu_detail = detail_band_l1(imu_predicted, imu_target, dim=1)
    total = (image + detail_weight * image_detail) + 0.5 * (imu + detail_weight * imu_detail)
    return total, {
        "reconstruction_image": image,
        "reconstruction_image_detail": image_detail,
        "reconstruction_imu": imu,
        "reconstruction_imu_detail": imu_detail,
    }
