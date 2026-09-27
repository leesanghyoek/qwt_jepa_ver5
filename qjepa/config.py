"""Configuration loading, validation, and object factories."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .corruptions import (
    IMAGE_MODES,
    IMU_MODES,
    ImuCorruptionConfig,
    LowLightImageCorruptionConfig,
    LowLightImageCorruptor,
    TrajectoryImuCorruptor,
)
from .data.normalize import ImuNormalizer
from .transforms import QWT_BACKENDS
from .models import LatentDecoders, LatentPretrainingModel, MultimodalBackbone
from .models.predictors import PREDICTOR_TYPES


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML config must be a mapping")
    parent = raw.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        config = _merge(load_config(parent_path), raw)
    else:
        config = raw
    config["_config_path"] = str(path)
    validate_config(config)
    return config


SPLIT_INTEGER_KEYS = ("split_color_width", "split_color_blocks", "split_edge_width",
                      "split_edge_blocks", "split_color_scale", "split_illumination_scale")
SPLIT_WEIGHT_KEYS = ("split_color_weight", "split_edge_weight", "split_gradient_weight")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("pipeline_version") != 3:
        raise ValueError("pipeline_version must be exactly 3")
    run_kind = config.get("run_kind", "main")
    if run_kind not in {"main", "smoke"}:
        raise ValueError("run_kind must be main or smoke")
    data = config["data"]
    phase1 = config["phase1"]
    phase2 = config["phase2"]
    model = config["model"]
    if str(config["runtime"].get("gpu_count", "auto")) not in {"auto", "1", "2"}:
        raise ValueError("runtime.gpu_count must be auto, 1 or 2")
    restart_limit = config["runtime"].get("restart_above_rss_gib")
    if restart_limit is not None and (isinstance(restart_limit, bool)
                                      or not isinstance(restart_limit, (int, float)) or restart_limit <= 0):
        raise ValueError("runtime.restart_above_rss_gib must be a positive number of GiB")
    if model.get("image_transform") not in QWT_BACKENDS or model.get("imu_transform") != "haar1d":
        raise ValueError(
            f"model.image_transform must be one of {sorted(QWT_BACKENDS)} and imu_transform haar1d"
        )
    if model.get("time_metadata_dim") != 3 or model.get("imu_summary_bins") != 4:
        raise ValueError("v3 fusion requires three time metadata values and four IMU summary bins")
    if run_kind == "main":
        if data.get("image_size") != [256, 256] or data.get("imu_window") != 128:
            raise ValueError("Main v3 requires one RGB 256x256 frame and an IMU window of 128")
        if phase1.get("batch_size", 0) < max(8, phase1.get("minimum_statistics_batch", 8)):
            raise ValueError("Phase-1 physical batch must meet minimum_statistics_batch")
        if config["monitor"].get("validation_bank_size", 64) < 64:
            raise ValueError("Main validation_bank_size must be at least 64 (or all available samples)")
    if config["monitor"].get("validation_bank_size", 64) < 2:
        raise ValueError("Validation bank needs at least two samples")
    for phase in (phase1, phase2):
        for key in ("batch_size", "gradient_accumulation", "max_successful_updates"):
            if not isinstance(phase.get(key), int) or phase[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
    for key in ("validation_batches", "log_every_updates", "checkpoint_every_updates"):
        if config["runtime"].get(key, 0) < 1:
            raise ValueError(f"runtime.{key} must be positive")
    # Phase 1 khong co duong khoi phuc trong khong gian pixel; chi he so.
    if phase1.get("reconstruction_loss_weight", 0.0) != 0.0:
        raise ValueError("phase1.reconstruction_loss_weight must be 0.0; use the coefficient weight")
    coefficient_weight = phase1.get("coefficient_reconstruction_loss_weight", 0.0)
    if bool(phase1.get("decoder_enabled", False)):
        if coefficient_weight <= 0:
            raise ValueError(
                "phase1.decoder_enabled needs coefficient_reconstruction_loss_weight > 0,"
                " otherwise the decoder trains without steering the latent"
            )
        if not 0.0 <= phase1.get("reconstruction_detail_weight", 0.0):
            raise ValueError("phase1.reconstruction_detail_weight cannot be negative")
    elif coefficient_weight != 0.0:
        raise ValueError("phase1.coefficient_reconstruction_loss_weight needs decoder_enabled")
    if phase1.get("gradient_accumulation", 1) != 1:
        raise ValueError("Phase 1 uses real batch statistics; gradient_accumulation must be 1")
    if not phase1.get("online_clean_forward_for_regularization", False):
        raise ValueError("Phase 1 requires the gradient-enabled clean online branch")
    if phase1.get("covariance_pooling", "per_position") not in ("per_position", "pooled"):
        raise ValueError("phase1.covariance_pooling must be per_position or pooled")
    _validate_phase1_predictor(phase1)
    if phase1.get("variance_weight", 0) <= 0 or phase1.get("covariance_weight", 0) <= 0:
        raise ValueError("Main latent training requires explicit variance and covariance losses")
    if phase1.get("jepa_weight") != 1.0 or phase1.get("precision") != "fp32":
        raise ValueError("Supported phase-1 recipe requires jepa_weight=1 and FP32")
    required_maps = {"FI", "FU", "ZI", "ZU", "FI_clean", "FU_clean", "ZI_clean", "ZU_clean"}
    if set(phase1.get("regularized_maps", ())) != required_maps:
        raise ValueError("phase1.regularized_maps must contain all eight raw feature maps")
    required_phase2 = {"freeze_backbone": True}
    for key, required in required_phase2.items():
        if phase2.get(key) != required:
            raise ValueError(f"phase2.{key} must be {required!r}")
    # decoder_input phai noi that ve viec decoder nhan gi. Bat skip ma van khai
    # "chi latent" la dung loai noi doi ma cac kiem tra o day sinh ra de chan.
    skips = bool(phase2.get("encoder_skips", False))
    expected_input = "latent_plus_encoder_skips" if skips else "fused_dense_latent_only"
    if phase2.get("decoder_input") != expected_input:
        raise ValueError(
            f"phase2.decoder_input must be {expected_input!r} when encoder_skips is {skips}"
        )
    # Phai khai TUONG MINH kieu merge. Neu de mac dinh, mot config cu (khong co
    # khoa nay) van hash giong het mot checkpoint cu, roi lang le dung kien truc
    # moi — va loi chi lo ra o load_state_dict, sau khi da dung sai model.
    if skips and "skip_gating" not in phase2:
        raise ValueError(
            "phase2.encoder_skips needs an explicit phase2.skip_gating so the"
            " configuration hash records which merge the checkpoint was trained with"
        )
    # Hai khoa nay phai noi cung mot chuyen, neu khong config se noi doi ve
    # viec decoder that su lam gi.
    residual = bool(phase2.get("input_coefficient_residual", False))
    expected_output = "input_residual" if residual else "absolute_prediction"
    if phase2.get("output_coefficients") != expected_output:
        raise ValueError(
            f"phase2.output_coefficients must be {expected_output!r} when"
            f" input_coefficient_residual is {residual}"
        )
    # Cung ly do voi skip_gating: de mac dinh thi mot config cu hash giong het mot
    # checkpoint cu roi lang le dung kien truc khac.
    if residual and "residual_sees_input" not in phase2:
        raise ValueError(
            "phase2.input_coefficient_residual needs an explicit"
            " phase2.residual_sees_input so the hash records which head was trained"
        )
    if phase2.get("reconstruction_detail_weight", 0.0) < 0:
        raise ValueError("phase2.reconstruction_detail_weight cannot be negative")
    if phase2.get("imu_variation_weight", 0.0) < 0:
        raise ValueError("phase2.imu_variation_weight cannot be negative")
    if phase2.get("detail_energy_weight", 0.0) < 0:
        raise ValueError("phase2.detail_energy_weight cannot be negative")
    if phase2.get("image_detail_loss", "coefficient") not in ("coefficient", "modulus"):
        raise ValueError("phase2.image_detail_loss must be coefficient or modulus")
    if phase2.get("image_detail_source", "decoder_coefficients") not in ("decoder_coefficients", "restored_image"):
        raise ValueError("phase2.image_detail_source must be decoder_coefficients or restored_image")
    image_decoder = phase2.get("image_decoder", "qwt_coefficients")
    if image_decoder not in ("qwt_coefficients", "resnet_pixel", "split_color_edge"):
        raise ValueError("phase2.image_decoder must be qwt_coefficients, resnet_pixel or split_color_edge")
    if image_decoder == "split_color_edge":
        # Explicit, like skip_gating: the hash must record what was trained.
        for key in SPLIT_INTEGER_KEYS:
            value = phase2.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"phase2.image_decoder split_color_edge needs a positive integer phase2.{key}")
        for key in SPLIT_WEIGHT_KEYS:
            value = phase2.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"phase2.image_decoder split_color_edge needs a nonnegative phase2.{key}")
        for side in data["image_size"]:
            if side % phase2["split_color_scale"] or side % phase2["split_illumination_scale"]:
                raise ValueError("phase2 split scales must divide the image sides")
        arch = phase2.get("split_branch_arch", "resnet")
        if arch not in ("resnet", "unet", "unet_edge", "nafnet_edge"):
            raise ValueError("phase2.split_branch_arch must be resnet, unet, unet_edge or nafnet_edge")
        # The aux settings only mean something to the NAFNet branch, the only one with
        # decoder heads; other branches produce no aux outputs, so the term is absent.
        if arch == "nafnet_edge":
            _validate_nafnet_edge(phase2, data["image_size"])
        fft_weight = phase2.get("split_edge_fft_weight", 0.0)
        if isinstance(fft_weight, bool) or not isinstance(fft_weight, (int, float)) or fft_weight < 0:
            raise ValueError("phase2.split_edge_fft_weight must be a nonnegative number")
        if arch in ("unet", "unet_edge"):
            unet_keys = (("split_color_unet_widths", phase2["split_color_scale"]),
                         ("split_edge_unet_widths", 1))
            for key, scale in unet_keys if arch == "unet" else unet_keys[1:]:
                widths = phase2.get(key)
                if (not isinstance(widths, (list, tuple)) or len(widths) < 2
                        or any(isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in widths)):
                    raise ValueError(f"phase2.{key} must list at least two positive channel counts")
                step = scale * 2 ** (len(widths) - 1)
                if any(side % step for side in data["image_size"]):
                    raise ValueError(f"phase2.{key}: image sides must divide by {step}")
            blocks = phase2.get("split_unet_blocks")
            if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
                raise ValueError("phase2.split_unet_blocks must be a positive integer")
        # Absent in p8-era configs: off, so their checkpoints and hashes stay as they were.
        color_global = phase2.get("split_color_global", False)
        if not isinstance(color_global, bool):
            raise ValueError("phase2.split_color_global must be true or false")
        if color_global and arch == "unet":
            raise ValueError("phase2.split_color_global needs the ResNet colour branch (resnet or unet_edge)")
        stats = phase2.get("split_color_stats_weight", 0.0)
        if isinstance(stats, bool) or not isinstance(stats, (int, float)) or stats < 0:
            raise ValueError("phase2.split_color_stats_weight must be a nonnegative number")
    if image_decoder == "resnet_pixel":
        # Explicit, like skip_gating: the hash must record the trained width/depth.
        for key in ("image_resnet_width", "image_resnet_blocks"):
            value = phase2.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"phase2.image_decoder resnet_pixel needs a positive integer phase2.{key}")
        if data["image_size"][0] % 2 or data["image_size"][1] % 2:
            raise ValueError("phase2.image_decoder resnet_pixel needs even image sides")
    perceptual = phase2.get("perceptual_weight", 0.0)
    if isinstance(perceptual, bool) or not isinstance(perceptual, (int, float)) or perceptual < 0:
        raise ValueError("phase2.perceptual_weight must be a nonnegative number")
    if phase2.get("smooth_l1_beta", 0.0) <= 0:
        raise ValueError("phase2.smooth_l1_beta must be positive")
    scenarios = phase2.get("train_scenarios")
    if scenarios is not None:
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("phase2.train_scenarios must be a nonempty list")
        for scenario in scenarios:
            if not isinstance(scenario, dict) or set(scenario) != {"image_mode", "imu_mode", "weight"}:
                raise ValueError("Each phase2.train_scenarios entry needs image_mode, imu_mode and weight")
            if scenario["image_mode"] not in IMAGE_MODES or scenario["imu_mode"] not in IMU_MODES:
                raise ValueError("Invalid phase2.train_scenarios corruption mode")
            weight = scenario["weight"]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
                raise ValueError("phase2.train_scenarios weights must be finite and positive")
    if "blur_validation_samples" in phase2 and (not isinstance(phase2["blur_validation_samples"], int)
            or phase2["blur_validation_samples"] < 1):
        raise ValueError("phase2.blur_validation_samples must be a positive integer")
    if "full_guard" in phase2:
        guard = phase2["full_guard"]
        required = {"max_psnr_drop_db", "max_ssim_drop", "max_accel_rmse_ratio", "max_gyro_rmse_ratio"}
        if not isinstance(guard, dict) or set(guard) != required:
            raise ValueError(f"phase2.full_guard needs exactly {sorted(required)}")
        for key, value in guard.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"phase2.full_guard.{key} must be finite")
            if value < (1.0 if key.endswith("ratio") else 0.0):
                raise ValueError(f"phase2.full_guard.{key} must be nonnegative or at least 1")
        if "blur_validation_samples" not in phase2:
            raise ValueError("phase2.full_guard requires blur_validation_samples")
    if phase2.get("jepa_loss_weight") != 0.0 or phase2.get("sensitivity_loss_weight") != 0.0:
        raise ValueError("Phase 2 cannot optimize latent/Jacobian losses")
    if phase2.get("reconstruction_loss_weight") != 1.0 or phase2.get("precision") != "fp32":
        raise ValueError("Supported phase-2 recipe requires reconstruction weight 1 and FP32")
    if data.get("split_unit") != "trajectory":
        raise ValueError("Data split unit must be trajectory")
    if data.get("minimum_trajectories_per_batch", 0) < 1:
        raise ValueError("minimum_trajectories_per_batch must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_normalizer(metadata: dict[str, Any]) -> ImuNormalizer:
    normalization = metadata["normalization"]
    return ImuNormalizer(normalization["mean"], normalization["std"])


def build_backbone(config: dict[str, Any]) -> MultimodalBackbone:
    model = config["model"]
    channels = tuple(model["encoder_channels"])
    return MultimodalBackbone(
        channels=channels,
        embedding_dim=model["embedding_dim"],
        fusion_hidden=model["fusion_hidden_dim"],
        imu_summary_bins=model["imu_summary_bins"],
        time_metadata_dim=model["time_metadata_dim"],
        gate_bias=model["gate_bias_init"],
        groups=model["groupnorm_groups"],
        image_transform=model["image_transform"],
    )


def _positive_ints(value: Any, length: int | None = None) -> bool:
    return (isinstance(value, (list, tuple)) and (length is None or len(value) == length)
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in value))


def _validate_nafnet_edge(phase2: dict[str, Any], image_size: list[int]) -> None:
    """Widths, block counts and auxiliary factors of the NAFNet edge branch."""
    widths = phase2.get("split_edge_naf_widths")
    if not _positive_ints(widths) or len(widths) < 2:
        raise ValueError("phase2.split_edge_naf_widths must list at least two positive channel counts")
    if any(side % 2 ** (len(widths) - 1) for side in image_size):
        raise ValueError(f"phase2.split_edge_naf_widths: image sides must divide by {2 ** (len(widths) - 1)}")
    for key in ("split_edge_naf_enc_blocks", "split_edge_naf_dec_blocks"):
        if not _positive_ints(phase2.get(key), len(widths) - 1):
            raise ValueError(f"phase2.{key} must list len(split_edge_naf_widths) - 1 positive block counts")
    middle = phase2.get("split_edge_naf_middle_blocks")
    if isinstance(middle, bool) or not isinstance(middle, int) or middle < 1:
        raise ValueError("phase2.split_edge_naf_middle_blocks must be a positive integer")
    factors = phase2.get("split_edge_aux_factors", [])
    if not isinstance(factors, (list, tuple)) or any(
            not isinstance(f, int) or f < 2 or f & (f - 1) or f > 2 ** (len(widths) - 2) for f in factors):
        raise ValueError("phase2.split_edge_aux_factors must be powers of two that have a decoder level")
    weight = phase2.get("split_edge_aux_weight", 0.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError("phase2.split_edge_aux_weight must be a nonnegative number")
    if weight > 0 and not factors:
        raise ValueError("phase2.split_edge_aux_weight > 0 needs split_edge_aux_factors")


def _validate_phase1_predictor(phase1: dict[str, Any]) -> None:
    """Predictor shape, token masking and the multi-scale JEPA terms."""
    predictor = phase1.get("predictor_type", "token")
    if predictor not in PREDICTOR_TYPES:
        raise ValueError(f"phase1.predictor_type must be one of {PREDICTOR_TYPES}")
    kernel = phase1.get("predictor_kernel", 3)
    layers = phase1.get("predictor_mixing_layers", 2)
    if not isinstance(kernel, int) or kernel < 1 or kernel % 2 == 0:
        raise ValueError("phase1.predictor_kernel must be an odd positive integer")
    if not isinstance(layers, int) or layers < 1:
        raise ValueError("phase1.predictor_mixing_layers must be a positive integer")
    for key, size_key in (("image_mask_ratio", "image_mask_block"), ("imu_mask_ratio", "imu_mask_span")):
        ratio = phase1.get(key, 0.0)
        if not 0.0 <= ratio <= 0.75:
            raise ValueError(f"phase1.{key} must be in [0, 0.75]")
        if ratio > 0:
            # A token-wise predictor sees nothing of a hidden token: it could only
            # learn the mean target there.
            if predictor != "spatial":
                raise ValueError(f"phase1.{key} > 0 needs phase1.predictor_type: spatial")
            size = phase1.get(size_key)
            if (not isinstance(size, (list, tuple)) or len(size) != 2
                    or not all(isinstance(v, int) and v >= 1 for v in size) or size[0] > size[1]):
                raise ValueError(f"phase1.{size_key} must be [min, max] token counts, 1 <= min <= max")
    fine = phase1.get("multiscale_fine_weight", 0.0)
    coarse = phase1.get("multiscale_coarse_weight", 0.0)
    if fine < 0 or coarse < 0:
        raise ValueError("phase1 multi-scale JEPA weights cannot be negative")
    if fine > 0 and predictor != "spatial":
        raise ValueError("phase1.multiscale_fine_weight > 0 needs phase1.predictor_type: spatial")
    if coarse > 0:
        pool = phase1.get("multiscale_coarse_pool")
        if not isinstance(pool, int) or pool < 2:
            raise ValueError("phase1.multiscale_coarse_pool must be an integer >= 2")


def build_phase1_model(config: dict[str, Any], normalizer: ImuNormalizer) -> LatentPretrainingModel:
    enabled = bool(config["phase1"].get("decoder_enabled", False))
    phase1 = config["phase1"]
    return LatentPretrainingModel(
        backbone=build_backbone(config),
        normalizer=normalizer,
        predictor_hidden=config["model"]["predictor_hidden_dim"],
        # The predictor never leaves phase 1, so its shape lives in phase1: the
        # phase-2 contract does not move when it changes.
        predictor_type=phase1.get("predictor_type", "token"),
        predictor_kernel=int(phase1.get("predictor_kernel", 3)),
        predictor_layers=int(phase1.get("predictor_mixing_layers", 2)),
        fine_scale=float(phase1.get("multiscale_fine_weight", 0.0)) > 0,
        # Neo phase 1 khong bao gio nhan skip: neu no co duong vong tu encoder thi
        # no thoa man duoc neo ma khong ep gi vao latent — dung cai ma neo sinh ra
        # de ngan. Cung ly do voi viec no giu he so tuyet doi thay vi residual.
        # The phase-1 anchor predicts clean COEFFICIENTS from the latent alone;
        # phase2.image_decoder must never reach it, or the phase-1 model changes.
        decoders=build_decoders(config, residual=False, skips=False,
                                image_decoder="qwt_coefficients") if enabled else None,
    )


def build_decoders(
    config: dict[str, Any], *, residual: bool | None = None, skips: bool | None = None,
    image_decoder: str | None = None,
) -> LatentDecoders:
    image_size = config["data"]["image_size"]
    channels = tuple(config["model"]["encoder_channels"])
    if image_decoder is None:
        # Absent from checkpoints trained before the key existed: those used the
        # coefficient decoder, and that is what their weights rebuild into.
        image_decoder = config["phase2"].get("image_decoder", "qwt_coefficients")
    if residual is None:
        residual = bool(config["phase2"].get("input_coefficient_residual", False))
    if skips is None:
        skips = bool(config["phase2"].get("encoder_skips", False))
    return LatentDecoders(
        image_coefficient_size=(image_size[0] // 2, image_size[1] // 2),
        imu_coefficient_length=config["data"]["imu_window"] // 2,
        channels=channels,
        groups=config["model"]["groupnorm_groups"],
        residual=residual,
        # Thu tu tu tho den min, khop voi thu tu encoder tra ve.
        skip_channels=(channels[2], channels[1], channels[0]) if skips else None,
        # .get chu khong phai [...]: ham nay cung doc config NAM TRONG checkpoint,
        # va checkpoint train truoc khi khoa ra doi thi khong co no. "Khong co"
        # nghia la kien truc truoc do, tuc False — dung gia tri tai tao lai dung
        # mang da train. validate_config van bat khai tuong minh cho config MOI,
        # nen khong co duong nao doi kien truc am tham.
        skip_gating=bool(config["phase2"].get("skip_gating", False)) if skips else True,
        sees_input=bool(config["phase2"].get("residual_sees_input", False)) if residual else False,
        image_decoder=image_decoder,
        resnet_width=int(config["phase2"].get("image_resnet_width", 64)),
        resnet_blocks=int(config["phase2"].get("image_resnet_blocks", 8)),
        split={
            "color_width": int(config["phase2"].get("split_color_width", 32)),
            "color_blocks": int(config["phase2"].get("split_color_blocks", 6)),
            "edge_width": int(config["phase2"].get("split_edge_width", 64)),
            "edge_blocks": int(config["phase2"].get("split_edge_blocks", 6)),
            "color_scale": int(config["phase2"].get("split_color_scale", 2)),
            "illumination_scale": int(config["phase2"].get("split_illumination_scale", 8)),
            # Absent from p8 configs, which used the single-level ResNet branches.
            "branch_arch": str(config["phase2"].get("split_branch_arch", "resnet")),
            "color_widths": tuple(config["phase2"].get("split_color_unet_widths", (12, 16, 24, 32))),
            "edge_widths": tuple(config["phase2"].get("split_edge_unet_widths", (16, 24, 32, 48, 56))),
            "unet_blocks": int(config["phase2"].get("split_unet_blocks", 1)),
            "color_global": bool(config["phase2"].get("split_color_global", False)),
            "naf": {
                "widths": tuple(config["phase2"].get("split_edge_naf_widths", ())),
                "enc_blocks": tuple(config["phase2"].get("split_edge_naf_enc_blocks", ())),
                "middle_blocks": int(config["phase2"].get("split_edge_naf_middle_blocks", 0)),
                "dec_blocks": tuple(config["phase2"].get("split_edge_naf_dec_blocks", ())),
                "aux_factors": tuple(config["phase2"].get("split_edge_aux_factors", ())),
            } if config["phase2"].get("split_branch_arch") == "nafnet_edge" else None,
        },
    )


def build_corruptors(config: dict[str, Any]):
    image_values = config["corruption"]["image"]
    imu_values = config["corruption"]["imu"]
    image_cfg = LowLightImageCorruptionConfig(**image_values)
    imu_cfg = ImuCorruptionConfig(**imu_values)
    seed = config["data"]["corruption_seed"]
    return LowLightImageCorruptor(image_cfg, seed), TrajectoryImuCorruptor(imu_cfg, seed)


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
