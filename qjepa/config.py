"""Configuration loading, validation, and object factories."""

from __future__ import annotations

import copy
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from .corruptions import (
    IMAGE_MODES,
    IMU_MODES,
    ImuCorruptionConfig,
    LowLightImageCorruptionConfig,
    LowLightImageCorruptor,
    TrajectoryImuCorruptor,
)
from .corruptions.image import DEGRADATION_FEATURES
from .data.normalize import ImuNormalizer
from .transforms import IMAGE_INPUTS, QWT_BACKENDS
from .models import LatentDecoders, LatentPretrainingModel, MultimodalBackbone
from .models.backbone import ENCODER_TYPES
from .models.blocks import ENCODER_NORMS
from .models.ijepa import CONTEXT_INPUTS, IJEPAPretrainingModel
from .models.vit import INPUT_STANDARDIZATIONS, TOKEN_STRIDE
from .models.decoders import PIXEL_IMAGE_DECODERS
from .models.predictors import PREDICTOR_TYPES
from .training.phase1 import NOISE_DIRECTIONS


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
    for switch, default in (("cudnn_benchmark", False), ("cudnn_enabled", True)):
        if not isinstance(config["runtime"].get(switch, default), bool):
            raise ValueError(f"runtime.{switch} must be true or false")
    parallel = config["runtime"].get("parallel", "data_parallel")
    if parallel not in {"data_parallel", "ddp"}:
        raise ValueError("runtime.parallel must be data_parallel or ddp")
    if parallel == "ddp" and str(config["runtime"].get("gpu_count", "auto")) != "1":
        for name, section in (("phase1", phase1), ("phase2", phase2)):
            if section["batch_size"] % 2:
                raise ValueError(f"runtime.parallel ddp splits {name}.batch_size over 2 GPUs; it must be even")
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
    # Absent before the I-JEPA arm: the VICReg JEPA every checkpoint so far was trained with.
    ijepa = phase1.get("objective", "jepa") == "ijepa"
    if not ijepa and not phase1.get("online_clean_forward_for_regularization", False):
        raise ValueError("Phase 1 requires the gradient-enabled clean online branch")
    if phase1.get("covariance_pooling", "per_position") not in ("per_position", "pooled"):
        raise ValueError("phase1.covariance_pooling must be per_position or pooled")
    _validate_phase1_predictor(phase1)
    _validate_ijepa(config)
    _validate_sharpness(config)
    if phase2.get("backbone_weights", "context") not in BACKBONE_WEIGHTS:
        raise ValueError(f"phase2.backbone_weights must be one of {BACKBONE_WEIGHTS}")
    _validate_light(config)
    _validate_light_branch(config)
    _validate_source_size(config)
    floors = config.get("encoder_sensitivity", {}).get("signal_floor_log_gain")
    if floors is not None and (not isinstance(floors, dict) or not set(floors) <= {"image", "imu"} or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in floors.values())):
        raise ValueError("encoder_sensitivity.signal_floor_log_gain must map image/imu to numbers")
    if not ijepa and (phase1.get("variance_weight", 0) <= 0 or phase1.get("covariance_weight", 0) <= 0):
        raise ValueError("Main latent training requires explicit variance and covariance losses")
    if phase1.get("jepa_weight") != 1.0 or phase1.get("precision") != "fp32":
        raise ValueError("Supported phase-1 recipe requires jepa_weight=1 and FP32")
    required_maps = {"FI", "FU", "ZI", "ZU", "FI_clean", "FU_clean", "ZI_clean", "ZU_clean"}
    if not ijepa and set(phase1.get("regularized_maps", ())) != required_maps:
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
    # Absent from configs before the IMU refiner: 0 blocks, no refiner, no jitter term.
    for key, default, least in (("imu_refiner_blocks", 0, 0), ("imu_refiner_width", 32, 1)):
        value = phase2.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise ValueError(f"phase2.{key} must be an integer >= {least}")
    jitter_weight = phase2.get("imu_jitter_weight", 0.0)
    if isinstance(jitter_weight, bool) or not isinstance(jitter_weight, (int, float)) or jitter_weight < 0:
        raise ValueError("phase2.imu_jitter_weight must be a nonnegative number")
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
        for key in ("split_edge_refiner_blocks", "split_edge_refiner_width"):
            value = phase2.get(key, 0 if key.endswith("blocks") else 32)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key.endswith("blocks") else 1):
                raise ValueError(f"phase2.{key} must be a {'nonnegative' if key.endswith('blocks') else 'positive'} integer")
        for key in ("split_edge_stage1_weight", "split_edge_smooth_weight"):
            value = phase2.get(key, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"phase2.{key} must be a nonnegative number")
        if phase2.get("split_edge_stage1_weight", 0.0) and not phase2.get("split_edge_refiner_blocks", 0):
            raise ValueError("phase2.split_edge_stage1_weight needs an edge refiner (split_edge_refiner_blocks > 0)")
        # Absent in p15/p16 configs: 1, the refiner at the input resolution.
        refiner_scale = phase2.get("split_edge_refiner_scale", 1)
        if isinstance(refiner_scale, bool) or not isinstance(refiner_scale, int) or refiner_scale < 1:
            raise ValueError("phase2.split_edge_refiner_scale must be a positive integer")
        if refiner_scale > 1 and not phase2.get("split_edge_refiner_blocks", 0):
            raise ValueError("phase2.split_edge_refiner_scale needs an edge refiner (split_edge_refiner_blocks > 0)")
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
    if phase2.get("reconstruction_loss_weight") != 1.0 or phase2.get("precision") not in ("fp32", "amp_fp16"):
        raise ValueError("Supported phase-2 recipe requires reconstruction weight 1 and precision fp32 or amp_fp16")
    crop = phase2.get("perceptual_crop", 0)
    if isinstance(crop, bool) or not isinstance(crop, int) or crop < 0 or crop % 4:
        raise ValueError("phase2.perceptual_crop must be 0 (whole frame) or a positive multiple of 4")
    # Checked only when the term is on: smoke runs keep the recipe's crop with weight 0.
    if crop and phase2.get("perceptual_weight", 0.0) > 0 and any(crop > side for side in data["image_size"]):
        raise ValueError("phase2.perceptual_crop cannot exceed the image size")
    if data.get("split_unit") != "trajectory":
        raise ValueError("Data split unit must be trajectory")
    if data.get("split_rule", "hash") not in ("hash", "per_environment"):
        raise ValueError("data.split_rule must be hash (a draw over all trajectories) or per_environment")
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
        # Absent before the sharpness plan: GroupNorm, the encoders every checkpoint holds.
        encoder_norm=model.get("encoder_norm", "group"),
        # Absent before the luminance arm: RGB, the 48-channel QWT every checkpoint holds.
        image_input=model.get("image_input", "rgb"),
        # Absent before the I-JEPA arm: the CNN encoders and the gated fusion.
        encoder_type=model.get("encoder_type", "cnn"),
        vit_depth=int(model.get("vit_depth", 6)),
        vit_heads=int(model.get("vit_heads", 4)),
        # Absent before p29: the coefficients reach the ViT unscaled.
        vit_input_standardize=model.get("vit_input_standardize"),
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
    fourier = phase2.get("split_edge_naf_fourier_levels", [])
    if (not isinstance(fourier, (list, tuple)) or len(set(fourier)) != len(fourier)
            or any(isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= len(widths) - 2
                   for level in fourier)):
        raise ValueError("phase2.split_edge_naf_fourier_levels must list distinct NAFNet levels "
                         f"in 0..{len(widths) - 2}")
    stage_levels = phase2.get("split_edge_naf_stage_levels", [])
    if (not isinstance(stage_levels, (list, tuple)) or len(set(stage_levels)) != len(stage_levels)
            or any(isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= min(3, len(widths) - 2)
                   for level in stage_levels)):
        raise ValueError("phase2.split_edge_naf_stage_levels must list distinct levels in 1..3 "
                         "(the image encoder's stages at 1/2, 1/4, 1/8 of the frame)")
    if stage_levels and not phase2.get("encoder_skips", False):
        raise ValueError("phase2.split_edge_naf_stage_levels needs phase2.encoder_skips: true "
                         "(the encoder stages are computed only then)")
    fourier_width = phase2.get("split_edge_naf_fourier_width", 16)
    if isinstance(fourier_width, bool) or not isinstance(fourier_width, int) or fourier_width < 1:
        raise ValueError("phase2.split_edge_naf_fourier_width must be a positive integer")


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
    # Absent before p16_infomax: 0, none of the four information terms.
    finer = phase1.get("multiscale_finer_weight", 0.0)
    if isinstance(finer, bool) or not isinstance(finer, (int, float)) or finer < 0:
        raise ValueError("phase1.multiscale_finer_weight must be a nonnegative number")
    if finer > 0 and not fine > 0:
        raise ValueError("phase1.multiscale_finer_weight > 0 needs phase1.multiscale_fine_weight > 0")
    for key in ("coding_rate_weight", "infonce_weight"):
        value = phase1.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"phase1.{key} must be a nonnegative number")
    for key, default in (("coding_rate_eps_squared", 0.5), ("infonce_temperature", 0.1)):
        value = phase1.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"phase1.{key} must be a positive number")
    if coarse > 0:
        pool = phase1.get("multiscale_coarse_pool")
        if not isinstance(pool, int) or pool < 2:
            raise ValueError("phase1.multiscale_coarse_pool must be an integer >= 2")


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _interval(value: Any, low: float, high: float) -> bool:
    """[a, b] with low <= a <= b <= high."""
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
            and low <= value[0] <= value[1] <= high)


# The VICReg JEPA's terms. I-JEPA trains with none of them, and a config that names
# one of them nonzero would say the run trained something it did not.
IJEPA_ABSENT_TERMS = ("coefficient_reconstruction_loss_weight", "variance_weight", "covariance_weight",
                      "coding_rate_weight", "infonce_weight", "multiscale_fine_weight",
                      "multiscale_coarse_weight", "multiscale_finer_weight", "degradation_weight",
                      "image_mask_ratio", "imu_mask_ratio")


def _validate_ijepa(config: dict[str, Any]) -> None:
    """model.encoder_type vit and phase1.objective ijepa come together. Every I-JEPA
    setting is explicit, so the hash records the recipe that was trained."""
    model, phase1, phase2 = config["model"], config["phase1"], config["phase2"]
    encoder_type = model.get("encoder_type", "cnn")
    objective = phase1.get("objective", "jepa")
    if encoder_type not in ENCODER_TYPES:
        raise ValueError(f"model.encoder_type must be one of {ENCODER_TYPES}")
    if objective not in ("jepa", "ijepa"):
        raise ValueError("phase1.objective must be jepa or ijepa")
    if (encoder_type == "vit") != (objective == "ijepa"):
        raise ValueError("phase1.objective ijepa and model.encoder_type vit come together: I-JEPA's context "
                         "encoder sees only the context tokens, which a CNN cannot do")
    standardize = model.get("vit_input_standardize")
    if standardize is not None and standardize not in INPUT_STANDARDIZATIONS:
        raise ValueError(f"model.vit_input_standardize must be one of {INPUT_STANDARDIZATIONS} (absent: off)")
    if standardize is not None and encoder_type != "vit":
        raise ValueError("model.vit_input_standardize scales the ViT's input coefficients; set model.encoder_type: vit")
    if encoder_type != "vit":
        return
    for key in ("vit_depth", "vit_heads"):
        if not _positive_int(model.get(key)):
            raise ValueError(f"model.encoder_type vit needs a positive integer model.{key}")
    if model["embedding_dim"] % model["vit_heads"] or model["embedding_dim"] % 4:
        raise ValueError("model.embedding_dim must divide by model.vit_heads and by 4 (2-D positions)")
    if model.get("encoder_norm", "group") != "group" or model.get("encoder_norm_calibration", False):
        raise ValueError("model.encoder_norm centre and its calibration belong to the CNN; the ViT has LayerNorm")
    sides = [*config["data"]["image_size"], config["data"]["imu_window"]]
    if any(side % TOKEN_STRIDE for side in sides):
        raise ValueError(f"model.encoder_type vit needs image sides and data.imu_window divisible by {TOKEN_STRIDE}")
    if phase2.get("encoder_skips", False) or phase2.get("decoder_predictor_input", False):
        raise ValueError("The ViT has one resolution and I-JEPA's predictor needs masks: "
                         "phase2.encoder_skips and phase2.decoder_predictor_input must be false")
    for key in ("ijepa_targets", "ijepa_image_min_keep", "ijepa_imu_min_keep", "ijepa_predictor_dim",
                "ijepa_predictor_depth", "ijepa_predictor_heads"):
        if not _positive_int(phase1.get(key)):
            raise ValueError(f"phase1.objective ijepa needs a positive integer phase1.{key}")
    if phase1["ijepa_predictor_dim"] % phase1["ijepa_predictor_heads"] or phase1["ijepa_predictor_dim"] % 4:
        raise ValueError("phase1.ijepa_predictor_dim must divide by ijepa_predictor_heads and by 4")
    for key, low, high in (("ijepa_target_scale", 0.0, 1.0), ("ijepa_context_scale", 0.0, 1.0),
                           ("ijepa_target_aspect", 1e-3, 1e3)):
        if not _interval(phase1.get(key), low, high) or phase1[key][0] <= 0:
            raise ValueError(f"phase1.{key} must be [min, max] with 0 < min <= max <= {high:g}")
    if phase1.get("ijepa_context_input") not in CONTEXT_INPUTS:
        raise ValueError(f"phase1.ijepa_context_input must be one of {CONTEXT_INPUTS}")
    start, end = phase1.get("teacher_momentum_start"), phase1.get("teacher_momentum_end")
    if not (_nonnegative_number(start) and _nonnegative_number(end) and start < 1.0 and start <= end <= 1.0):
        raise ValueError("phase1 teacher momentum must rise from start < 1 to end <= 1")
    if not _nonnegative_number(phase1.get("weight_decay_end")):
        raise ValueError("phase1.objective ijepa needs a nonnegative phase1.weight_decay_end")
    clip = phase1.get("gradient_clip_norm")
    if clip is not None and (not _nonnegative_number(clip) or clip <= 0):
        raise ValueError("phase1.gradient_clip_norm must be positive, or null for none (I-JEPA)")
    for key in IJEPA_ABSENT_TERMS:
        if phase1.get(key, 0.0) != 0.0:
            raise ValueError(f"phase1.objective ijepa trains I-JEPA's loss alone: phase1.{key} must be 0")
    for key in ("decoder_enabled", "predictor_degradation_condition"):
        if phase1.get(key, False):
            raise ValueError(f"phase1.objective ijepa trains I-JEPA's loss alone: phase1.{key} must be false")
    if config.get("encoder_sensitivity", {}).get("enabled", False):
        raise ValueError("phase1.objective ijepa trains I-JEPA's loss alone: encoder_sensitivity.enabled must be false")


def _nonnegative_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and np.isfinite(value) and value >= 0


LIGHT_KEYS = tuple(field.name for field in fields(LowLightImageCorruptionConfig) if field.name.startswith("light_"))
ILLUM_KEYS = tuple(field.name for field in fields(LowLightImageCorruptionConfig) if field.name.startswith("illum_"))
FOG_KEYS = tuple(field.name for field in fields(LowLightImageCorruptionConfig) if field.name.startswith("fog_"))
# Optional switches inside a group: absent means the old behaviour, so they need not be written out.
OPTIONAL_KEYS = ("illum_highlight_rolloff", "illum_max_brighten_stops")


def _validate_light(config: dict[str, Any]) -> None:
    """corruption.image.light_*: den va loe sang (qjepa/corruptions/light.py).

    Bat (light_probability > 0) thi PHAI ghi du moi khoa light_*: hash chi doc file config,
    nen gia tri mac dinh trong dataclass ma doi ve sau se khong lo ra trong hash."""
    image = config["corruption"]["image"]
    try:
        LowLightImageCorruptionConfig(**image).validate()
    except TypeError as error:
        raise ValueError(f"corruption.image: {error}") from error
    for switch, keys in (("light_probability", LIGHT_KEYS), ("illum_probability", ILLUM_KEYS),
                         ("fog_probability", FOG_KEYS)):
        if float(image.get(switch, 0.0)) > 0:
            missing = [key for key in keys if key not in image and key not in OPTIONAL_KEYS]
            if missing:
                raise ValueError(f"corruption.image.{switch} > 0 needs every {switch.split('_')[0]}_* key written "
                                 f"out (the hash records only what the config says); missing: {', '.join(missing)}")


def _validate_source_size(config: dict[str, Any]) -> None:
    """data.source_size: read frames at this size and train on image_size crops of them (absent: as before)."""
    source = config["data"].get("source_size")
    if source is None:
        return
    if (not isinstance(source, (list, tuple)) or len(source) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in source)):
        raise ValueError("data.source_size must be two integers [height, width]")
    if any(s < i for s, i in zip(source, config["data"]["image_size"])):
        raise ValueError("data.source_size must be at least data.image_size (the training crop)")
    if any(s % 16 for s in source):
        raise ValueError("data.source_size must divide by 16: the image encoder halves it four times")


SPLIT_LIGHT_KEYS = ("split_light_width", "split_light_scale", "split_light_levels", "split_light_weight",
                    "split_light_loss_scale")


def _validate_light_branch(config: dict[str, Any]) -> None:
    """phase2.split_light_*: LightBranch truoc nhanh mau va nhanh duong net (go loe, lam sang cho toi).

    Bat thi PHAI ghi du moi khoa split_light_* (hash chi doc file config)."""
    phase2 = config["phase2"]
    enabled = phase2.get("split_light_branch", False)
    if not isinstance(enabled, bool):
        raise ValueError("phase2.split_light_branch must be true or false")
    if not enabled:
        return
    if phase2.get("image_decoder") != "split_color_edge":
        raise ValueError("phase2.split_light_branch works in the split_color_edge decoder")
    missing = [key for key in SPLIT_LIGHT_KEYS if key not in phase2]
    if missing:
        raise ValueError(f"phase2.split_light_branch needs every split_light_* key written out; missing: "
                         f"{', '.join(missing)}")
    width, scale, levels = phase2["split_light_width"], phase2["split_light_scale"], phase2["split_light_levels"]
    loss_scale = phase2["split_light_loss_scale"]
    for name, value in (("split_light_width", width), ("split_light_scale", scale), ("split_light_levels", levels),
                        ("split_light_loss_scale", loss_scale)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"phase2.{name} must be a positive integer")
    if not _nonnegative_number(phase2["split_light_weight"]):
        raise ValueError("phase2.split_light_weight must be a nonnegative number")
    for side in config["data"]["image_size"]:
        if side % scale or side % loss_scale or side // scale < 2 ** levels:
            raise ValueError("data.image_size must divide by split_light_scale and split_light_loss_scale, "
                             "and leave at least one pixel after split_light_levels halvings")


def _validate_sharpness(config: dict[str, Any]) -> None:
    """The sharpness plan's keys. Every one is optional: absent, the run trains as before."""
    phase1, phase2 = config["phase1"], config["phase2"]
    if config["model"].get("encoder_norm", "group") not in ENCODER_NORMS:
        raise ValueError(f"model.encoder_norm must be one of {ENCODER_NORMS}")
    calibration = config["model"].get("encoder_norm_calibration", False)
    if not isinstance(calibration, bool):
        raise ValueError("model.encoder_norm_calibration must be true or false")
    if calibration and config["model"].get("encoder_norm", "group") != "centre":
        raise ValueError("model.encoder_norm_calibration scales centre norms; set model.encoder_norm: centre")
    image_input = config["model"].get("image_input", "rgb")
    if image_input not in IMAGE_INPUTS:
        raise ValueError(f"model.image_input must be one of {IMAGE_INPUTS}")
    if image_input == "luminance" and phase2.get("image_decoder", "qwt_coefficients") not in PIXEL_IMAGE_DECODERS:
        # Luminance coefficients synthesise back to Y: only a pixel decoder, which
        # reads the RGB frame itself, can put the colour back.
        raise ValueError("model.image_input luminance needs phase2.image_decoder resnet_pixel or split_color_edge")
    if config.get("encoder_sensitivity", {}).get("noise_direction", "corruption") not in NOISE_DIRECTIONS:
        raise ValueError(f"encoder_sensitivity.noise_direction must be one of {NOISE_DIRECTIONS}")
    for name, section in (("phase1", phase1), ("phase2", phase2)):
        if not isinstance(section.get("augment_hflip", False), bool):
            raise ValueError(f"{name}.augment_hflip must be true or false")
    degradation = phase1.get("degradation_weight", 0.0)
    if not _nonnegative_number(degradation):
        raise ValueError("phase1.degradation_weight must be a nonnegative number")
    condition = phase1.get("predictor_degradation_condition", False)
    if not isinstance(condition, bool):
        raise ValueError("phase1.predictor_degradation_condition must be true or false")
    if condition and not degradation > 0:
        raise ValueError("phase1.predictor_degradation_condition needs phase1.degradation_weight > 0 (its head)")
    if condition and phase1.get("predictor_type", "token") != "spatial":
        raise ValueError("phase1.predictor_degradation_condition needs phase1.predictor_type: spatial")
    predictor_input = phase2.get("decoder_predictor_input", False)
    if not isinstance(predictor_input, bool):
        raise ValueError("phase2.decoder_predictor_input must be true or false")
    if predictor_input and phase1.get("predictor_type", "token") != "spatial":
        raise ValueError("phase2.decoder_predictor_input needs phase1.predictor_type: spatial")
    if predictor_input and phase2.get("image_decoder", "qwt_coefficients") not in PIXEL_IMAGE_DECODERS:
        raise ValueError(f"phase2.decoder_predictor_input needs a pixel image decoder {PIXEL_IMAGE_DECODERS}")
    after = phase2.get("backbone_finetune_after_updates")
    if after is not None:
        if isinstance(after, bool) or not isinstance(after, int) or not 0 <= after < phase2["max_successful_updates"]:
            raise ValueError("phase2.backbone_finetune_after_updates must be an integer in "
                             "[0, phase2.max_successful_updates)")
        scale = phase2.get("backbone_finetune_lr_scale")
        if not _nonnegative_number(scale) or not 0 < scale <= 1:
            raise ValueError("phase2.backbone_finetune_lr_scale must be in (0, 1]")
    elif "backbone_finetune_lr_scale" in phase2:
        raise ValueError("phase2.backbone_finetune_lr_scale needs phase2.backbone_finetune_after_updates")
    if not _nonnegative_number(phase2.get("imu_increment_weight", 0.0)):
        raise ValueError("phase2.imu_increment_weight must be a nonnegative number")
    if phase2.get("imu_increment_weight", 0.0) > 0:
        windows = phase2.get("imu_increment_windows", [8, 32])
        if not _positive_ints(windows) or not windows or any(config["data"]["imu_window"] % w for w in windows):
            raise ValueError("phase2.imu_increment_windows must list window lengths that divide data.imu_window")


def build_phase1_model(
    config: dict[str, Any], normalizer: ImuNormalizer
) -> LatentPretrainingModel | IJEPAPretrainingModel:
    enabled = bool(config["phase1"].get("decoder_enabled", False))
    phase1 = config["phase1"]
    if phase1.get("objective", "jepa") == "ijepa":
        return IJEPAPretrainingModel(
            build_backbone(config), normalizer,
            predictor_dim=int(phase1["ijepa_predictor_dim"]),
            predictor_depth=int(phase1["ijepa_predictor_depth"]),
            predictor_heads=int(phase1["ijepa_predictor_heads"]),
            masking={"targets": int(phase1["ijepa_targets"]),
                     "target_scale": tuple(phase1["ijepa_target_scale"]),
                     "target_aspect": tuple(phase1["ijepa_target_aspect"]),
                     "context_scale": tuple(phase1["ijepa_context_scale"]),
                     "image_min_keep": int(phase1["ijepa_image_min_keep"]),
                     "imu_min_keep": int(phase1["ijepa_imu_min_keep"])},
            context_input=phase1["ijepa_context_input"],
        )
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
        finer_scale=float(phase1.get("multiscale_finer_weight", 0.0)) > 0,
        # Absent before the sharpness plan: no degradation head, no condition.
        degradation_outputs=len(DEGRADATION_FEATURES) if float(phase1.get("degradation_weight", 0.0)) > 0 else 0,
        degradation_condition=bool(phase1.get("predictor_degradation_condition", False)),
        # Neo phase 1 khong bao gio nhan skip: neu no co duong vong tu encoder thi
        # no thoa man duoc neo ma khong ep gi vao latent — dung cai ma neo sinh ra
        # de ngan. Cung ly do voi viec no giu he so tuyet doi thay vi residual.
        # The phase-1 anchor predicts clean COEFFICIENTS from the latent alone;
        # phase2.image_decoder must never reach it, or the phase-1 model changes.
        decoders=build_decoders(config, residual=False, skips=False,
                                image_decoder="qwt_coefficients", predictor_input=False) if enabled else None,
    )


BACKBONE_WEIGHTS = ("context", "target")


def phase2_backbone(config: dict[str, Any], phase1_model) -> nn.Module:
    """The phase-1 backbone phase 2 freezes. phase2.backbone_weights: context (absent: the
    online encoder every run so far used) or target (the EMA teacher's encoder weights, which
    I-JEPA evaluates with)."""
    if config["phase2"].get("backbone_weights", "context") == "target":
        phase1_model.teachers.load_into(phase1_model.backbone)
    return phase1_model.backbone


def phase2_latent_modules(
    config: dict[str, Any], phase1_model: LatentPretrainingModel | None = None
) -> tuple[nn.Module | None, nn.Module | None]:
    """(image predictor, degradation head) that phase 2 keeps, or (None, None).

    With ``phase1_model`` they are its trained modules (train-phase2); without, the
    same architecture is built fresh for a phase-2 checkpoint's weights to fill.
    """
    if not bool(config["phase2"].get("decoder_predictor_input", False)):
        return None, None
    if phase1_model is None:
        phase1_model = build_phase1_model(config, ImuNormalizer())
    head = phase1_model.degradation_head if phase1_model.degradation_condition else None
    return phase1_model.image_predictor, head


def build_decoders(
    config: dict[str, Any], *, residual: bool | None = None, skips: bool | None = None,
    image_decoder: str | None = None, predictor_input: bool | None = None,
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
    if predictor_input is None:
        # Absent before the sharpness plan: no merge, the decoders every checkpoint holds.
        predictor_input = bool(config["phase2"].get("decoder_predictor_input", False))
    return LatentDecoders(
        predictor_merge=predictor_input,
        image_coefficient_size=(image_size[0] // 2, image_size[1] // 2),
        image_coefficient_channels=48 if config["model"].get("image_input", "rgb") == "rgb" else 16,
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
        imu_refiner={
            "width": int(config["phase2"].get("imu_refiner_width", 32)),
            "blocks": int(config["phase2"]["imu_refiner_blocks"]),
        } if int(config["phase2"].get("imu_refiner_blocks", 0)) > 0 else None,
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
                # Absent before p16_fourier_imu: no wavelet-Fourier blocks.
                "fourier_levels": tuple(config["phase2"].get("split_edge_naf_fourier_levels", ())),
                "fourier_width": int(config["phase2"].get("split_edge_naf_fourier_width", 16)),
                # Absent before p16_infomax: no encoder stages. Level l (1/2^l of the frame)
                # takes the image encoder's stage at that resolution: channels[l - 1].
                "stage_channels": {int(level): int(channels[int(level) - 1]) for level in
                                   config["phase2"].get("split_edge_naf_stage_levels", ())},
            } if config["phase2"].get("split_branch_arch") == "nafnet_edge" else None,
            # Absent before p22_light: no light branch.
            "light": {
                "width": int(config["phase2"]["split_light_width"]),
                "scale": int(config["phase2"]["split_light_scale"]),
                "levels": int(config["phase2"]["split_light_levels"]),
            } if config["phase2"].get("split_light_branch", False) else None,
            "refiner": {
                "width": int(config["phase2"].get("split_edge_refiner_width", 32)),
                "blocks": int(config["phase2"].get("split_edge_refiner_blocks", 0)),
                "scale": int(config["phase2"].get("split_edge_refiner_scale", 1)),
            } if int(config["phase2"].get("split_edge_refiner_blocks", 0)) > 0 else None,
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
