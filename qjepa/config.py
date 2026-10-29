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
from .corruptions.halo import HaloFlareBank
from .corruptions.lens_flare import LensFlareBank
from .corruptions.image import DEGRADATION_FEATURES
from .data.normalize import ImuNormalizer
from .transforms import IMAGE_INPUTS, QWT_BACKENDS
from .models import LatentDecoders, LatentPretrainingModel, MultimodalBackbone
from .models.backbone import ENCODER_TYPES
from .models.blocks import ENCODER_NORMS
from .models.ijepa import CONTEXT_INPUTS, IJEPAPretrainingModel
from .models.vit import INPUT_STANDARDIZATIONS, TOKEN_STRIDE
from .models.decoders import IMAGE_LATENTS, PIXEL_IMAGE_DECODERS
from .models.predictors import PREDICTOR_TYPES
from .training.losses import LOWFREQ_SCALE
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
    _validate_exposure(config)
    _validate_tone_grid(config)
    _validate_relight(config)
    _validate_chroma(config)
    _validate_tone_gate(config)
    _validate_exposure_standardize(config)
    _validate_source_size(config)
    _validate_flare_pairs(config)
    _validate_scene_pairs(config)
    _validate_imu_absent(config)
    _validate_source_scale(config)
    _validate_lowlight_pairs(config)
    _validate_restore(config)
    _validate_flare_layer(config)
    _validate_flare_stage(config)
    _validate_target_brighten(config)
    _validate_kind_balance(config)
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
    if image_decoder not in ("qwt_coefficients", "resnet_pixel", "split_color_edge", "restore_rgb"):
        raise ValueError("phase2.image_decoder must be qwt_coefficients, resnet_pixel, split_color_edge or restore_rgb")
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
    # p49: train / valid / test share of the TartanAir motion groups. Absent: 0.8 / 0.1 / 0.1 as before.
    if "split_ratios" in data:
        ratios = data["split_ratios"]
        if (not isinstance(ratios, (list, tuple)) or len(ratios) != 3 or not all(_nonnegative_number(v) for v in ratios)
                or not np.isclose(sum(ratios), 1.0) or ratios[0] <= 0 or ratios[1] <= 0 or ratios[2] <= 0):
            raise ValueError("data.split_ratios must be [train, valid, test], each > 0, summing to 1")
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


# p48: the clean-frame decoder on I-JEPA's latent (models/ijepa.py ReconstructionHead).
IJEPA_RECONSTRUCTION_KEYS = ("ijepa_reconstruction_weight", "ijepa_reconstruction_width",
                             "ijepa_reconstruction_detail_weight")


def _validate_ijepa_reconstruction(phase1: dict[str, Any]) -> None:
    """Absent weight (or 0): no decoder, and no width or detail weight that would say otherwise. A positive
    weight needs both spelled out, so the hash records the decoder that was trained."""
    weight = phase1.get("ijepa_reconstruction_weight", 0.0)
    if not _nonnegative_number(weight):
        raise ValueError("phase1.ijepa_reconstruction_weight must be a nonnegative number")
    if weight == 0:
        extra = [key for key in IJEPA_RECONSTRUCTION_KEYS[1:] if key in phase1]
        if extra:
            raise ValueError(f"phase1.{extra[0]} needs a positive phase1.ijepa_reconstruction_weight")
        return
    if not _positive_int(phase1.get("ijepa_reconstruction_width")):
        raise ValueError("phase1.ijepa_reconstruction_weight > 0 needs a positive integer "
                         "phase1.ijepa_reconstruction_width")
    if not _nonnegative_number(phase1.get("ijepa_reconstruction_detail_weight")):
        raise ValueError("phase1.ijepa_reconstruction_weight > 0 needs a nonnegative "
                         "phase1.ijepa_reconstruction_detail_weight")


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
        present = [key for key in IJEPA_RECONSTRUCTION_KEYS if key in phase1]
        if present:
            raise ValueError(f"phase1.{present[0]} belongs to phase1.objective: ijepa")
        return
    _validate_ijepa_reconstruction(phase1)
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
HALO_KEYS = tuple(field.name for field in fields(LowLightImageCorruptionConfig) if field.name.startswith("halo_"))
LENS_KEYS = tuple(field.name for field in fields(LowLightImageCorruptionConfig) if field.name.startswith("lens_"))
# Optional switches inside a group: absent means the old behaviour, so they need not be written out.
OPTIONAL_KEYS = ("illum_highlight_rolloff", "illum_max_brighten_stops",
                 # p45: absent means no Wu et al. layers (fraction 0), so p39's config hashes and draws as before; with
                 # the fraction on, _require_wu_keys wants the other two written out.
                 "lens_flare_wu_fraction", "lens_flare_wu_captured_fraction", "lens_flare_wu_gain",
                 "lens_wu_ghost_boost", "lens_wu_ghost_radius", "lens_wu_ghost_blades", "lens_wu_ghost_round",
                 "lens_wu_min_flare")
WU_LENS_KEYS = ("lens_flare_wu_captured_fraction", "lens_flare_wu_gain", "lens_wu_ghost_boost", "lens_wu_ghost_radius",
                "lens_wu_ghost_blades", "lens_wu_ghost_round", "lens_wu_min_flare")


def _require_wu_keys(lens: dict[str, Any], where: str) -> None:
    if float(lens.get("lens_flare_wu_fraction", 0.0)) > 0:
        missing = [key for key in WU_LENS_KEYS if key not in lens]
        if missing:
            raise ValueError(f"{where}.lens_flare_wu_fraction > 0 needs {', '.join(missing)} written out (the hash "
                             "records only what the config says)")


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
                         ("fog_probability", FOG_KEYS), ("halo_probability", HALO_KEYS),
                         ("lens_flare_probability", LENS_KEYS)):
        if float(image.get(switch, 0.0)) > 0:
            missing = [key for key in keys if key not in image and key not in OPTIONAL_KEYS]
            if missing:
                raise ValueError(f"corruption.image.{switch} > 0 needs every {switch.split('_')[0]}_* key written "
                                 f"out (the hash records only what the config says); missing: {', '.join(missing)}")
    _require_wu_keys(image, "corruption.image")
    if float(image.get("lens_wu_min_flare", 0.0)) > 0:
        raise ValueError("corruption.image.lens_wu_min_flare: the redraw of faint Wu layers is the flare stage's "
                         "(flare.lens); keep it 0 here")
    if float(image.get("halo_probability", 0.0)) > 0:
        root = config["data"].get("halo_root")
        if not isinstance(root, str) or not root:
            raise ValueError("corruption.image.halo_probability > 0 needs data.halo_root: the folder holding "
                             "halo_index.csv (Kaggle dataset halo-reflective-1280)")
    if float(image.get("lens_flare_probability", 0.0)) > 0:
        root = config["data"].get("lens_flare_root")
        if not isinstance(root, str) or not root:
            raise ValueError("corruption.image.lens_flare_probability > 0 needs data.lens_flare_root: the folder holding "
                             "Flare7K/ and Flare-R/ (Kaggle dataset khoidb/flare7k)")


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


FLARE_PAIR_KEYS = ("flare_pairs_root", "flare_pairs_fraction", "flare_pairs_short_side",
                   "flare_pairs_holdout_fraction", "flare_pairs_clear_probability")
# Optional (absent: none, as before p40): "dihedral" turns each training pair's crop by k x 90 degrees and maybe flips it.
FLARE_PAIR_OPTIONAL_KEYS = ("flare_pairs_augment",)
FLARE_PAIR_AUGMENTS = ("none", "dihedral")


def _validate_flare_pairs(config: dict[str, Any]) -> None:
    """data.flare_pairs_*: FlareImage pairs in every training batch (qjepa/data/flare_pairs.py). Absent: as before.

    Bat (flare_pairs_fraction > 0) thi PHAI ghi du moi khoa (hash chi doc file config). flare_pairs_root la duong
    dan, nam ngoai hash nhu halo_root."""
    data = config["data"]
    known = FLARE_PAIR_KEYS + FLARE_PAIR_OPTIONAL_KEYS
    unknown = sorted(key for key in data if key.startswith("flare_pairs_") and key not in known)
    if unknown:
        raise ValueError(f"data: unknown keys {unknown}; FlareImage keys are {list(known)}")
    augment = data.get("flare_pairs_augment", "none")
    if augment not in FLARE_PAIR_AUGMENTS:
        raise ValueError(f"data.flare_pairs_augment must be one of {list(FLARE_PAIR_AUGMENTS)}")
    if augment == "dihedral" and data["image_size"][0] != data["image_size"][1]:
        raise ValueError("data.flare_pairs_augment dihedral turns the crop by 90 degrees: data.image_size must be square")
    fraction = data.get("flare_pairs_fraction", 0.0)
    if not _nonnegative_number(fraction) or fraction > 0.5:
        raise ValueError("data.flare_pairs_fraction must be a number in [0, 0.5]")
    if fraction == 0:
        return
    missing = [key for key in FLARE_PAIR_KEYS if key not in data]
    if missing:
        raise ValueError(f"data.flare_pairs_fraction > 0 needs every flare_pairs_* key written out; missing: "
                         f"{', '.join(missing)}")
    if not isinstance(data["flare_pairs_root"], str) or not data["flare_pairs_root"]:
        raise ValueError("data.flare_pairs_root must name the FlareImage folder (metadata.csv, gt/, train/)")
    short_side = data["flare_pairs_short_side"]
    if isinstance(short_side, bool) or not isinstance(short_side, int) or short_side < max(data["image_size"]):
        raise ValueError("data.flare_pairs_short_side must be an integer of at least the training crop data.image_size")
    holdout = data["flare_pairs_holdout_fraction"]
    if not _nonnegative_number(holdout) or not 0 < holdout <= 0.4:
        raise ValueError("data.flare_pairs_holdout_fraction must be in (0, 0.4]: valid and test each keep that much")
    clear = data["flare_pairs_clear_probability"]
    if not _nonnegative_number(clear) or clear > 1:
        raise ValueError("data.flare_pairs_clear_probability must be in [0, 1]")
    minimum = data.get("minimum_trajectories_per_batch", 1)
    for phase in ("phase1", "phase2"):
        size = config[phase]["batch_size"]
        slots = max(1, round(fraction * size))
        if size - slots < minimum:
            raise ValueError(f"data.flare_pairs_fraction {fraction}: {slots} of {phase}.batch_size {size} are "
                             f"FlareImage, leaving fewer than minimum_trajectories_per_batch {minimum} TartanAir samples")


SCENE_PAIR_KEYS = ("scene_pairs_root", "scene_pairs_fraction", "scene_pairs_short_side",
                   "scene_pairs_flare_probability")
# p49, optional: absent = the index.csv split as before (train / valid, no test).
SCENE_PAIR_OPTIONAL_KEYS = ("scene_pairs_holdout_fraction",)


def _validate_scene_pairs(config: dict[str, Any]) -> None:
    """p46 data.scene_pairs_*: scene pairs (photo with a composited flare / clean photo, e.g. VisDrone) in every
    training batch of both phases, beside the FlareImage pairs (qjepa/data/scene_pairs.py). Absent: as before.

    On (scene_pairs_fraction > 0): every key written out (the hash reads only the config file); scene_pairs_root is
    a path, out of the hash like flare_pairs_root. FlareImage, scene pairs and minimum_trajectories_per_batch
    TartanAir samples must all fit each batch."""
    data = config["data"]
    known = SCENE_PAIR_KEYS + SCENE_PAIR_OPTIONAL_KEYS
    unknown = sorted(key for key in data if key.startswith("scene_pairs_") and key not in known)
    if unknown:
        raise ValueError(f"data: unknown keys {unknown}; scene-pair keys are {list(known)}")
    if "scene_pairs_holdout_fraction" in data:
        holdout = data["scene_pairs_holdout_fraction"]
        if not _nonnegative_number(holdout) or not 0 < holdout <= 0.25:
            raise ValueError("data.scene_pairs_holdout_fraction must be in (0, 0.25]: test and valid each keep that much")
    fraction = data.get("scene_pairs_fraction", 0.0)
    if not _nonnegative_number(fraction) or fraction > 0.5:
        raise ValueError("data.scene_pairs_fraction must be a number in [0, 0.5]")
    if fraction == 0:
        return
    missing = [key for key in SCENE_PAIR_KEYS if key not in data]
    if missing:
        raise ValueError(f"data.scene_pairs_fraction > 0 needs every scene_pairs_* key written out; missing: "
                         f"{', '.join(missing)}")
    if not isinstance(data["scene_pairs_root"], str) or not data["scene_pairs_root"]:
        raise ValueError("data.scene_pairs_root must name the scene-pair folder (index.csv, gt/, input/)")
    short_side = data["scene_pairs_short_side"]
    if isinstance(short_side, bool) or not isinstance(short_side, int) or short_side < max(data["image_size"]):
        raise ValueError("data.scene_pairs_short_side must be an integer of at least the training crop data.image_size")
    flared = data["scene_pairs_flare_probability"]
    if not _nonnegative_number(flared) or flared > 1:
        raise ValueError("data.scene_pairs_flare_probability must be in [0, 1]")
    if data["image_size"][0] != data["image_size"][1]:
        raise ValueError("data.scene_pairs_* turn each training crop by k x 90 degrees: data.image_size must be square")
    minimum = data.get("minimum_trajectories_per_batch", 1)
    flare_fraction = float(data.get("flare_pairs_fraction", 0.0))
    for phase in ("phase1", "phase2"):
        size = config[phase]["batch_size"]
        scene_slots = max(1, round(fraction * size))
        flare_slots = max(1, round(flare_fraction * size)) if flare_fraction > 0 else 0
        if size - scene_slots - flare_slots < minimum:
            raise ValueError(f"data.scene_pairs_fraction {fraction}: {scene_slots} scene pairs and {flare_slots} "
                             f"FlareImage pairs of {phase}.batch_size {size} leave fewer than "
                             f"minimum_trajectories_per_batch {minimum} TartanAir samples")


def _validate_source_scale(config: dict[str, Any]) -> None:
    """p47 data.source_scale_range [low, high]: each frame is read at a drawn fraction of data.source_size before the
    training crop, so the same scenes come at many pixel scales. Absent: at source_size, as before."""
    value = config["data"].get("source_scale_range")
    if value is None:
        return
    source = config["data"].get("source_size")
    if not source:
        raise ValueError("data.source_scale_range scales data.source_size: set data.source_size")
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_nonnegative_number(v) for v in value)
            or not 0 < value[0] <= value[1] <= 1):
        raise ValueError("data.source_scale_range must be [low, high] with 0 < low <= high <= 1")
    if any(round(side * value[0]) < crop for side, crop in zip(source, config["data"]["image_size"])):
        raise ValueError("data.source_scale_range: source_size x low must still hold the training crop data.image_size")


LOWLIGHT_PAIR_KEYS = ("lowlight_pairs_root", "lowlight_pairs_fraction", "lowlight_pairs_short_side")


def _validate_lowlight_pairs(config: dict[str, Any]) -> None:
    """p47 data.lowlight_pairs_*: real low-light photo pairs (LOL; lowlight-pairs-builder.ipynb) in every training batch
    of both phases, beside the FlareImage and scene pairs. Absent: as before. On: every key written out;
    lowlight_pairs_root is a path, out of the hash. All the pairs and minimum_trajectories_per_batch TartanAir samples
    must fit each batch."""
    data = config["data"]
    unknown = sorted(key for key in data if key.startswith("lowlight_pairs_") and key not in LOWLIGHT_PAIR_KEYS)
    if unknown:
        raise ValueError(f"data: unknown keys {unknown}; low-light pair keys are {list(LOWLIGHT_PAIR_KEYS)}")
    fraction = data.get("lowlight_pairs_fraction", 0.0)
    if not _nonnegative_number(fraction) or fraction > 0.5:
        raise ValueError("data.lowlight_pairs_fraction must be a number in [0, 0.5]")
    if fraction == 0:
        return
    missing = [key for key in LOWLIGHT_PAIR_KEYS if key not in data]
    if missing:
        raise ValueError(f"data.lowlight_pairs_fraction > 0 needs every lowlight_pairs_* key written out; missing: "
                         f"{', '.join(missing)}")
    if not isinstance(data["lowlight_pairs_root"], str) or not data["lowlight_pairs_root"]:
        raise ValueError("data.lowlight_pairs_root must name the low-light pair folder (index.csv, gt/, input/)")
    short_side = data["lowlight_pairs_short_side"]
    if isinstance(short_side, bool) or not isinstance(short_side, int) or short_side < max(data["image_size"]):
        raise ValueError("data.lowlight_pairs_short_side must be an integer of at least data.image_size")
    if data["image_size"][0] != data["image_size"][1]:
        raise ValueError("data.lowlight_pairs_* turn each training crop by k x 90 degrees: image_size must be square")
    minimum = data.get("minimum_trajectories_per_batch", 1)
    for phase in ("phase1", "phase2"):
        size = config[phase]["batch_size"]
        slots = sum(max(1, round(float(data.get(key, 0.0)) * size)) for key in (
            "flare_pairs_fraction", "scene_pairs_fraction", "lowlight_pairs_fraction")
            if float(data.get(key, 0.0)) > 0)
        if size - slots < minimum:
            raise ValueError(f"{slots} pair slots of {phase}.batch_size {size} leave fewer than "
                             f"minimum_trajectories_per_batch {minimum} TartanAir samples")


RESTORE_KEYS = ("restore_widths", "restore_enc_blocks", "restore_middle_blocks", "restore_dec_blocks",
                "restore_fourier_levels", "restore_fourier_width", "restore_tone_width", "restore_tone_weight",
                "restore_fft_weight")
# p51 phase2.restore_net: nafnet (absent: as before) or restormer (the released Restormer, qjepa/models/restormer.py).
# With restormer the NAFNet's shape keys must be null (the YAML merge cannot drop them) and these are written out.
RESTORE_NETS = ("nafnet", "restormer")
RESTORE_NAF_KEYS = ("restore_widths", "restore_enc_blocks", "restore_middle_blocks", "restore_dec_blocks",
                    "restore_fourier_levels", "restore_fourier_width")
RESTORE_RESTORMER_KEYS = ("restore_restormer_weights", "restore_pretrained_lr_scale")
# p52, Restormer only, each optional (absent: as p51): the Restormer trains on crops, adds detail only, the light gets
# a bilateral grid after the tone curve, and activation checkpointing can go (RestoreRGBDecoder's docstring).
RESTORE_DETAIL_KEYS = ("restore_train_crop", "restore_highpass_sigma", "restore_light_grid", "restore_checkpointing")
MOTION_FIELD_KEYS = ("seconds", "samples", "focal_px", "scale_px", "max_px")
# The split decoder's stages: restore_rgb has none of them, so they must be off (or absent) beside it.
SPLIT_ONLY_SWITCHES = ("split_light_branch", "split_tone_grid", "split_relight", "split_chroma_detail",
                       "split_tone_gate", "split_exposure_stats", "split_exposure_standardize",
                       "split_color_global", "decoder_predictor_input", "encoder_skips")


def _validate_restore(config: dict[str, Any]) -> None:
    """p47 phase2.image_decoder restore_rgb (RestoreRGBDecoder), phase2.motion_field and phase2.image_latent.

    restore_rgb: every restore_* key written out (the hash reads only the config file), and none of the split
    decoder's stages switched on. motion_field: only with restore_rgb, and only when the gyro is what blurs the frame
    (corruption.image.motion_from_imu). image_latent: zi (absent: as before) or none."""
    phase2 = config["phase2"]
    latent = phase2.get("image_latent", "zi")
    if latent not in IMAGE_LATENTS:
        raise ValueError(f"phase2.image_latent must be one of {IMAGE_LATENTS}")
    if latent == "none" and phase2.get("image_decoder") not in ("split_color_edge", "restore_rgb"):
        raise ValueError("phase2.image_latent none is for the pixel decoders split_color_edge and restore_rgb")
    field = phase2.get("motion_field")
    restore = phase2.get("image_decoder") == "restore_rgb"
    known = (*RESTORE_KEYS, "restore_net", *RESTORE_RESTORMER_KEYS, *RESTORE_DETAIL_KEYS)
    unknown = sorted(key for key in phase2 if key.startswith("restore_") and key not in known)
    if unknown:
        raise ValueError(f"phase2: unknown keys {unknown}; restore_rgb's keys are {list(known)}")
    if not restore:
        if field is not None or any(key in phase2 for key in known):
            raise ValueError("phase2.motion_field and phase2.restore_* belong to phase2.image_decoder restore_rgb")
        return
    net = phase2.get("restore_net", "nafnet")
    if net not in RESTORE_NETS:
        raise ValueError(f"phase2.restore_net must be one of {RESTORE_NETS}")
    if net == "restormer":
        _validate_restormer(config)
        return
    if any(key in phase2 for key in (*RESTORE_RESTORMER_KEYS, *RESTORE_DETAIL_KEYS)):
        raise ValueError(f"phase2.{'/'.join((*RESTORE_RESTORMER_KEYS, *RESTORE_DETAIL_KEYS))} belong to "
                         f"phase2.restore_net: restormer")
    missing = [key for key in RESTORE_KEYS if key not in phase2]
    if missing:
        raise ValueError(f"phase2.image_decoder restore_rgb needs every restore_* key written out; missing: "
                         f"{', '.join(missing)}")
    switched = [key for key in SPLIT_ONLY_SWITCHES if phase2.get(key, False)]
    if switched or "split_tone_gate_lr_scale" in phase2 or phase2.get("split_edge_refiner_blocks", 0):
        raise ValueError(f"phase2.image_decoder restore_rgb has none of the split decoder's stages: turn off "
                         f"{switched or ['split_tone_gate_lr_scale / split_edge_refiner_blocks']}")
    widths = phase2["restore_widths"]
    if not _positive_ints(widths) or len(widths) < 2:
        raise ValueError("phase2.restore_widths must list at least two positive channel counts")
    if any(side % 2 ** (len(widths) - 1) for side in config["data"]["image_size"]):
        raise ValueError(f"phase2.restore_widths: image sides must divide by {2 ** (len(widths) - 1)}")
    for key in ("restore_enc_blocks", "restore_dec_blocks"):
        if not _positive_ints(phase2[key], len(widths) - 1):
            raise ValueError(f"phase2.{key} must list len(restore_widths) - 1 positive block counts")
    for key in ("restore_middle_blocks", "restore_fourier_width", "restore_tone_width"):
        if not _positive_int(phase2[key]):
            raise ValueError(f"phase2.{key} must be a positive integer")
    levels = phase2["restore_fourier_levels"]
    if (not isinstance(levels, (list, tuple)) or len(set(levels)) != len(levels)
            or any(isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= len(widths) - 2
                   for level in levels)):
        raise ValueError(f"phase2.restore_fourier_levels must list distinct levels in 0..{len(widths) - 2}")
    _validate_restore_common(config)


def _validate_restormer(config: dict[str, Any]) -> None:
    """p51 phase2.restore_net restormer: the tone and loss keys and both Restormer keys written out, the NAFNet's shape
    keys null or absent (they would describe a network that is not there)."""
    from .models.restormer import RESTORMER_WEIGHTS
    phase2 = config["phase2"]
    missing = [key for key in ("restore_tone_width", "restore_tone_weight", "restore_fft_weight",
                               *RESTORE_RESTORMER_KEYS) if key not in phase2]
    if missing:
        raise ValueError(f"phase2.restore_net restormer needs {', '.join(missing)} written out")
    stale = [key for key in RESTORE_NAF_KEYS if phase2.get(key) is not None]
    if stale:
        raise ValueError(f"phase2.restore_net restormer has no NAFNet: set {', '.join(stale)} to null")
    switched = [key for key in SPLIT_ONLY_SWITCHES if phase2.get(key, False)]
    if switched:
        raise ValueError(f"phase2.image_decoder restore_rgb has none of the split decoder's stages: turn off {switched}")
    if phase2["restore_restormer_weights"] not in RESTORMER_WEIGHTS:
        raise ValueError(f"phase2.restore_restormer_weights must be one of {sorted(RESTORMER_WEIGHTS)}")
    scale = phase2["restore_pretrained_lr_scale"]
    if not _nonnegative_number(scale) or not 0 < scale <= 1:
        raise ValueError("phase2.restore_pretrained_lr_scale must be in (0, 1]")
    if not _positive_int(phase2["restore_tone_width"]):
        raise ValueError("phase2.restore_tone_width must be a positive integer")
    _validate_restore_detail(config)
    _validate_restore_common(config)


def _validate_restore_detail(config: dict[str, Any]) -> None:
    """p52 RESTORE_DETAIL_KEYS. restore_train_crop: on the 8-px grid of the Restormer's bottom level and of the low-
    frequency loss, smaller than the frame (the VGG term scores the whole window when it is not above perceptual_crop).
    restore_highpass_sigma: its blur's reach (3 sigma) inside the window the network sees."""
    from .models.restormer import STEP
    phase2 = config["phase2"]
    sides = config["data"]["image_size"]
    crop = phase2.get("restore_train_crop")
    if crop is not None:
        if not _positive_int(crop) or crop % STEP or crop % LOWFREQ_SCALE or crop >= min(sides):
            raise ValueError(f"phase2.restore_train_crop must be a multiple of {max(STEP, LOWFREQ_SCALE)} smaller "
                             f"than the frame {list(sides)}")
    sigma = phase2.get("restore_highpass_sigma")
    if sigma is not None and (not _nonnegative_number(sigma) or sigma <= 0
                              or np.ceil(3 * sigma) >= (crop or min(sides))):
        raise ValueError("phase2.restore_highpass_sigma must be positive with 3 x sigma inside the restored window")
    grid = phase2.get("restore_light_grid")
    if grid is not None and (not _positive_int(grid) or grid < 2):
        raise ValueError("phase2.restore_light_grid must be an integer >= 2 (cells per side)")
    if not isinstance(phase2.get("restore_checkpointing", True), bool):
        raise ValueError("phase2.restore_checkpointing must be true or false")


def _validate_restore_common(config: dict[str, Any]) -> None:
    """restore_rgb's loss weights and phase2.motion_field, whichever network restores."""
    phase2 = config["phase2"]
    field = phase2.get("motion_field")
    for key in ("restore_tone_weight", "restore_fft_weight"):
        if not _nonnegative_number(phase2[key]):
            raise ValueError(f"phase2.{key} must be a nonnegative number")
    if phase2["restore_tone_weight"] > 0 and any(side % LOWFREQ_SCALE for side in config["data"]["image_size"]):
        raise ValueError(f"phase2.restore_tone_weight needs image sides divisible by {LOWFREQ_SCALE}")
    if field is None:
        return
    if not isinstance(field, dict) or set(field) != set(MOTION_FIELD_KEYS):
        raise ValueError(f"phase2.motion_field must give exactly {', '.join(MOTION_FIELD_KEYS)}")
    samples = field["samples"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3 or samples % 2 == 0:
        raise ValueError("phase2.motion_field.samples must be an odd integer >= 3")
    for key in ("seconds", "focal_px", "scale_px", "max_px"):
        if not _nonnegative_number(field[key]) or field[key] <= 0:
            raise ValueError(f"phase2.motion_field.{key} must be a positive number")
    if not config["corruption"]["image"].get("motion_from_imu", False):
        raise ValueError("phase2.motion_field reads the gyro that blurs the frame: set "
                         "corruption.image.motion_from_imu: true")


def _validate_imu_absent(config: dict[str, Any]) -> None:
    """p46 data.imu_absent_probability: the share of training samples that come without IMU (all-zero window), so the
    model also restores an image alone. Absent: every sample has its IMU, as before. Below 1: the IMU branch must
    still learn from the rest."""
    value = config["data"].get("imu_absent_probability", 0.0)
    if not _nonnegative_number(value) or value >= 1:
        raise ValueError("data.imu_absent_probability must be a number in [0, 1)")


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


def _validate_exposure(config: dict[str, Any]) -> None:
    """p30: phase2.split_exposure_stats and phase2.lowfreq_mse_weight. Absent: off, as before."""
    phase2 = config["phase2"]
    stats = phase2.get("split_exposure_stats", False)
    if not isinstance(stats, bool):
        raise ValueError("phase2.split_exposure_stats must be true or false")
    if stats and not (phase2.get("image_decoder") == "split_color_edge"
                      and (phase2.get("split_color_global", False) or phase2.get("split_light_branch", False))):
        raise ValueError("phase2.split_exposure_stats feeds the global tone head (split_color_global) or the light "
                         "branch (split_light_branch) of the split_color_edge decoder; turn one of them on")
    weight = phase2.get("lowfreq_mse_weight", 0.0)
    if not _nonnegative_number(weight):
        raise ValueError("phase2.lowfreq_mse_weight must be a nonnegative number")
    if weight > 0 and any(side % LOWFREQ_SCALE for side in config["data"]["image_size"]):
        raise ValueError(f"phase2.lowfreq_mse_weight needs image sides divisible by {LOWFREQ_SCALE}")


TONE_GRID_KEYS = ("split_tone_grid_size", "split_tone_grid_bins", "split_tone_grid_width", "split_tone_grid_weight")


def _validate_tone_grid(config: dict[str, Any]) -> None:
    """p31: phase2.split_tone_grid, the bilateral grid of colour affines that replaces the light branch.

    On, every split_tone_grid_* key must be written out (the hash reads only the config file)."""
    phase2 = config["phase2"]
    enabled = phase2.get("split_tone_grid", False)
    if not isinstance(enabled, bool):
        raise ValueError("phase2.split_tone_grid must be true or false")
    if not enabled:
        return
    if phase2.get("image_decoder") != "split_color_edge":
        raise ValueError("phase2.split_tone_grid works in the split_color_edge decoder")
    if phase2.get("split_light_branch", False):
        raise ValueError("phase2.split_tone_grid replaces the light branch: set phase2.split_light_branch: false")
    missing = [key for key in TONE_GRID_KEYS if key not in phase2]
    if missing:
        raise ValueError(f"phase2.split_tone_grid needs every split_tone_grid_* key written out; missing: "
                         f"{', '.join(missing)}")
    for key, low in (("split_tone_grid_size", 2), ("split_tone_grid_bins", 2), ("split_tone_grid_width", 1)):
        value = phase2[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"phase2.{key} must be an integer >= {low}")
    if not _nonnegative_number(phase2["split_tone_grid_weight"]):
        raise ValueError("phase2.split_tone_grid_weight must be a nonnegative number")
    if any(side % LOWFREQ_SCALE for side in config["data"]["image_size"]):
        raise ValueError(f"phase2.split_tone_grid needs image sides divisible by {LOWFREQ_SCALE}")


RELIGHT_KEYS = ("split_relight_size", "split_relight_widths", "split_relight_weight")


def _validate_relight(config: dict[str, Any]) -> None:
    """p32: phase2.split_relight, the stop map divided out before the bilateral grid.

    On, every split_relight_* key must be written out (the hash reads only the config file)."""
    phase2 = config["phase2"]
    enabled = phase2.get("split_relight", False)
    if not isinstance(enabled, bool):
        raise ValueError("phase2.split_relight must be true or false")
    if not enabled:
        return
    if not phase2.get("split_tone_grid", False):
        raise ValueError("phase2.split_relight leaves white balance and the tone curve to the bilateral grid: "
                         "set phase2.split_tone_grid: true")
    missing = [key for key in RELIGHT_KEYS if key not in phase2]
    if missing:
        raise ValueError(f"phase2.split_relight needs every split_relight_* key written out; missing: "
                         f"{', '.join(missing)}")
    size, widths = phase2["split_relight_size"], phase2["split_relight_widths"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 8:
        raise ValueError("phase2.split_relight_size must be an integer >= 8")
    if (not isinstance(widths, (list, tuple)) or len(widths) < 2
            or any(isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in widths)):
        raise ValueError("phase2.split_relight_widths must be a list of at least two positive integers")
    if size % 2 ** (len(widths) - 1):
        raise ValueError("phase2.split_relight_size must halve cleanly once per extra split_relight_widths level")
    if not _nonnegative_number(phase2["split_relight_weight"]):
        raise ValueError("phase2.split_relight_weight must be a nonnegative number")


CHROMA_KEYS = ("split_chroma_width", "split_chroma_blocks")


def _validate_chroma(config: dict[str, Any]) -> None:
    """p34: phase2.split_chroma_detail, full-resolution chroma detail on top of the colour branch.

    On, every split_chroma_* key must be written out (the hash reads only the config file)."""
    phase2 = config["phase2"]
    enabled = phase2.get("split_chroma_detail", False)
    if not isinstance(enabled, bool):
        raise ValueError("phase2.split_chroma_detail must be true or false")
    if not enabled:
        return
    if phase2.get("image_decoder") != "split_color_edge":
        raise ValueError("phase2.split_chroma_detail adds to the split_color_edge decoder's colour branch")
    missing = [key for key in CHROMA_KEYS if key not in phase2]
    if missing:
        raise ValueError(f"phase2.split_chroma_detail needs every split_chroma_* key written out; missing: "
                         f"{', '.join(missing)}")
    for key, low in (("split_chroma_width", 4), ("split_chroma_blocks", 1)):
        value = phase2[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"phase2.{key} must be an integer >= {low}")


TONE_GATE_KEYS = ("split_tone_gate_width", "split_tone_gate_weight")


def _validate_tone_gate(config: dict[str, Any]) -> None:
    """p35: phase2.split_tone_gate, how much of the tone stage each frame gets, learnt from the corruption.

    On, every split_tone_gate_* key must be written out (the hash reads only the config file)."""
    phase2 = config["phase2"]
    enabled = phase2.get("split_tone_gate", False)
    if not isinstance(enabled, bool):
        raise ValueError("phase2.split_tone_gate must be true or false")
    if not enabled:
        return
    if not phase2.get("split_tone_grid", False):
        raise ValueError("phase2.split_tone_gate blends the bilateral grid's output: set phase2.split_tone_grid: true")
    missing = [key for key in TONE_GATE_KEYS if key not in phase2]
    if missing:
        raise ValueError(f"phase2.split_tone_gate needs every split_tone_gate_* key written out; missing: "
                         f"{', '.join(missing)}")
    width = phase2["split_tone_gate_width"]
    if isinstance(width, bool) or not isinstance(width, int) or width < 4:
        raise ValueError("phase2.split_tone_gate_width must be an integer >= 4")
    if not _nonnegative_number(phase2["split_tone_gate_weight"]):
        raise ValueError("phase2.split_tone_gate_weight must be a nonnegative number")


FLARE_LAYER_KEYS = ("split_flare_widths", "split_flare_blocks", "split_flare_weight")


def _validate_flare_layer(config: dict[str, Any]) -> None:
    """p41: phase2.split_flare_layer, a full-resolution flare layer the split decoder subtracts first.

    On, every split_flare_* key must be written out (the hash reads only the config file). Its target comes from
    the FlareImage pairs, so data.flare_pairs_fraction must be on."""
    phase2 = config["phase2"]
    enabled = phase2.get("split_flare_layer", False)
    if not isinstance(enabled, bool):
        raise ValueError("phase2.split_flare_layer must be true or false")
    if not enabled:
        return
    if phase2.get("image_decoder") not in ("split_color_edge", "restore_rgb"):
        raise ValueError("phase2.split_flare_layer is a stage of the split_color_edge or restore_rgb decoder")
    if float(config["data"].get("flare_pairs_fraction", 0.0)) <= 0:
        raise ValueError("phase2.split_flare_layer learns from FlareImage pairs: set data.flare_pairs_fraction")
    missing = [key for key in FLARE_LAYER_KEYS if key not in phase2]
    if missing:
        raise ValueError(f"phase2.split_flare_layer needs every split_flare_* key written out; missing: "
                         f"{', '.join(missing)}")
    widths = phase2["split_flare_widths"]
    if (not isinstance(widths, (list, tuple)) or len(widths) < 2
            or any(isinstance(w, bool) or not isinstance(w, int) or w < 4 for w in widths)):
        raise ValueError("phase2.split_flare_widths must list at least two integers >= 4")
    blocks = phase2["split_flare_blocks"]
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ValueError("phase2.split_flare_blocks must be an integer >= 1")
    if not _nonnegative_number(phase2["split_flare_weight"]):
        raise ValueError("phase2.split_flare_weight must be a nonnegative number")
    for key in ("split_flare_latent", "split_flare_pretrained", "split_flare_freeze"):
        if key in phase2 and not isinstance(phase2[key], bool):
            raise ValueError(f"phase2.{key} must be true or false")
    if phase2.get("split_flare_freeze", False) and not phase2.get("split_flare_pretrained", False):
        raise ValueError("phase2.split_flare_freeze freezes the flare stage's layer: set split_flare_pretrained: true")
    if phase2.get("split_flare_pretrained", False):
        # p42: the layer comes from train-flare (<output>/flare/last.pt), which trains it without a backbone.
        if "flare" not in config:
            raise ValueError("phase2.split_flare_pretrained loads the flare stage's layer: add the flare section")
        if phase2.get("split_flare_latent", True):
            raise ValueError("phase2.split_flare_pretrained: the flare stage trains a layer without the latent; set "
                             "split_flare_latent: false")
        stage = config["flare"]
        if list(phase2["split_flare_widths"]) != list(stage["widths"]) or phase2["split_flare_blocks"] != stage["blocks"]:
            raise ValueError("phase2.split_flare_widths / split_flare_blocks must equal flare.widths / flare.blocks")
        # p44: the layer phase 2 builds is the one the stage trained (absent on both sides: p42's FlareLayer).
        if (phase2.get("split_flare_model", "unet") != stage.get("model", "unet")
                or phase2.get("split_flare_scale") != stage.get("uformer_scale")):
            raise ValueError("phase2.split_flare_model / split_flare_scale must equal flare.model / flare.uformer_scale")
    model = phase2.get("split_flare_model", "unet")
    _validate_flare_model(phase2, "phase2.split_flare_model", "phase2.split_flare_scale", model,
                          phase2.get("split_flare_scale"))
    if model != "unet" and not phase2.get("split_flare_pretrained", False):
        raise ValueError("phase2.split_flare_model: uformer_flare7kpp comes from the flare stage: set "
                         "split_flare_pretrained: true")


TARGET_BRIGHTEN_KEYS = ("below", "target", "max_gain")


def _validate_target_brighten(config: dict[str, Any]) -> None:
    """p43: phase2.target_brighten -- a phase-2 target darker than ``below`` (mean sRGB luma) is brightened towards
    ``target`` by one linear-light gain of at most ``max_gain`` (light.brighten_dark). Absent: targets as before."""
    value = config["phase2"].get("target_brighten")
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != set(TARGET_BRIGHTEN_KEYS):
        raise ValueError(f"phase2.target_brighten must give exactly {', '.join(TARGET_BRIGHTEN_KEYS)}")
    if not all(_nonnegative_number(value[key]) for key in TARGET_BRIGHTEN_KEYS):
        raise ValueError("phase2.target_brighten values must be nonnegative numbers")
    if not 0 < value["below"] <= value["target"] < 1:
        raise ValueError("phase2.target_brighten needs 0 < below <= target < 1")
    if not 1 < value["max_gain"] <= 8:
        raise ValueError("phase2.target_brighten.max_gain must be in (1, 8]")


# p44: optional. model: "unet" (absent, p42's FlareLayer) or "uformer_flare7kpp" (Flare7K++'s Uformer, fine-tuned from
# its released checkpoint, data.uformer_checkpoint); uformer_scale: the Uformer's upsampling of each frame (2: a 256
# crop to the 512 it was trained at); select_best: validate at update 0 and every validation_every_updates on the
# held-out FlareImage pairs and keep the best weights (flare/best.pt) -- the pretrained ones if none beats them.
# p45: uformer_init: "release" (absent: Flare7K++'s released weights) or "stage" (a previous run's flare/best.pt, e.g.
# p44's, in data.uformer_checkpoint); select_on: "real" (absent: PSNR on the held-out FlareImage pairs) or
# "real_and_composite" (the mean of the pairs' and the valid composites' PSNR gains -- the composites hold the Wu et
# al. daytime flares the pairs do not).
# p46: kinds may add "scene" (a scene pair of data.scene_pairs_*, e.g. VisDrone: its flared photo -> its clean one);
# background_scene_fraction: of the composites' backgrounds, the share that is a scene pair's clean photo (absent: 0,
# drawn as before); select_on "real_composite_scene" adds the valid scene pairs' PSNR gain to real_and_composite's.
# p53: veil_grid: a bilateral grid (that many cells a side) after the Uformer takes off the smooth veil it leaves
# (UformerFlareLayer); veil_only: train only that grid, the Uformer kept as uformer_init gives it. Absent: no grid.
FLARE_STAGE_OPTIONAL_KEYS = ("model", "uformer_scale", "select_best", "uformer_init", "select_on",
                             "background_scene_fraction", "veil_grid", "veil_only")
FLARE_SELECT_ON = ("real", "real_and_composite", "real_composite_scene")
FLARE_MODELS = ("unet", "uformer_flare7kpp")


def _validate_flare_model(owner: dict[str, Any], model_key: str, scale_key: str, model: Any, scale: Any) -> None:
    if model not in FLARE_MODELS:
        raise ValueError(f"{model_key} must be one of {', '.join(FLARE_MODELS)}")
    if model == "uformer_flare7kpp":
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0.5 <= scale <= 4:
            raise ValueError(f"{scale_key} must be a number in [0.5, 4] with {model_key}: uformer_flare7kpp")
    elif scale is not None:
        raise ValueError(f"{scale_key} is the Uformer's upsampling: set {model_key}: uformer_flare7kpp")


def _validate_kind_balance(config: dict[str, Any]) -> None:
    """p44: phase2.kind_balance -- each kind of frame of a microbatch (corruptions.image.LOG_KINDS) weighs the same
    in the image loss's gradients. Absent: every frame weighs the same, as before."""
    value = config["phase2"].get("kind_balance", False)
    if not isinstance(value, bool):
        raise ValueError("phase2.kind_balance must be true or false")


FLARE_STAGE_KEYS = ("widths", "blocks", "batch_size", "max_successful_updates", "learning_rate", "minimum_lr",
                    "warmup_updates", "weight_decay", "gradient_clip_norm", "precision", "kinds",
                    "background_pair_fraction", "blur_probability", "blur_sigma_px", "grain_probability",
                    "photon_count", "lens", "validation_every_updates", "validation_composites",
                    "checkpoint_every_updates", "seed")


def _validate_flare_stage(config: dict[str, Any]) -> None:
    """p42: the flare section, the flare stage (train-flare) that trains the flare layer alone before phase 2.

    Absent: no flare stage, as before. Present: every key written out (the hash reads only the config file), and
    data.lens_flare_root (a path, out of the hash) names the Flare7K++ folder."""
    if "flare" not in config:
        return
    stage = config["flare"]
    if not isinstance(stage, dict):
        raise ValueError("flare must be a mapping")
    missing = [key for key in FLARE_STAGE_KEYS if key not in stage]
    if missing:
        raise ValueError(f"flare needs every key written out; missing: {', '.join(missing)}")
    unknown = sorted(set(stage) - set(FLARE_STAGE_KEYS) - set(FLARE_STAGE_OPTIONAL_KEYS))
    if unknown:
        raise ValueError(f"flare: unknown keys {unknown}")
    _validate_flare_model(stage, "flare.model", "flare.uformer_scale", stage.get("model", "unet"),
                          stage.get("uformer_scale"))
    if "select_best" in stage and not isinstance(stage["select_best"], bool):
        raise ValueError("flare.select_best must be true or false")
    if stage.get("uformer_init", "release") not in ("release", "stage"):
        raise ValueError("flare.uformer_init must be release or stage")
    if "uformer_init" in stage and stage.get("model", "unet") != "uformer_flare7kpp":
        raise ValueError("flare.uformer_init starts the Uformer: set flare.model: uformer_flare7kpp")
    if stage.get("select_on", "real") not in FLARE_SELECT_ON:
        raise ValueError(f"flare.select_on must be one of {', '.join(FLARE_SELECT_ON)}")
    if "select_on" in stage and not stage.get("select_best", False):
        raise ValueError("flare.select_on chooses flare/best.pt: set flare.select_best: true")
    veil = stage.get("veil_grid")
    if veil is not None:
        if isinstance(veil, bool) or not isinstance(veil, int) or veil < 2:
            raise ValueError("flare.veil_grid must be an integer >= 2")
        if stage.get("model", "unet") != "uformer_flare7kpp":
            raise ValueError("flare.veil_grid follows the Uformer: set flare.model: uformer_flare7kpp")
    if "veil_only" in stage:
        if not isinstance(stage["veil_only"], bool):
            raise ValueError("flare.veil_only must be true or false")
        if stage["veil_only"] and veil is None:
            raise ValueError("flare.veil_only trains only the veil grid: set flare.veil_grid")
        if stage["veil_only"] and stage.get("uformer_init", "release") != "stage":
            raise ValueError("flare.veil_only keeps the Uformer as a previous stage left it: set flare.uformer_init: stage")
    if not config["data"].get("lens_flare_root"):
        raise ValueError("flare needs data.lens_flare_root: the folder holding Flare7K/ and Flare-R/ (Kaggle "
                         "khoidb/flare7k)")
    widths = stage["widths"]
    if (not isinstance(widths, (list, tuple)) or len(widths) < 2
            or any(isinstance(w, bool) or not isinstance(w, int) or w < 4 for w in widths)):
        raise ValueError("flare.widths must list at least two integers >= 4")
    for key, low in (("blocks", 1), ("batch_size", 1), ("max_successful_updates", 1), ("warmup_updates", 0),
                     ("validation_every_updates", 1), ("validation_composites", 1),
                     ("checkpoint_every_updates", 1), ("seed", 0)):
        value = stage[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"flare.{key} must be an integer >= {low}")
    for key in ("learning_rate", "gradient_clip_norm"):
        if not _nonnegative_number(stage[key]) or stage[key] == 0:
            raise ValueError(f"flare.{key} must be a positive number")
    for key in ("minimum_lr", "weight_decay"):
        if not _nonnegative_number(stage[key]):
            raise ValueError(f"flare.{key} must be a nonnegative number")
    if stage["precision"] not in ("fp32", "amp_fp16"):
        raise ValueError("flare.precision must be fp32 or amp_fp16")
    kinds = stage["kinds"]
    if (not isinstance(kinds, dict) or set(kinds) not in ({"composite", "pair", "plain"},
                                                          {"composite", "pair", "plain", "scene"})
            or not all(_nonnegative_number(v) for v in kinds.values()) or sum(kinds.values()) <= 0):
        raise ValueError("flare.kinds must give composite, pair and plain (and optionally scene) nonnegative weights, "
                         "not all 0")
    background_scene = stage.get("background_scene_fraction", 0.0)
    if not _nonnegative_number(background_scene) or background_scene > 1:
        raise ValueError("flare.background_scene_fraction must be in [0, 1]")
    uses_scenes = (float(kinds.get("scene", 0.0)) > 0 or float(background_scene) > 0
                   or stage.get("select_on") == "real_composite_scene")
    if uses_scenes and float(config["data"].get("scene_pairs_fraction", 0.0)) <= 0:
        raise ValueError("flare.kinds.scene, flare.background_scene_fraction and flare.select_on real_composite_scene "
                         "read the scene pairs: set data.scene_pairs_*")
    for key in ("background_pair_fraction", "blur_probability", "grain_probability"):
        if not _nonnegative_number(stage[key]) or stage[key] > 1:
            raise ValueError(f"flare.{key} must be in [0, 1]")
    for key, positive in (("blur_sigma_px", False), ("photon_count", True)):
        pair = stage[key]
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(_nonnegative_number(v) for v in pair)
                or pair[0] > pair[1] or (positive and pair[0] <= 0)):
            raise ValueError(f"flare.{key} must be [low, high] with 0 {'<' if positive else '<='} low <= high")
    if (float(kinds["pair"]) > 0 or float(stage["background_pair_fraction"]) > 0) and \
            float(config["data"].get("flare_pairs_fraction", 0.0)) <= 0:
        raise ValueError("flare.kinds.pair and flare.background_pair_fraction read FlareImage: set data.flare_pairs_*")
    if not isinstance(stage["lens"], dict):
        raise ValueError("flare.lens must map lens_* keys to values")
    from .data.flare_stage import lens_config
    lens_config(stage["lens"]).validate()                    # unknown keys or bad ranges fail here
    _require_wu_keys(stage["lens"], "flare.lens")


def _validate_exposure_standardize(config: dict[str, Any]) -> None:
    """p38: phase2.split_exposure_standardize and phase2.split_tone_gate_lr_scale. Absent: as before.

    Standardisation needs a head that reads the exposure statistics (split_exposure_stats or the tone gate); the
    gate's learning-rate scale needs the gate."""
    phase2 = config["phase2"]
    enabled = phase2.get("split_exposure_standardize", False)
    if not isinstance(enabled, bool):
        raise ValueError("phase2.split_exposure_standardize must be true or false")
    if enabled and (phase2.get("image_decoder") != "split_color_edge"
                    or not (phase2.get("split_exposure_stats", False) or phase2.get("split_tone_gate", False))):
        raise ValueError("phase2.split_exposure_standardize standardises the exposure statistics of the "
                         "split_color_edge decoder: set split_exposure_stats or split_tone_gate")
    if "split_tone_gate_lr_scale" in phase2:
        scale = phase2["split_tone_gate_lr_scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 100:
            raise ValueError("phase2.split_tone_gate_lr_scale must be a number in (0, 100]")
        if not phase2.get("split_tone_gate", False):
            raise ValueError("phase2.split_tone_gate_lr_scale scales the tone gate's learning rate: set "
                             "split_tone_gate: true")


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
            # p48: the clean-frame decoder, only with a positive weight (absent: I-JEPA alone).
            reconstruction_width=(int(phase1["ijepa_reconstruction_width"])
                                  if phase1.get("ijepa_reconstruction_weight", 0.0) > 0 else None),
            reconstruction_detail_weight=float(phase1.get("ijepa_reconstruction_detail_weight", 0.0)),
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
        # Absent before p46_nolatent: the image decoder reads ZI.
        image_latent=str(config["phase2"].get("image_latent", "zi")),
        # p47: RestoreRGBDecoder (phase2.image_decoder restore_rgb).
        restore=_restore_settings(config) if image_decoder == "restore_rgb" else None,
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
            # Absent before p30: the tone head and the light branch see no exposure statistics.
            "exposure_stats": bool(config["phase2"].get("split_exposure_stats", False)),
            # Absent before p31: no bilateral tone grid.
            "tone_grid": {
                "size": int(config["phase2"]["split_tone_grid_size"]),
                "bins": int(config["phase2"]["split_tone_grid_bins"]),
                "width": int(config["phase2"]["split_tone_grid_width"]),
            } if config["phase2"].get("split_tone_grid", False) else None,
            # Absent before p32: no stop map before the grid.
            "relight": {
                "size": int(config["phase2"]["split_relight_size"]),
                "widths": tuple(int(width) for width in config["phase2"]["split_relight_widths"]),
            } if config["phase2"].get("split_relight", False) else None,
            # Absent before p34: chroma only from the colour branch's grid.
            "chroma": {
                "width": int(config["phase2"]["split_chroma_width"]),
                "blocks": int(config["phase2"]["split_chroma_blocks"]),
            } if config["phase2"].get("split_chroma_detail", False) else None,
            # Absent before p35: every frame gets the whole tone stage.
            "tone_gate": {
                "width": int(config["phase2"]["split_tone_gate_width"]),
            } if config["phase2"].get("split_tone_gate", False) else None,
            # Absent before p38: the heads read the raw exposure statistics.
            "exposure_standardize": bool(config["phase2"].get("split_exposure_standardize", False)),
            # Absent before p41: no flare layer.
            "flare": {
                "widths": tuple(int(width) for width in config["phase2"]["split_flare_widths"]),
                "blocks": int(config["phase2"]["split_flare_blocks"]),
                # Absent before p42: the layer reads the latent (p41).
                "latent": bool(config["phase2"].get("split_flare_latent", True)),
                # Absent before p44: p42's U-Net layer.
                "model": config["phase2"].get("split_flare_model", "unet"),
                "scale": float(config["phase2"].get("split_flare_scale") or 1.0),
                **_veil_settings(config),
            } if config["phase2"].get("split_flare_layer", False) else None,
            "refiner": {
                "width": int(config["phase2"].get("split_edge_refiner_width", 32)),
                "blocks": int(config["phase2"].get("split_edge_refiner_blocks", 0)),
                "scale": int(config["phase2"].get("split_edge_refiner_scale", 1)),
            } if int(config["phase2"].get("split_edge_refiner_blocks", 0)) > 0 else None,
        },
    )


def _veil_settings(config: dict[str, Any]) -> dict[str, Any]:
    """p53: the flare stage's veil grid, when phase 2 loads that stage (absent otherwise: the layer as before)."""
    veil = config.get("flare", {}).get("veil_grid")
    if veil is None or not config["phase2"].get("split_flare_pretrained", False):
        return {}
    return {"veil_grid": int(veil)}


def _flare_settings(config: dict[str, Any]) -> dict[str, Any] | None:
    """The flare layer of phase2.split_flare_* (None when split_flare_layer is off)."""
    phase2 = config["phase2"]
    if not phase2.get("split_flare_layer", False):
        return None
    return {
        "widths": tuple(int(width) for width in phase2["split_flare_widths"]),
        "blocks": int(phase2["split_flare_blocks"]),
        "latent": bool(phase2.get("split_flare_latent", True)),
        "model": phase2.get("split_flare_model", "unet"),
        "scale": float(phase2.get("split_flare_scale") or 1.0),
        **_veil_settings(config),
    }


def _restore_settings(config: dict[str, Any]) -> dict[str, Any]:
    """RestoreRGBDecoder's arguments from phase2.restore_*, split_flare_* and motion_field (p47)."""
    phase2 = config["phase2"]
    field = phase2.get("motion_field")
    restormer = phase2.get("restore_net", "nafnet") == "restormer"
    return {
        "naf": None if restormer else {
            "widths": tuple(phase2["restore_widths"]), "enc_blocks": tuple(phase2["restore_enc_blocks"]),
            "middle_blocks": int(phase2["restore_middle_blocks"]), "dec_blocks": tuple(phase2["restore_dec_blocks"]),
            "fourier_levels": tuple(phase2["restore_fourier_levels"]),
            "fourier_width": int(phase2["restore_fourier_width"])},
        "restormer": None if not restormer else {
            "weights": str(phase2["restore_restormer_weights"]),
            **({"checkpointing": bool(phase2["restore_checkpointing"])} if "restore_checkpointing" in phase2 else {})},
        "train_crop": phase2.get("restore_train_crop"),
        "highpass_sigma": phase2.get("restore_highpass_sigma"),
        "light_grid": phase2.get("restore_light_grid"),
        "tone_width": int(phase2["restore_tone_width"]),
        "flare": _flare_settings(config),
        "motion": None if field is None else {
            "seconds": float(field["seconds"]), "samples": int(field["samples"]), "focal_px": float(field["focal_px"]),
            "scale_px": float(field["scale_px"]), "max_px": float(field["max_px"]),
            "reference_width": int(config["data"]["image_size"][1])},
    }


def build_corruptors(config: dict[str, Any]):
    image_values = config["corruption"]["image"]
    imu_values = config["corruption"]["imu"]
    image_cfg = LowLightImageCorruptionConfig(**image_values)
    imu_cfg = ImuCorruptionConfig(**imu_values)
    seed = config["data"]["corruption_seed"]
    halo = None
    if image_cfg.halo_probability > 0:
        halo = HaloFlareBank(config["data"]["halo_root"], revision=image_cfg.halo_revision,
                             effects=image_cfg.halo_effects, holdout_fraction=image_cfg.halo_holdout_fraction)
    lens = None
    if image_cfg.lens_flare_probability > 0:
        lens = LensFlareBank(config["data"]["lens_flare_root"], holdout_fraction=image_cfg.lens_flare_holdout_fraction,
                             wu_root=config["data"].get("wu_flare_root") if image_cfg.lens_flare_wu_fraction > 0 else None)
    return (LowLightImageCorruptor(image_cfg, seed, halo_bank=halo, lens_bank=lens),
            TrajectoryImuCorruptor(imu_cfg, seed))


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
