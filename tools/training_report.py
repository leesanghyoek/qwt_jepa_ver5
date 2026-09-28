"""Summarise a finished run's phase-1 and phase-2 logs as plain text.

Built to be pasted into another agent's prompt: no images, no colour, no
dependency beyond the standard library plus PyYAML, and every number labelled
with what it is and what it should be compared against.

    python3 tools/training_report.py                 # tự dò run mới nhất
    python3 tools/training_report.py --run outputs/p2_hilbert_imublur
    python3 tools/training_report.py > report.txt

It reads ``<run>/phase1/train.jsonl`` and ``<run>/phase2/train.jsonl``. Those
files mix three kinds of record -- per-update metrics, periodic validation
checkpoints, and a one-off initialisation reference -- so they are separated by
the keys each carries rather than by position, and a log truncated mid-run still
reports whatever it does contain.

Keys are discovered from the log rather than hardcoded, so a run written by an
older or newer revision still reports; anything unrecognised is listed at the
end instead of silently dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Metric -> (label, higher is better). Used for the beat-the-baseline verdicts.
IMAGE_METRICS = {
    "image_psnr_db": ("PSNR (dB)", True),
    "image_ssim": ("SSIM", True),
    "image_mae": ("MAE ảnh", False),
    "image_color_error": ("sai số màu", False),
}
IMU_METRICS = {
    "accel_rmse": ("accel RMSE (m/s²)", False),
    "gyro_rmse": ("gyro RMSE (rad/s)", False),
    "accel_variation_rmse": ("accel variation RMSE", False),
    "gyro_variation_rmse": ("gyro variation RMSE", False),
}
PHASE1_TERMS = ("loss", "jepa", "jepa_image", "jepa_imu", "jepa_image_fine", "jepa_image_coarse",
                "variance", "covariance",
                "reconstruction", "reconstruction_image", "reconstruction_image_detail",
                "reconstruction_imu", "reconstruction_imu_detail",
                "encoder_sensitivity", "gradient_norm")
PHASE2_TERMS = ("loss", "image_l1", "image_detail_l1", "image_detail_modulus_l1",
                "image_detail_energy", "image_detail_invisible_fraction",
                "image_color_l1", "image_color_stats_l1", "image_edge_detail_l1", "image_edge_gradient_l1",
                "image_edge_fft_l1", "image_edge_aux_l1", "image_perceptual",
                "image_edge_stage1_l1", "image_edge_roughness",
                "imu_accel_smooth_l1", "imu_gyro_smooth_l1", "imu_detail_l1",
                "imu_detail_energy", "imu_accel_variation_l1", "imu_gyro_variation_l1",
                "gradient_norm")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  [cảnh báo] {path}:{number} không phải JSON hợp lệ; bỏ qua",
                  file=sys.stderr)
    return records


def split_records(records: list[dict]):
    """Per-update steps, validation checkpoints, and the initialisation reference."""
    steps, checkpoints, reference = [], [], None
    for record in records:
        if record.get("event") == "initialization_reference":
            reference = record
        elif any(key.startswith("validation_") for key in record) or "latent_gate_status" in record:
            checkpoints.append(record)
        elif "successful_updates" in record:
            steps.append(record)
    return steps, checkpoints, reference


def window_median(steps: list[dict], key: str, at_start: bool, fraction: float = 0.05):
    """Median over the first or last slice, so one noisy update cannot set the number."""
    values = [r[key] for r in steps if isinstance(r.get(key), (int, float))
              and math.isfinite(r[key])]
    if not values:
        return None
    count = max(1, int(len(values) * fraction))
    return median(values[:count] if at_start else values[-count:])


def fmt(value, width: int = 12, places: int = 6) -> str:
    if value is None:
        return "—".rjust(width)
    if isinstance(value, float) and not math.isfinite(value):
        return "non-finite".rjust(width)
    magnitude = abs(value)
    if magnitude and (magnitude < 1e-4 or magnitude >= 1e6):
        return f"{value:.3e}".rjust(width)
    return f"{value:.{places}f}".rjust(width)


def change(first, last) -> str:
    """Percent change, but only where it means something.

    A log ratio crosses zero, and a percent change across a sign flip is noise
    dressed as a number -- report the absolute move instead.
    """
    if first is None or last is None or not math.isfinite(first):
        return "—".rjust(9)
    if first == 0 or (first < 0) != (last < 0):
        return f"{last - first:+.3f}".rjust(9)
    return f"{100.0 * (last / first - 1.0):+.1f}%".rjust(9)


def term_table(steps: list[dict], terms, title: str) -> None:
    present = [t for t in terms if any(t in r for r in steps)]
    if not present:
        return
    print(f"\n  {title}")
    print(f"  {'thành phần':<32}{'đầu':>12}{'thấp nhất':>14}{'cuối':>12}{'đổi':>10}")
    for term in present:
        values = [r[term] for r in steps
                  if isinstance(r.get(term), (int, float)) and math.isfinite(r[term])]
        first = window_median(steps, term, True)
        last = window_median(steps, term, False)
        print(f"  {term:<32}{fmt(first)}{fmt(min(values) if values else None, 14)}"
              f"{fmt(last)}{change(first, last)}")


def sensitivity_section(steps: list[dict]) -> list[str]:
    """The Jacobian term, split by branch: image and IMU behave nothing alike."""
    live = [r for r in steps if r.get("encoder_sensitivity_weight", 0) > 0]
    if not live:
        return ["Khối Jacobian không bao giờ bật (encoder_sensitivity_weight = 0 suốt run)."]
    print("\n  Độ nhạy encoder (Jacobian) — tách theo nhánh")
    print(f"  {'nhánh':<10}{'n':>7}{'g_nhiễu đầu':>14}{'g_nhiễu cuối':>14}"
          f"{'g_tín hiệu cuối':>17}{'tỉ số đầu':>12}{'tỉ số cuối':>12}")
    findings = []
    for branch in ("image", "imu"):
        rows = [r for r in live if r.get("encoder_source") == branch]
        if not rows:
            continue
        noise_first = window_median(rows, "sensitivity_noise_gain", True, 0.1)
        noise_last = window_median(rows, "sensitivity_noise_gain", False, 0.1)
        signal_last = window_median(rows, "sensitivity_signal_gain", False, 0.1)
        ratio_first = window_median(rows, "sensitivity_ratio", True, 0.1)
        ratio_last = window_median(rows, "sensitivity_ratio", False, 0.1)
        print(f"  {branch:<10}{len(rows):>7}{fmt(noise_first, 14)}{fmt(noise_last, 14)}"
              f"{fmt(signal_last, 17)}{fmt(ratio_first, 12, 4)}{fmt(ratio_last, 12, 4)}")
        if ratio_first is not None and ratio_last is not None and ratio_last > ratio_first * 1.1:
            findings.append(
                f"Nhánh {branch}: tỉ số độ nhạy TĂNG {ratio_first:.3f} -> {ratio_last:.3f}; "
                "encoder đang nhạy với nhiễu hơn lúc đầu, ngược mục tiêu của khối này.")
        if signal_last is not None and signal_last < 1e-3:
            findings.append(
                f"Nhánh {branch}: sensitivity_signal_gain cuối = {signal_last:.2e}, gần 0 — "
                "dấu hiệu biểu diễn co về hằng số (collapse).")
    valid = window_median(live, "sensitivity_valid_fraction", False, 0.1)
    if valid is not None and valid < 0.7:
        findings.append(f"sensitivity_valid_fraction cuối = {valid:.2f}: nhiều mẫu không có "
                        "hướng nhiễu để đo (clean_probability quá cao?).")
    return findings


def jepa_quality_section(steps: list[dict], checkpoints: list[dict], reference: dict | None) -> list[str]:
    """JEPA that can be compared across runs: normalized, cosine, visible vs masked."""
    rows = [("ảnh · JEPA chuẩn hoá", "jepa_image_normalized"), ("ảnh · cosine", "jepa_image_cosine"),
            ("IMU · JEPA chuẩn hoá", "jepa_imu_normalized"), ("IMU · cosine", "jepa_imu_cosine"),
            ("ảnh mịn 2× · chuẩn hoá", "jepa_image_fine_normalized")]
    checked = [c for c in checkpoints if "validation_jepa_image_normalized" in c]
    if not checked and not any("jepa_image_normalized" in r for r in steps):
        print("\n  (run này chưa log JEPA chuẩn hoá — code cũ hơn phương án A)")
        return []
    print("\n  Chất lượng JEPA — so được giữa các run")
    print("    JEPA chuẩn hoá = JEPA ÷ JEPA của cách đoán một token cố định (collapse):"
          "\n    1 = không hơn collapse, 0 = hoàn hảo. JEPA thô tăng khi latent mang nhiều hơn;"
          "\n    số này thì không. Validation không che token, nên so thẳng với run cũ.")
    print(f"    {'':<26}{'khởi tạo':>12}{'val cuối':>12}{'train 5% cuối':>16}")
    final = checked[-1] if checked else {}
    for label, key in rows:
        start = (reference or {}).get(f"validation_{key}")
        last = final.get(f"validation_{key}")
        train = window_median(steps, key, False)
        if start is None and last is None and train is None:
            continue
        print(f"    {label:<26}{fmt(start, 12, 4)}{fmt(last, 12, 4)}{fmt(train, 16, 4)}")
    split = [(branch, window_median(steps, f"jepa_{branch}_visible", False),
              window_median(steps, f"jepa_{branch}_masked", False)) for branch in ("image", "imu")]
    if any(visible is not None for _, visible, _ in split):
        print("    Train, token thấy vs token bị che (5% cuối; token che khó hơn, cao hơn là bình thường):")
        for branch, visible, masked in split:
            if visible is not None:
                print(f"      {branch:<6}thấy {fmt(visible, 10, 4)}   che {fmt(masked, 10, 4)}")
    findings = []
    image_last = final.get("validation_jepa_image_normalized")
    if isinstance(image_last, (int, float)) and image_last > 0.8:
        findings.append(f"JEPA ảnh chuẩn hoá cuối = {image_last:.2f}: predictor gần như không hơn "
                        "việc đoán một token cố định — latent nhiễu gần như không nói gì về đích sạch.")
    return findings


def latent_gate_section(checkpoints: list[dict], reference: dict | None) -> list[str]:
    gated = [c for c in checkpoints if "latent_gate_status" in c]
    if not gated:
        return []
    statuses = {}
    for check in gated:
        statuses[check["latent_gate_status"]] = statuses.get(check["latent_gate_status"], 0) + 1
    print("\n  Cổng latent (chống collapse): "
          + " · ".join(f"{name} ×{count}" for name, count in sorted(statuses.items())))
    findings = []
    failing = [c for c in gated if c["latent_gate_status"] != "PASS"]
    if failing:
        last = failing[-1]
        reasons = last.get("latent_gate_reasons") or []
        findings.append(
            f"Cổng latent không PASS ở {len(failing)}/{len(gated)} lần kiểm tra; lần cuối "
            f"(update {last.get('successful_updates')}): {'; '.join(map(str, reasons))[:400]}")
    if reference:
        keys = [k for k in reference
                if k.endswith(("same_position_std", "pooled_effective_rank", "raw_rms"))]
        if keys and gated:
            final = gated[-1]
            ratios = {}
            for key in sorted(keys):
                start, now = reference.get(key), final.get(key)
                if isinstance(start, (int, float)) and isinstance(now, (int, float)) and start:
                    ratios[key] = now / start
            if ratios:
                # Twenty-odd near-1.00 rows bury the two that matter, so only the
                # ones that actually moved are printed; the rest get one line.
                suspicious = {k: v for k, v in ratios.items() if v < 0.8 or v > 1.25}
                print("\n  Chẩn đoán collapse — tỉ lệ so với lúc khởi tạo (1.00 = không đổi)")
                print(f"    {len(ratios)} chỉ số · khoảng [{min(ratios.values()):.2f}, "
                      f"{max(ratios.values()):.2f}] · {len(suspicious)} nằm ngoài [0.80, 1.25]")
                for key, ratio in sorted(suspicious.items(), key=lambda kv: kv[1])[:12]:
                    short = key.replace("validation_", "").replace("_normalized", "~")
                    print(f"    {short:<44}x{ratio:.2f}")
                for key, ratio in ratios.items():
                    if ratio < 0.5:
                        findings.append(
                            f"{key.replace('validation_', '')} chỉ còn {ratio:.2f}× so với "
                            "khởi tạo — latent đang co lại.")
    return findings


def validation_section(checkpoints: list[dict], max_rows: int) -> list[str]:
    rows = [c for c in checkpoints if any(k.startswith("validation_image") for k in c)]
    if not rows:
        return ["Phase 2 không có bản ghi validation nào."]
    findings: list[str] = []
    columns = [(f"validation_{k}", label, higher)
               for k, (label, higher) in {**IMAGE_METRICS, **IMU_METRICS}.items()
               if f"validation_{k}" in rows[-1]]

    shown = rows if len(rows) <= max_rows else (
        rows[:1] + rows[1::max(1, len(rows) // (max_rows - 2))][:max_rows - 2] + rows[-1:])
    print("\n  Validation theo checkpoint")
    print(f"  {'update':>8}" + "".join(f"{label:>18}" for _, label, _ in columns))
    for row in shown:
        line = f"  {row.get('successful_updates', '?'):>8}"
        for key, _, _ in columns:
            line += fmt(row.get(key), 18, 4)
        print(line)

    print("\n  So với baseline (input chưa xử lý) tại checkpoint CUỐI")
    print(f"  {'chỉ số':<26}{'input':>14}{'khôi phục':>14}{'cải thiện':>12}   kết luận")
    final = rows[-1]
    for key, label, higher in columns:
        restored = final.get(key)
        baseline = final.get(key.replace("validation_", "validation_baseline_"))
        if not isinstance(restored, (int, float)) or not isinstance(baseline, (int, float)):
            continue
        better = restored > baseline if higher else restored < baseline
        delta = (restored - baseline) if higher else (baseline - restored)
        relative = f"{100.0 * delta / abs(baseline):+.1f}%" if baseline else "—"
        print(f"  {label:<26}{fmt(baseline, 14, 4)}{fmt(restored, 14, 4)}{relative:>12}   "
              f"{'VƯỢT' if better else 'THUA'}")
        if not better:
            findings.append(f"{label}: khôi phục ({restored:.4f}) KHÔNG tốt hơn input "
                            f"({baseline:.4f}) — model đang làm hỏng chỉ số này.")

    # PSNR/SSIM cannot tell real sharpening from a fixed high-frequency pattern.
    # Both ratios are power relative to the clean frame, so 1.00 = like clean.
    if isinstance(final.get("validation_image_stripe_power"), (int, float)):
        print("\n  Độ nét thật và sọc — năng lượng phổ so với ảnh clean (1.00 = như clean)")
        print(f"  {'':<34}{'input':>12}{'khôi phục':>12}")
        for key, label in (("image_edge_power", "cạnh/texture (chu kỳ 4-16 px) ↑"),
                           ("image_stripe_power", "sọc (chu kỳ 2 px) — ~1 là tốt")):
            print(f"  {label:<34}{fmt(final.get('validation_baseline_' + key), 12, 3)}"
                  f"{fmt(final.get('validation_' + key), 12, 3)}")
        # Power counts noise and invented texture; these count only detail at the right place.
        if isinstance(final.get("validation_image_fine_detail_in_place"), (int, float)):
            print("  — đúng chỗ (có tính pha: nhiễu và texture bịa không được tính) —")
            for key, label in (("image_edge_in_place", "đường nét 4-16 px đúng chỗ ↑"),
                               ("image_fine_detail_in_place", "vật nhỏ/xa 2-4 px đúng chỗ ↑")):
                print(f"  {label:<34}{fmt(final.get('validation_baseline_' + key), 12, 3)}"
                      f"{fmt(final.get('validation_' + key), 12, 3)}")
        if isinstance(final.get("validation_image_excess_roughness"), (int, float)):
            print("  — độ mượt (0 = mượt như ảnh sạch ở vùng phẳng) —")
            print(f"  {'gồ ghề thừa (/255) ↓':<34}{fmt(final.get('validation_baseline_image_excess_roughness'), 12, 3)}"
                  f"{fmt(final.get('validation_image_excess_roughness'), 12, 3)}")
        stripe = final["validation_image_stripe_power"]
        if stripe > 1.5:
            findings.append(
                f"Năng lượng sọc chu kỳ 2 px = {stripe:.2f}x ảnh clean — ảnh có sọc ngang/dọc/ô "
                "bàn cờ. Đó là dấu vân tay của một số hạng loss thưởng NĂNG LƯỢNG chi tiết mà "
                "không xét vị trí (modulus, detail_energy): một hệ số chi tiết lệch đều tổng hợp "
                "ra đúng sọc chu kỳ 2 px.")

    # Colour error cannot tell a washed-out frame from a colour cast; these can.
    if isinstance(final.get("validation_image_saturation_clean"), (int, float)):
        print("\n  Độ tươi màu và tương phản — so với ảnh clean (1.00 = như clean)")
        print(f"  {'':<34}{'input':>12}{'khôi phục':>12}")
        for key, label in (("image_saturation", "độ đậm màu (Cb, Cr)"),
                           ("image_contrast", "tương phản (độ lệch Y)")):
            clean_value = final.get(f"validation_{key}_clean") or 0.0
            ratio = lambda prefix: (final.get(f"{prefix}{key}") / clean_value
                                    if clean_value and isinstance(final.get(f"{prefix}{key}"), (int, float))
                                    else None)
            print(f"  {label:<34}{fmt(ratio('validation_baseline_'), 12, 3)}{fmt(ratio('validation_'), 12, 3)}")
            restored = ratio("validation_")
            if restored is not None and restored < 0.9:
                findings.append(
                    f"{label} của ảnh khôi phục chỉ bằng {restored:.2f}x ảnh clean — ảnh "
                    f"{'nhạt màu' if key == 'image_saturation' else 'phẳng, thiếu tương phản'}. "
                    "L1 từng pixel trả lời màu/độ sáng không chắc bằng trung vị, nghiêng về xám/phẳng.")

    # The variation metric is a rate (diff / 0.01 s), so scaling it back to a
    # per-sample difference makes it comparable with the absolute error. Near 1.0
    # the consecutive errors are uncorrelated -- jitter, not offset -- and no
    # per-sample loss term can see that.
    print("\n  Độ rung IMU — sai số giữa 2 mẫu liền kề / sai số tổng")
    print(f"  {'':<10}{'input':>20}{'khôi phục':>20}   (càng thấp càng mượt)")
    for axis in ("accel", "gyro"):
        rows_ok = True
        cells = []
        for prefix in ("validation_baseline_", "validation_"):
            rmse = final.get(f"{prefix}{axis}_rmse")
            rate = final.get(f"{prefix}{axis}_variation_rmse")
            if not isinstance(rmse, (int, float)) or not isinstance(rate, (int, float)) or not rmse:
                rows_ok = False
                break
            cells.append(rate * 0.01 / rmse)
        if not rows_ok:
            continue
        print(f"  {axis:<10}{cells[0]:>20.2f}{cells[1]:>20.2f}")
        if cells[1] > 0.6:
            findings.append(
                f"{axis}: sai số giữa hai mẫu liền kề còn {cells[1]:.2f}x sai số tổng "
                "— lỗi gần như toàn bộ là RUNG, không phải lệch. imu_variation_weight là "
                "số hạng duy nhất nhìn thấy điều này.")

    score_key = "validation_joint_validation_score"
    scored = [r for r in rows if isinstance(r.get(score_key), (int, float))]
    if scored:
        best = min(scored, key=lambda r: r[score_key])
        print(f"\n  Checkpoint tốt nhất (joint_validation_score, thấp hơn là tốt): "
              f"update {best.get('successful_updates')} = {best[score_key]:.6f}"
              f"   ·   cuối cùng = {scored[-1][score_key]:.6f}")
        position = scored.index(best) / max(1, len(scored) - 1)
        if position < 0.6 and len(scored) >= 4:
            findings.append(
                f"Điểm tốt nhất rơi ở {position:.0%} quãng train rồi xấu đi — train thừa; "
                "dùng best_joint_validation.pt, và cân nhắc giảm số update.")

    ssim = [r.get("validation_image_ssim") for r in rows
            if isinstance(r.get("validation_image_ssim"), (int, float))]
    if len(ssim) >= 4:
        tail = ssim[-4:]
        if max(tail) - min(tail) < 0.002:
            findings.append(
                f"SSIM đứng yên ở 4 checkpoint cuối ({min(tail):.4f}–{max(tail):.4f}) — "
                "đã bão hoà; thêm update sẽ không giúp gì.")
    return findings


def invisible_detail_finding(steps: list[dict]) -> list[str]:
    # QWT coefficients are 4x redundant and synthesis averages the trees, so a
    # decoder can park detail energy where the image never shows it. Input
    # coefficients sit at ~0; any real share means the detail terms, if scored
    # on the decoder output, are being lowered without the image changing.
    last = window_median(steps, "image_detail_invisible_fraction", at_start=False)
    if not isinstance(last, (int, float)) or last < 0.05:
        return []
    return [f"{last:.0%} năng lượng chi tiết decoder xuất ra KHÔNG hiện lên ảnh (nằm trong "
            "null space của synthesis). Nếu phase2.image_detail_source = decoder_coefficients, "
            "các số hạng chi tiết ảnh đang giảm mà ảnh không nét hơn — dùng restored_image."]


def phase_header(name: str, steps: list[dict], checkpoints: list[dict]) -> list[str]:
    findings: list[str] = []
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    if not steps and not checkpoints:
        print("  (không có log)")
        return [f"{name}: không tìm thấy bản ghi nào."]
    updates = [r["successful_updates"] for r in steps if "successful_updates" in r]
    print(f"  bản ghi update: {len(steps)}   ·   checkpoint: {len(checkpoints)}"
          f"   ·   update cuối: {max(updates) if updates else '?'}")
    skipped = [r for r in steps if r.get("skipped")]
    if skipped:
        findings.append(f"{name}: {len(skipped)} update bị BỎ QUA "
                        f"(lý do đầu tiên: {skipped[0].get('reason', '?')}).")
    nonfinite = [r for r in steps if isinstance(r.get("loss"), (int, float))
                 and not math.isfinite(r["loss"])]
    if nonfinite:
        findings.append(f"{name}: {len(nonfinite)} update có loss non-finite.")
    lr = [r.get("learning_rate") for r in steps if isinstance(r.get("learning_rate"), (int, float))]
    if lr:
        print(f"  learning rate: {lr[0]:.3e} -> {lr[-1]:.3e}")
    return findings


def config_section(run: Path) -> None:
    print(f"\n{'=' * 78}\nCẤU HÌNH\n{'=' * 78}")
    candidates = [run / "config.yaml", run / "phase1/resolved_config.yaml",
                  run / "phase2/resolved_config.yaml"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None or yaml is None:
        print("  (không tìm thấy resolved_config.yaml hoặc thiếu PyYAML)")
        return
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    print(f"  nguồn: {path}")
    interesting = [
        ("model.image_transform", ("model", "image_transform")),
        ("corruption.image.motion_from_imu", ("corruption", "image", "motion_from_imu")),
        ("phase1.max_successful_updates", ("phase1", "max_successful_updates")),
        ("phase1.batch_size", ("phase1", "batch_size")),
        ("phase1.covariance_pooling", ("phase1", "covariance_pooling")),
        ("phase1.image_positions_per_update", ("phase1", "image_positions_per_update")),
        ("phase1.teacher_momentum_end", ("phase1", "teacher_momentum_end")),
        ("phase1.predictor_type", ("phase1", "predictor_type")),
        ("phase1.image_mask_ratio", ("phase1", "image_mask_ratio")),
        ("phase1.imu_mask_ratio", ("phase1", "imu_mask_ratio")),
        ("phase1.multiscale_fine_weight", ("phase1", "multiscale_fine_weight")),
        ("phase1.multiscale_coarse_weight", ("phase1", "multiscale_coarse_weight")),
        ("phase1.coefficient_reconstruction_loss_weight",
         ("phase1", "coefficient_reconstruction_loss_weight")),
        ("encoder_sensitivity.weight_max", ("encoder_sensitivity", "weight_max")),
        ("phase2.max_successful_updates", ("phase2", "max_successful_updates")),
        ("phase2.encoder_skips", ("phase2", "encoder_skips")),
        ("phase2.reconstruction_detail_weight", ("phase2", "reconstruction_detail_weight")),
        ("phase2.detail_energy_weight", ("phase2", "detail_energy_weight")),
        ("phase2.imu_variation_weight", ("phase2", "imu_variation_weight")),
        ("phase2.image_detail_loss", ("phase2", "image_detail_loss")),
        ("phase2.image_detail_source", ("phase2", "image_detail_source")),
        ("phase2.image_decoder", ("phase2", "image_decoder")),
        ("phase2.split_branch_arch", ("phase2", "split_branch_arch")),
        ("phase2.split_edge_unet_widths", ("phase2", "split_edge_unet_widths")),
        ("phase2.split_color_global", ("phase2", "split_color_global")),
        ("phase2.split_color_stats_weight", ("phase2", "split_color_stats_weight")),
        ("phase2.split_edge_naf_widths", ("phase2", "split_edge_naf_widths")),
        ("phase2.split_edge_fft_weight", ("phase2", "split_edge_fft_weight")),
        ("phase2.split_edge_aux_weight", ("phase2", "split_edge_aux_weight")),
        ("phase2.perceptual_weight", ("phase2", "perceptual_weight")),
        ("phase2.perceptual_crop", ("phase2", "perceptual_crop")),
        ("phase2.precision", ("phase2", "precision")),
        ("phase2.split_edge_refiner_blocks", ("phase2", "split_edge_refiner_blocks")),
        ("phase2.split_edge_smooth_weight", ("phase2", "split_edge_smooth_weight")),
        ("corruption.image.photon_count", ("corruption", "image", "photon_count")),
        ("corruption.image.noise_only_probability", ("corruption", "image", "noise_only_probability")),
        ("phase2.smooth_l1_beta", ("phase2", "smooth_l1_beta")),
        ("phase2.residual_sees_input", ("phase2", "residual_sees_input")),
    ]
    for label, keys in interesting:
        node = config
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        print(f"  {label:<48}{node}")


def find_run(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    roots = [Path.cwd() / "outputs", Path(__file__).resolve().parent.parent / "outputs"]
    runs = []
    for root in roots:
        if root.is_dir():
            runs.extend(p.parent for p in root.glob("*/phase*/train.jsonl"))
    if not runs:
        raise SystemExit("Không tự dò được run nào. Truyền --run <thư mục chứa phase1/ và phase2/>")
    best = max(set(runs), key=lambda p: max(
        (f.stat().st_mtime for f in p.glob("phase*/train.jsonl")), default=0))
    print(f"Tự dò ra run: {best}   (đặt --run để chọn khác)")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None,
                        help="thư mục chứa phase1/ và phase2/; mặc định: run mới nhất trong outputs/")
    parser.add_argument("--max-rows", type=int, default=14,
                        help="số dòng tối đa của bảng validation")
    args = parser.parse_args()
    run = find_run(args.run)

    print(f"""
BÁO CÁO TRAIN — {run}

BỐI CẢNH (cho người/agent đọc báo cáo này mà chưa biết dự án)
  Pipeline hai giai đoạn khôi phục ảnh RGB 256x256 thiếu sáng/mờ và IMU 6 kênh
  nhiễu, dữ liệu TartanAir V2.
    Phase 1 — học latent. KHÔNG khôi phục ảnh. Encoder đọc đầu vào NHIỄU và học
      dự đoán latent mà một teacher EMA tạo ra từ đầu vào SẠCH (JEPA), cộng
      variance/covariance chống collapse, một decoder "neo" chấm điểm trên hệ số
      wavelet sạch rồi bị vứt, và một số hạng độ nhạy encoder (Jacobian).
    Phase 2 — khôi phục. Backbone ĐÓNG BĂNG. Chỉ hai decoder mới được train; chúng
      dự đoán HIỆU CHỈNH cộng vào hệ số wavelet của chính đầu vào nhiễu.
  Chiều tốt: PSNR/SSIM cao hơn là tốt; mọi *MAE/RMSE/loss thấp hơn là tốt.
  "baseline" = chính đầu vào chưa xử lý, tức "không làm gì". Vượt baseline mới là
  có đóng góp; thua baseline nghĩa là model làm hỏng chỉ số đó.
  Trần đã biết của dự án: latent ZI là 128x16x16 (mỗi ô mô tả khối 16x16 pixel),
  và mọi số hạng loss phase 2 đều là sai số trung bình nên nghiệm tối ưu của chúng
  là ảnh mờ — đừng kỳ vọng SSIM tăng mạnh chỉ nhờ chỉnh trọng số.""")
    config_section(run)

    findings: list[str] = []
    phase1_steps, phase1_checks, reference = split_records(read_jsonl(run / "phase1/train.jsonl"))
    findings += phase_header("PHASE 1 — học latent", phase1_steps, phase1_checks)
    if phase1_steps:
        term_table(phase1_steps, PHASE1_TERMS, "Thành phần loss (trung vị 5% đầu và 5% cuối)")
        findings += sensitivity_section(phase1_steps)
    findings += jepa_quality_section(phase1_steps, phase1_checks, reference)
    findings += latent_gate_section(phase1_checks, reference)

    phase2_steps, phase2_checks, _ = split_records(read_jsonl(run / "phase2/train.jsonl"))
    findings += phase_header("PHASE 2 — khôi phục", phase2_steps, phase2_checks)
    if phase2_steps:
        term_table(phase2_steps, PHASE2_TERMS, "Thành phần loss (trung vị 5% đầu và 5% cuối)")
        overflows = [r["amp_overflow"] for r in phase2_steps if "amp_overflow" in r]
        if overflows:
            share = sum(overflows) / len(overflows)
            print(f"\n  fp16: {sum(overflows)}/{len(overflows)} update tràn số (bị bỏ bước tối ưu), "
                  f"scale cuối {phase2_steps[-1].get('amp_scale', float('nan')):.0f}")
            if share > 0.05:
                findings.append(f"fp16 tràn số ở {100 * share:.1f}% update — nhiều hơn mức bình thường "
                                "(vài update đầu); cân nhắc precision: fp32.")
        findings += invisible_detail_finding(phase2_steps)
    if phase2_checks:
        findings += validation_section(phase2_checks, args.max_rows)

    known = set(PHASE1_TERMS) | set(PHASE2_TERMS) | {
        "successful_updates", "skipped", "reason", "learning_rate", "teacher_momentum",
        "encoder_source", "encoder_sensitivity_weight", "probe_clipped_fraction",
        "sensitivity_noise_gain", "sensitivity_signal_gain", "sensitivity_ratio",
        "sensitivity_valid_fraction", "jepa_image_normalized", "jepa_imu_normalized",
        "amp_overflow", "amp_scale",
        "jepa_image_cosine", "jepa_imu_cosine", "jepa_image_visible", "jepa_image_masked",
        "jepa_imu_visible", "jepa_imu_masked", "jepa_image_fine_normalized",
    }
    extra = sorted({k for r in phase1_steps + phase2_steps for k in r} - known)
    if extra:
        print(f"\n  Khoá khác có trong log (không được tóm tắt ở trên): {', '.join(extra)}")

    print(f"\n{'=' * 78}\nCẦN CHÚ Ý\n{'=' * 78}")
    if findings:
        for number, finding in enumerate(findings, 1):
            print(f"  {number}. {finding}")
    else:
        print("  Không có dấu hiệu bất thường nào trong các phép kiểm tra tự động.")
    print("\n  Các phép kiểm tra đã chạy: update bị bỏ qua · loss non-finite · cổng latent ·"
          "\n  co biểu diễn (rank/std/rms so với khởi tạo) · tỉ số độ nhạy theo nhánh ·"
          "\n  từng chỉ số validation so với baseline · vị trí checkpoint tốt nhất · bão hoà SSIM.")
    print("\n  Lưu ý khi đọc: baseline = chính đầu vào chưa xử lý. Một chỉ số THUA baseline"
          "\n  nghĩa là model làm hỏng chỉ số đó, không phải chỉ là cải thiện ít.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
