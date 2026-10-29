"""Architecture figure of the run the notebook trains (configs/kaggle_uformer.yaml, p44_uformer;
qwt-jaco-jepa-ijepa.ipynb), as plain SVG.

Drawn from the config: phase 1 is I-JEPA alone -- ONE ViT over the image's and the IMU's wavelet
coefficients (attention between the two kinds of token is the fusion, ZI = FI, ZU = FU), the context
encoder on the NOISY pair's context tokens, one narrow predictor, the EMA teacher on the CLEAN pair; the
mask drawn is a real one (multiblock_masks with the config's settings). Phase 2 freezes the encoder and
reads its output ZI, ZU from every token of the noisy pair. With phase2.split_tone_grid (p31) the noisy
frame first goes through the tone stage -- a coefficient net (frame at 64 x 64, ZI, exposure statistics)
predicts a bilateral grid of 3 x 4 colour affines, sliced per pixel at (x, y, guide) and applied to the
pixel itself -- and its output J feeds the colour branch and the edge branch; without it the stage is the
light branch (veil V, log gain g). With phase2.split_relight (p32) a U-Net first predicts the stop map S
(frame at 64 x 64, ZI, exposure statistics; learned only from the corruption's true map) and the frame is
divided by 2^S in linear light before the grid, which the panel then draws as one box. The loss box lists
the terms the config turns on.

Later keys, each drawn only when the config turns it on: the tone gate (p35, phase2.split_tone_gate), full-resolution
chroma detail (p34), FlareImage pairs in every batch (p36+, data.flare_pairs_fraction), the flare layer subtracted
before the tone stage (p41, phase2.split_flare_layer) and, with a flare section (p42), the flare stage that trains it
alone (band 4, train-flare) before phase 2 loads and freezes it. With flare.model uformer_flare7kpp (p44) that layer is
Flare7K++'s Uformer, fine-tuned from its released weights and kept at its best validation (flare/best.pt); with
phase2.kind_balance the loss box says every kind of frame weighs the same.

p47+ (phase2.image_decoder restore_rgb) draws phase 2 as the decoder it is: the frame without its flare -> one tone
curve and colour matrix per frame (GlobalToneColor) -> one RGB restoration net, which also reads the frame without its
flare and the gyro's path during the exposure (phase2.motion_field) and adds its answer to the toned frame; with
phase2.restore_net restormer (p51) that net is the released Restormer drawn level by level, its two zero-initialised
layers (extra_embed, latent_in) marked. With phase1.ijepa_reconstruction_weight (p48) phase 1 gains the reconstruction
head that redraws the clean frame from ZI. Band 4 counts drone pairs (flare.kinds scene, p46) when the config has them.
p52 (phase2.restore_light_grid / restore_highpass_sigma / restore_train_crop): the light grid after the tone curve, the
Restormer's answer cut to detail only, and the window it trains on. p53 (flare.veil_grid / veil_only): the veil grid
after the Uformer, in the flare layer (3) and in band 4, where only the grid trains.

Usage: python3 tools/draw_architecture_ijepa.py docs/kien_truc_ijepa.svg [config]
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.config import load_config  # noqa: E402
from qjepa.models.ijepa import multiblock_masks  # noqa: E402

CONFIG_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "configs/kaggle_veil.yaml"
CONFIG = load_config(CONFIG_PATH)
P1, P2, MODEL = CONFIG["phase1"], CONFIG["phase2"], CONFIG["model"]
RUN = Path(CONFIG["runtime"]["output_dir"]).name
GRID = bool(P2.get("split_tone_grid", False))
RELIGHT = bool(P2.get("split_relight", False))
EXPOSURE = bool(P2.get("split_exposure_stats", False))
LOWFREQ = float(P2.get("lowfreq_mse_weight", 0.0))
GATE = bool(P2.get("split_tone_gate", False))
CHROMA = bool(P2.get("split_chroma_detail", False))
PAIRS = float(CONFIG["data"].get("flare_pairs_fraction", 0.0))
FLARE = bool(P2.get("split_flare_layer", False))
STAGE = CONFIG.get("flare") if FLARE and P2.get("split_flare_pretrained", False) else None
UFORMER = P2.get("split_flare_model", "unet") == "uformer_flare7kpp"
KIND_BALANCE = bool(P2.get("kind_balance", False))
RESTORE = P2.get("image_decoder") == "restore_rgb"                 # p47+
RESTORMER = RESTORE and P2.get("restore_net") == "restormer"      # p51
FIELD = P2.get("motion_field") if RESTORE else None
LIGHT_GRID = P2.get("restore_light_grid") if RESTORMER else None    # p52
HIGHPASS = P2.get("restore_highpass_sigma") if RESTORMER else None
CROP = P2.get("restore_train_crop") if RESTORMER else None
VEIL = STAGE.get("veil_grid") if STAGE and UFORMER else None        # p53
VEIL_ONLY = bool(STAGE.get("veil_only", False)) if VEIL else False
SCENES = float(CONFIG["data"].get("scene_pairs_fraction", 0.0))
NO_IMU = float(CONFIG["data"].get("imu_absent_probability", 0.0))
RECON = float(P1.get("ijepa_reconstruction_weight", 0.0)) > 0    # p48
DY = 40 if FLARE else 0                  # room above phase 2 for the flare layer on the input line
BAND = 190 if STAGE else 0               # band 4: the flare stage
RH = 80 if RECON else 0                  # phase 1's row for the reconstruction head
# A typical draw (context 40% of the tokens; median 42%, p10-p90 33-51% over 400 draws) whose four
# target blocks barely overlap (6%; median 32%), so each block can be seen.
MASK_SEED = 382

W, H = 1520, 790 + DY + BAND + RH
FONT = "Segoe UI, Roboto, 'Noto Sans', Helvetica, Arial, sans-serif"
C = {  # fill, stroke
    'data':   ('#F5F5F5', '#616161'),
    'tf':     ('#E8EAF6', '#3949AB'),
    'bb':     ('#E3F2FD', '#1E88E5'),
    'lat':    ('#F3E5F5', '#8E24AA'),
    'p1':     ('#FFF8E1', '#F9A825'),
    'color':  ('#FFF3E0', '#EF6C00'),
    'edge':   ('#E8F5E9', '#2E7D32'),
    'out':    ('#E0F2F1', '#00897B'),
    'loss':   ('#FCE4EC', '#D81B60'),
    'flare':  ('#FFEBEE', '#C62828'),
}
el = []


def text(x, y, s, size=15, weight='normal', color='#1A1A1A', anchor='middle', style=''):
    el.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
              f'text-anchor="{anchor}" {style}>{escape(s)}</text>')


def node(x, y, w, h, kind, title, sub=(), rx=10, title_size=16):
    fill, stroke = C[kind]
    el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    lines = [title] + list(sub)
    total = 20 + 17 * (len(lines) - 1)
    ty = y + h / 2 - total / 2 + 15
    text(x + w / 2, ty, title, size=title_size, weight='bold')
    for i, s in enumerate(sub):
        text(x + w / 2, ty + 20 + 17 * i, s, size=13, color='#37474F')
    return (x, y, w, h)


def region(x, y, w, h, kind, label, dash='7 5', right=False):
    fill, stroke = C[kind]
    el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" fill-opacity="0.35" '
              f'stroke="{stroke}" stroke-width="2" stroke-dasharray="{dash}"/>')
    text(x + w - 14 if right else x + 14, y + 22, label, size=15, weight='bold', color=stroke,
         anchor='end' if right else 'start')


def arrow(points, color='#455A64', width=2.2, dash=None, label=None, lx=None, ly=None, lsize=13, lanchor='middle'):
    d = 'M ' + ' L '.join(f'{px} {py}' for px, py in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    marker = 'url(#ah)' if color == '#455A64' else f'url(#ah-{color[1:]})'
    el.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{extra} marker-end="{marker}"/>')
    if label:
        text(lx, ly, label, size=lsize, color=color, anchor=lanchor, style='font-style="italic"')


def line(points, color, width=2.4):
    el.append(f'<path d="M ' + ' L '.join(f'{px} {py}' for px, py in points) +
              f'" fill="none" stroke="{color}" stroke-width="{width}"/>')


def badge(x, y, label, color='#D81B60'):
    width = 12 + 7 * len(label)
    el.append(f'<rect x="{x}" y="{y}" width="{width}" height="18" rx="9" fill="{color}"/>')
    text(x + width / 2, y + 13, label, size=11, weight='bold', color='#FFFFFF')


def level(x, y, top, bottom, w=66, h=30):
    """One level of the restoration net: a small box, size and width on top, blocks below."""
    fill, stroke = C['edge']
    el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    text(x + w / 2, y + 13, top, size=10, weight='bold', color=stroke)
    text(x + w / 2, y + 25, bottom, size=10, color='#37474F')
    return (x, y, w, h)


def dot(x, y, color):
    el.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')


def mid_right(b): x, y, w, h = b; return (x + w, y + h / 2)
def mid_left(b):  x, y, w, h = b; return (x, y + h / 2)
def mid_top(b):   x, y, w, h = b; return (x + w / 2, y)
def mid_bot(b):   x, y, w, h = b; return (x + w / 2, y + h)


# ---------------- header
if RESTORE:
    text(20, 28, f'{RUN} — kiến trúc đang train · phase 1 I-JEPA{" + tái tạo ảnh sạch" if RECON else ""} (ViT trên hệ số'
         f' wavelet) · phase 2: {"khử lóe + màn sáng → " if VEIL else "khử lóe → " if FLARE else ""}'
         f'{"tone + lưới ánh sáng → " if LIGHT_GRID else "tone cả ảnh → "}'
         f'{"Restormer train sẵn chỉ thêm chi tiết" if HIGHPASS else "Restormer train sẵn, fine-tune" if RESTORMER else "NAFNet RGB"}'
         f'{" · ④ giai đoạn lóe" if STAGE else ""}',
         size=16, weight='bold', color='#D81B60', anchor='start')
else:
    text(20, 28, f'{RUN} — kiến trúc đang train · phase 1 I-JEPA (backbone ViT trên hệ số wavelet) · phase 2 khôi phục'
         f'{" với nhánh khử lóe + " if FLARE else " với "}'
         f'{"bản đồ stop + lưới song phương" if RELIGHT else "tầng tone lưới song phương" if GRID else "nhánh ánh sáng"}'
         f'{" · ④ giai đoạn lóe" if STAGE else ""}',
         size=16, weight='bold', color='#D81B60', anchor='start')
text(20, 48, 'MỘT ViT chung cho ảnh và IMU (attention giữa hai loại token là fusion) · ngữ cảnh NHIỄU, teacher EMA SẠCH'
     ' · khôi phục đọc ĐẦU RA CONTEXT ENCODER (đóng băng, mọi token nhiễu) · teacher + predictor bỏ sau phase 1',
     size=12, color='#AD1457', anchor='start')

el.append(f'<g transform="translate(0,{DY})">')
# ---------------- regions
region(190, 150, 490, 320, 'bb', '① Context encoder — học ở phase 1, đóng băng ở phase 2')
region(842, 88, 663, 432, 'edge', '③ Phase 2 — khôi phục' if RESTORE else '③ Phase 2 — khôi phục (backbone đóng băng)',
       right=RELIGHT)   # ảnh nhiễu rơi ở x 1113
OY = 60                                                   # phase 1 sits below phase 2's taller region
region(190, 505 + OY, 940, 200 + RH, 'p1', '② Phase 1 — I-JEPA')

# ---------------- inputs and backbone
if SCENES:
    img_in = node(20, 192, 150, 96, 'data', 'Ảnh vào', ['3 × 256 × 256', 'mờ (gyro), tối, hạt',
                                                        '+ FlareImage, drone',
                                                        f'{NO_IMU:.0%} không IMU' if NO_IMU else ''])
elif PAIRS:
    img_in = node(20, 192, 150, 96, 'data', 'Ảnh vào', ['3 × 256 × 256', 'mờ, tối, hạt', f'+ {PAIRS:g} batch'.replace('.', ','),
                                                        'là ảnh lóe thật'])
else:
    img_in = node(20, 200, 150, 80, 'data', 'Ảnh mờ + tối', ['3 × 256 × 256', 'nhiễu, tối theo vùng'])
imu_in = node(20, 365, 150, 70, 'data', 'IMU nhiễu', ['6 × 128'])
qwt = node(210, 205, 135, 70, 'tf', 'QWT Hilbert', ['kênh sáng Y', '16 × 128 × 128'])
haar = node(210, 365, 135, 70, 'tf', 'Haar', ['12 × 64'])
depth, heads, dim = MODEL["vit_depth"], MODEL["vit_heads"], MODEL["embedding_dim"]
scaled = {'global': 'hệ số × g (mỗi loại 1 g)', 'channel': 'hệ số × g (mỗi kênh 1 g)'}.get(
    MODEL.get('vit_input_standardize'), '')
vit = node(370, 195, 200, 250, 'bb', 'CONTEXT ENCODER', ['ViT chung ảnh + IMU', scaled,
                                                        'ảnh: patch 8 = 16 px', '→ 256 token', '',
                                                        'IMU: patch 8 = 16 mẫu', '→ 8 token', '',
                                                        f'264 token × {dim}', f'{depth} khối · {heads} head',
                                                        '+ loại token + vị trí'])
text(637, 305, 'attention', size=12, weight='bold', color='#1E88E5')
text(637, 320, 'ảnh ↔ IMU', size=12, weight='bold', color='#1E88E5')
text(637, 335, '= fusion', size=12, weight='bold', color='#1E88E5')
zi = node(705, 205, 110, 70, 'lat', 'ZI = FI', [f'{dim} × 16 × 16'])
zu = node(705, 365, 110, 70, 'lat', 'ZU = FU', [f'{dim} × 8'])
text(760, 303, 'ĐẦU RA ENCODER', size=11, weight='bold', color='#8E24AA')
text(760, 318, 'mọi token nhiễu', size=11, color='#8E24AA')
text(760, 333, '→ khôi phục ③', size=11, weight='bold', color='#8E24AA')
el.append('<circle cx="190" cy="240" r="13" fill="#E8EAF6" stroke="#3949AB" stroke-width="2"/>')
text(190, 245, 'Y', size=14, weight='bold', color='#3949AB')
text(190, 196, 'tách màu', size=11, weight='bold', color='#3949AB')
arrow([mid_right(img_in), mid_left(qwt)]); arrow([mid_right(imu_in), mid_left(haar)])
arrow([mid_right(qwt), (370, 240)]); arrow([mid_right(haar), (370, 400)])
arrow([(570, 240), mid_left(zi)]); arrow([(570, 400), mid_left(zu)])

# ---------------- phase 2
if not RESTORE:
    # the split decoder's tone stage
    el.append('<rect x="856" y="116" width="636" height="110" rx="10" fill="#FFFFFF" fill-opacity="0.8" '
              'stroke="#EF6C00" stroke-width="1.5"/>')
    if RELIGHT:
        cells, bins, size = P2['split_tone_grid_size'], P2['split_tone_grid_bins'], P2['split_relight_size']
        text(868, 136, 'TẦNG TONE', size=14, weight='bold', color='#EF6C00', anchor='start')   # ảnh nhiễu rơi vào x 1113
        if GATE:
            text(1132, 137, 'cổng g/ảnh: J = vào + g·(tone − vào)', size=12, weight='bold', color='#EF6C00',
                 anchor='start')
        else:
            badge(1236, 123, 'mới · p32: bản đồ stop')
        t1 = node(866, 146, 165, 68, 'color', 'Bản đồ stop S', [f'U-Net {size}² · ảnh + ZI',
                                                               '+ thống kê phơi sáng' if EXPOSURE else ''], title_size=14)
        t2 = node(1046, 146, 135, 68, 'color', 'Làm sáng lại', ['ảnh · 2^(−S)', 'ánh sáng tuyến tính'], title_size=14)
        t3 = node(1196, 146, 140, 68, 'color', 'Lưới song phương', [f'{cells}×{cells}×{bins} ô · A 3×4',
                                                                  'cắt ở (x, y, sáng)'], title_size=14)
        t4 = node(1351, 146, 133, 68, 'color', "J = A·[ảnh′, 1]", ['màu, đường cong', 'đầu: = đồng nhất'],
                  title_size=14)
    elif GRID:
        cells, bins = P2['split_tone_grid_size'], P2['split_tone_grid_bins']
        text(868, 136, 'TẦNG TONE · LƯỚI SONG PHƯƠNG (HDRNet)', size=14, weight='bold', color='#EF6C00',
             anchor='start')
        badge(1180, 123, 'mới · p31')
        t1 = node(866, 146, 165, 68, 'color', 'Mạng hệ số', ['ảnh 64² + ZI',
                                                            '+ thống kê phơi sáng' if EXPOSURE else ''], title_size=14)
        t2 = node(1046, 146, 135, 68, 'color', 'Lưới hệ số', [f'{cells}×{cells}×{bins} ô', 'mỗi ô: A 3×4'], title_size=14)
        t3 = node(1196, 146, 140, 68, 'color', 'Cắt lưới', ['tại (x, y, độ sáng)', 'bản đồ dẫn'], title_size=14)
        t4 = node(1351, 146, 133, 68, 'color', 'J = A·[ảnh, 1]', ['áp lên chính pixel', 'đầu: = đồng nhất'],
                  title_size=14)
    else:
        text(868, 136, 'NHÁNH ÁNH SÁNG · U-Net 64²', size=14, weight='bold', color='#EF6C00', anchor='start')
        t1 = node(866, 146, 165, 68, 'color', 'U-Net 64²', ['ảnh + ZI', '+ thống kê phơi sáng' if EXPOSURE else ''],
                  title_size=14)
        t2 = node(1046, 146, 135, 68, 'color', 'Sương V, sáng g', ['bản đồ trơn'], title_size=14)
        t3 = node(1196, 146, 140, 68, 'color', 'Phóng lên', ['song tuyến'], title_size=14)
        t4 = node(1351, 146, 133, 68, 'color', 'J = (ảnh − V)·eᵍ', ['ánh sáng tuyến tính'], title_size=14)
    for left, right in ((t1, t2), (t2, t3), (t3, t4)):
        arrow([mid_right(left), mid_left(right)], color='#EF6C00', width=2)
# The noisy frame: into the coefficient net, the guide and the affine.
line([(95, 192 if PAIRS or SCENES else 200), (95, 76), (1378 if RESTORE else 1113 if RELIGHT else 1418, 76)], '#2E7D32',
     width=3)
if FLARE:
    widths = '·'.join(str(w) for w in P2['split_flare_widths'])
    if UFORMER:
        scale = f'{float(P2["split_flare_scale"]):g}'.replace('.', ',')
        if VEIL:
            fl = node(560, 32, 270, 84, 'flare', '③ KHỬ LÓE: Uformer + lưới', [f'Uformer Flare7K++ · ×{scale} → 512²',
                                                                               f'lưới màn sáng {VEIL}×{VEIL} ô (mới)',
                                                                               'ra: vào − lớp lóe · không đọc ZI'],
                      title_size=14)
            badge(566, 122, 'lưới train ở ④ · best.pt · đóng băng', color='#C62828')
        else:
            fl = node(560, 32, 270, 84, 'flare', '③ NHÁNH KHỬ LÓE: Uformer', ['Flare7K++ · 20,5 M tham số',
                                                                             f'ảnh ×{scale} → ô vuông 512² · fp32',
                                                                             'ra: vào − lớp lóe · không đọc ZI'],
                      title_size=14)
            badge(566, 122, 'fine-tune ở ④ · best.pt · đóng băng', color='#C62828')
    else:
        fl = node(560, 32, 270, 84, 'flare', '③ NHÁNH KHỬ LÓE', [f'U-Net 256² [{widths}] · {P2["split_flare_blocks"]} khối',
                                                                'ra: vào − lớp lóe (RGB)',
                                                                'đọc ZI' if P2.get('split_flare_latent', True)
                                                                else 'không đọc ZI'], title_size=14)
        badge(566, 122, 'train riêng ở ④ · đóng băng' if STAGE else 'mới · p41: loss lớp lóe riêng', color='#C62828')
    text(330, 68, 'ảnh nhiễu RGB 256²', size=13, color='#2E7D32', style='font-style="italic"')
    text(846, 68, 'ảnh đã bỏ lóe → tone cả ảnh và extra_embed của Restormer' if RESTORMER else
         'ảnh đã bỏ lóe → tone cả ảnh và mạng khôi phục' if RESTORE else
         'ảnh đã bỏ lóe → tầng tone (đường duy nhất mang MÀU)', size=12, color='#2E7D32', anchor='start',
         style='font-style="italic"')
else:
    text(470, 68, 'ảnh nhiễu RGB 256² — chỉ đi qua tầng tone (đường duy nhất mang MÀU)', size=13, color='#2E7D32',
         style='font-style="italic"')
arrow([(850, 76), (850, 170), (866, 170)], color='#2E7D32', width=2.4)
if RESTORE:                                              # p47+: also into the restoration net
    arrow([(1378, 76), (1378, 122)], color='#2E7D32', width=2.4)
elif RELIGHT:                                            # the grid reads the relit frame
    arrow([(1113, 76), (1113, 146)], color='#2E7D32', width=2.4)
else:
    arrow([(1266, 76), (1266, 146)], color='#2E7D32', width=2.4)
    arrow([(1418, 76), (1418, 146)], color='#2E7D32', width=2.4)
# ZI into the coefficient net.
arrow([(760, 205), (760, 194), (866, 194)], color='#8E24AA', width=2.4, label='ZI', lx=800, ly=188, lsize=12)


FX = 905
if RESTORE:
    # ---------------- phase 2 (p47+ restore_rgb): tone of the whole frame, then one RGB restoration net
    img_out = node(1392, 290, 100, 78, 'out', 'Ảnh', ['phục hồi', '= sáng + Δ' if HIGHPASS else '= tone + Δ'])
    if HIGHPASS:
        for k, line_text in enumerate(('Δ chỉ chi tiết:', f'Δ − Gauss σ {HIGHPASS:g}'.replace('.', ','),
                                       'không đổi độ sáng', 'theo vùng')):
            text(1442, 392 + 15 * k, line_text, size=11, weight='bold' if k == 0 else 'normal', color='#00897B')
    tone = node(866, 122, 180, 92, 'color', 'Tone cả ảnh', ['đường cong + ma trận màu', 'mỗi ảnh · phơi sáng + ZI',
                                                           'khởi tạo 0 = đồng nhất'], title_size=14)
    gyro_channels = 0
    if FIELD:
        samples = int(FIELD['samples'])
        gyro_channels = 2 * (samples - 1) + 1
        gyro = node(1066, 122, 180, 92, 'edge', 'Đường đi gyro', [f'trong {FIELD["seconds"]:g} s phơi sáng'.replace('.', ','),
                                                                  f'{samples} mẫu → {gyro_channels - 1} độ dịch u, v',
                                                                  f'+ cờ có IMU = {gyro_channels} kênh'], title_size=14)
        arrow([(1156, 98), (1156, 122)], dash='4 3', width=1.8, label='gyro của IMU nhiễu', lx=1162, ly=112, lsize=11,
              lanchor='start')
    grid = None
    if LIGHT_GRID and not FIELD:
        grid = node(1066, 122, 180, 92, 'color', 'Lưới ánh sáng (mới)', [f'{LIGHT_GRID}×{LIGHT_GRID} ô × 8 mức sáng',
                                                                       'affine màu theo vùng · ZI',
                                                                       'khởi tạo 0 = đồng nhất'], title_size=14)
        arrow([mid_right(tone), mid_left(grid)], color='#EF6C00', width=2.4)
    extra = f'ảnh bỏ lóe + {gyro_channels} kênh gyro' if gyro_channels else 'ảnh bỏ lóe'
    if RESTORMER:
        xin = node(1266, 122, 225, 92, 'edge', 'extra_embed (mới)', ['conv 3×3: ' + extra, '→ 48 kênh, cộng vào cấp 1',
                                                                    'khởi tạo 0: đầu = bản gốc'], title_size=14)
    else:
        xin = node(1266, 122, 225, 92, 'edge', 'Kênh vào thêm', [extra, 'ghép cạnh ảnh sau tone'], title_size=14)
    if FIELD:
        arrow([mid_right(gyro), mid_left(xin)], color='#2E7D32', width=2)
    el.append('<rect x="866" y="236" width="506" height="210" rx="12" fill="#FFFFFF" fill-opacity="0.85" '
              'stroke="#2E7D32" stroke-width="1.5"/>')
    if grid is not None:
        arrow([(1156, 214), (1156, 220), (900, 220), (900, 282)], color='#EF6C00', width=2.4)
    else:
        arrow([(900, 214), (900, 282)], color='#EF6C00', width=2.4)
    arrow([(1378, 214), (1378, 226), (930, 226), (930, 282)], color='#2E7D32', width=2.2)
    if RESTORMER:
        text(938, 254, 'ảnh sau tone + lưới → patch_embed (gốc)' if grid is not None else 'ảnh sau tone → patch_embed (gốc)',
             size=11, weight='bold', color='#EF6C00', anchor='start')
        text(938, 270, '+ extra_embed (mới) ở cấp 1', size=11, weight='bold', color='#2E7D32', anchor='start')
        e1 = level(880, 282, '256² · 48', '4 khối')
        e2 = level(950, 318, '128² · 96', '6 khối')
        e3 = level(1020, 354, '64² · 192', '6 khối')
        bottom = level(1090, 390, '32² · 384', '8 khối')
        d3 = level(1160, 354, '64² · 192', '6 khối')
        d2 = level(1230, 318, '128² · 96', '6 khối')
        d1 = level(1300, 282, '256² · 96', '4 khối + 4')
        green = '#2E7D32'
        arrow([(913, 312), (913, 333), (950, 333)], color=green, width=1.8)
        arrow([(983, 348), (983, 369), (1020, 369)], color=green, width=1.8)
        arrow([(1086, 376), (1112, 376), (1112, 390)], color=green, width=1.8)
        arrow([(1156, 405), (1193, 405), (1193, 384)], color=green, width=1.8)
        arrow([(1226, 369), (1263, 369), (1263, 348)], color=green, width=1.8)
        arrow([(1296, 333), (1333, 333), (1333, 312)], color=green, width=1.8)
        for (sx, sy), (tx, ty) in (((946, 290), (1300, 290)), ((1016, 326), (1230, 326)), ((1086, 362), (1160, 362))):
            arrow([(sx, sy), (tx, ty)], color='#90A4AE', dash='5 4', width=1.5)
        text(1123, 284, 'nối tắt', size=10, color='#78909C', style='font-style="italic"')
        text(1290, 408, 'RESTORMER khử mờ', size=13, weight='bold', color='#2E7D32')
        text(1290, 423, 'GoPro · train sẵn · 26,1 M', size=11, color='#37474F')
        scale = f'{float(P2["restore_pretrained_lr_scale"]):g}'.replace('.', ',')
        text(1290, 437, f'lr × {scale} · train ô {CROP}²' if CROP else f'fine-tune lr × {scale}', size=11,
             weight='bold', color='#37474F')
        text(870, 399, 'ZI → latent_in 1×1 (= 0)', size=11, weight='bold', color='#8E24AA', anchor='start')
        text(870, 437, 'khối: attention theo kênh (MDTA) + GDFN', size=10, color='#78909C', anchor='start')
        arrow([(1366, 297), (1380, 297), (1380, 329), (1392, 329)])
        zi_target = (1090, 405)
    else:
        widths = '·'.join(str(w) for w in P2['restore_widths'])
        net = node(880, 282, 480, 130, 'edge', 'NAFNet RGB 256² (train từ đầu)', [f'kênh [{widths}]',
                                                                                 'ZI ở đáy', 'cộng vào ảnh sau tone'])
        arrow([mid_right(net), (1380, 347), (1380, 329), (1392, 329)])
        zi_target = (880, 375)
    # ZI: into the tone step (above) and the net's bottom level.
    el.append(f'<path d="M 815 240 L 852 240 L 852 {zi_target[1]} L {zi_target[0] - 8} {zi_target[1]}" fill="none" '
              'stroke="#8E24AA" stroke-width="2.4"/>')
    arrow([(zi_target[0] - 9, zi_target[1]), zi_target], color='#8E24AA', width=2.4)
else:
    # ---------------- phase 2: colour and edge branches on J
    img_out = node(1392, 290, 100, 78, 'out', 'Ảnh', ['phục hồi'])
    EDGE_Y = 372


    def unet_block(x, yc, kind, title):
        """One branch as a U-Net at block level: Encoder -> Bottleneck (ZI joins) -> Decoder, skip across."""
        fill, stroke = C[kind]
        el.append(f'<polygon points="{x},{yc - 30} {x + 80},{yc - 13} {x + 80},{yc + 13} {x},{yc + 30}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        el.append(f'<polygon points="{x + 152},{yc - 13} {x + 232},{yc - 30} {x + 232},{yc + 30} {x + 152},{yc + 13}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        el.append(f'<rect x="{x + 86}" y="{yc - 13}" width="60" height="26" rx="4" fill="{C["lat"][0]}" '
                  f'stroke="{C["lat"][1]}" stroke-width="2"/>')
        text(x + 36, yc + 5, 'Encoder', size=12, weight='bold', color=stroke)
        text(x + 196, yc + 5, 'Decoder', size=12, weight='bold', color=stroke)
        text(x + 116, yc + 4, 'Bottleneck', size=10, weight='bold', color=C['lat'][1])
        line([(x + 80, yc), (x + 86, yc)], stroke, 2)
        line([(x + 146, yc), (x + 152, yc)], stroke, 2)
        el.append(f'<path d="M {x + 40} {yc + 24} C {x + 56} {yc + 54}, {x + 176} {yc + 54}, {x + 192} {yc + 24}" '
                  f'fill="none" stroke="#90A4AE" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ah-90A4AE)"/>')
        text(x + 116, yc + 53, 'skip connection', size=10, color='#78909C', style='font-style="italic"')
        text(x + 116, yc + 71, title, size=13, weight='bold', color=stroke)
        return (x, yc - 30, 232, 60)


    colb = node(FX, 250, 232, 60, 'color', 'Nhánh MÀU · ResNet 128²',
                ['tone chung (+ phân vị) → từng vùng' if EXPOSURE else 'tone/màu cả ảnh → từng vùng']
                + (['+ chi tiết màu 256² (p34)'] if CHROMA else []), title_size=14)
    edgb = unet_block(FX, EDGE_Y, 'edge', 'Nhánh ĐƯỜNG NÉT (Y của J) · NAFNet + 4 WF')
    for x in (FX + 4, FX + 198):
        el.append(f'<rect x="{x}" y="{EDGE_Y - 48}" width="30" height="15" rx="4" fill="#FCE4EC" stroke="#D81B60" '
                  f'stroke-width="1.6"/>')
        text(x + 15, EDGE_Y - 37, 'WF', size=10, weight='bold', color='#D81B60')
        arrow([(x + 15, EDGE_Y - 33), (x + 15, EDGE_Y - 24)], color='#D81B60', width=1.4)
    refine = node(1160, EDGE_Y - 24, 118, 48, 'edge', 'CNN làm nét', ['+ mượt · 4 khối'], rx=8, title_size=13)
    join = node(1292, 300, 58, 58, 'out', 'Ghép', [], rx=29, title_size=14)
    # J: from the tone stage down to both branches.
    line([(1418, 214), (1418, 236), (885, 236), (885, 280)], '#EF6C00', width=3)
    arrow([(885, 280), (FX, 280)], color='#EF6C00', width=3)
    arrow([(885, 280), (885, EDGE_Y), (FX - 2, EDGE_Y)], color='#EF6C00', width=3)
    text(1300, 254, 'J (ảnh đã chỉnh sáng) → cả hai nhánh', size=12, weight='bold', color='#EF6C00')
    # ZI: to the colour branch and the edge bottleneck (hopping over J).
    DX = FX + 116
    el.append(f'<path d="M 815 240 L 845 240 L 845 317 L 878 317 A 7 7 0 0 1 892 317 L {DX} 317" '
              f'fill="none" stroke="#8E24AA" stroke-width="2.4"/>')
    arrow([(DX, 317), (DX, 312)], color='#8E24AA', width=2.4)
    arrow([(DX, 317), (DX, EDGE_Y - 14)], color='#8E24AA', width=2.4)
    text(1030, 334, 'ZI', size=12, weight='bold', color='#8E24AA', anchor='start')
    arrow([mid_right(colb), (1272, 280), (1272, 320), (1292, 320)])
    arrow([mid_right(edgb), mid_left(refine)])
    arrow([mid_right(refine), (1286, EDGE_Y), (1286, 338), (1292, 338)])
    arrow([mid_right(join), mid_left(img_out)])

imu_dec = node(FX, 458, 232, 44, 'edge', 'Decoder IMU', ['Haar · không skip từ encoder'], title_size=14)
imu_ref = node(1160, 452, 190, 58, 'edge', 'CNN làm mượt IMU', ['1-D · dilation 1-2-4-8 · 0,65 s',
                                                              'đọc cả IMU nhiễu'], rx=8, title_size=13)
imu_out = node(1392, 452, 100, 58, 'out', 'IMU', ['phục hồi'])
# ZU: branches off phase 1's line to the IMU decoder.
dot(760, 480, '#455A64')
arrow([(760, 480), (FX, 480)])
arrow([mid_right(imu_dec), mid_left(imu_ref)])
arrow([mid_right(imu_ref), mid_left(imu_out)])


# ---------------- phase 1: I-JEPA
clean = node(20, 565 + OY, 150, 70, 'data', 'Ảnh + IMU', ['SẠCH'])
teach = node(205, 548 + OY, 180, 94, 'p1', 'Teacher EMA', ['bản EMA của encoder', 'cặp SẠCH → LayerNorm',
                                                        '= đích ở khối đích'])
badge(215, 646 + OY, 'bỏ sau phase 1', color='#78909C')
recon_weight = f'{float(P1.get("ijepa_reconstruction_weight", 0.0)):g}'.replace('.', ',')
loss1 = node(420, 548 + OY, 240, 94, 'loss', 'Loss I-JEPA', ['smooth L1(đoán, đích)',
                                                            f'ảnh ½ + IMU ½ · batch {P1["batch_size"]}']
             + ([f'+ tái tạo × {recon_weight}: L1 + độ dốc', 'không VICReg · Jacobian'] if RECON
                else ['không VICReg · neo · Jacobian']))
pred = node(705, 548 + OY, 170, 94, 'p1', 'Predictor chung',
            [f'ViT hẹp · {P1["ijepa_predictor_dim"]} kênh · {P1["ijepa_predictor_depth"]} khối',
             'ngữ cảnh ảnh + IMU', '+ mask token ở vị trí đích'], title_size=15)
badge(715, 646 + OY, 'bỏ sau phase 1', color='#78909C')
arrow([mid_right(clean), mid_left(teach)])
arrow([mid_right(teach), mid_left(loss1)], label='đích', lx=402, ly=586 + OY)
arrow([mid_left(pred), mid_right(loss1)], label='đoán', lx=683, ly=586 + OY)
arrow([mid_bot(zu), (760, 548 + OY)], label='ZU', lx=752, ly=500 + OY, lanchor='end')
el.append(f'<path d="M 815 262 L 829 262 L 829 473 A 7 7 0 0 1 829 487 L 829 {548 + OY}" '
          'fill="none" stroke="#455A64" stroke-width="2.2" marker-end="url(#ah)"/>')
text(823, 470 + OY, 'ZI', size=13, color='#455A64', anchor='end', style='font-style="italic"')
text(837, 538, 'lúc train: encoder chạy trên token NGỮ CẢNH' + (' (+ cả khung cho đầu tái tạo)' if RECON else ''),
     size=12, weight='bold', color='#455A64', anchor='start')
text(837, 553, 'lúc khôi phục: trên mọi token của cặp nhiễu', size=12, color='#455A64', anchor='start')
el.append(f'<path d="M 470 445 L 470 {522 + OY} L 375 {522 + OY} L 375 {548 + OY}" fill="none" stroke="#F9A825" '
          'stroke-width="2" stroke-dasharray="6 4" marker-end="url(#ah-F9A825)"/>')
text(478, 490 + OY, f'EMA: m {P1["teacher_momentum_start"]:g} → {P1["teacher_momentum_end"]:g} (tuyến tính)'.replace('.', ','),
     size=12, color='#B26A00', anchor='start', style='font-style="italic"')
context, targets = multiblock_masks(
    (16, 16), 1, MASK_SEED, targets=P1["ijepa_targets"], target_scale=tuple(P1["ijepa_target_scale"]),
    target_aspect=tuple(P1["ijepa_target_aspect"]), context_scale=tuple(P1["ijepa_context_scale"]),
    min_keep=P1["ijepa_image_min_keep"])
seen, hidden = set(context[0].tolist()), set(targets[0].flatten().tolist())
GX, GY, CELL = 912, 540 + OY, 6.5
for index in range(256):
    row, column = divmod(index, 16)
    fill = '#F6B26B' if index in hidden else '#9FC5E8' if index in seen else '#FFFFFF'
    el.append(f'<rect x="{GX + column * CELL}" y="{GY + row * CELL}" width="{CELL}" height="{CELL}" '
              f'fill="{fill}" stroke="#CFD8DC" stroke-width="0.5"/>')
el.append(f'<rect x="{GX}" y="{GY}" width="{16 * CELL}" height="{16 * CELL}" fill="none" stroke="#546E7A" stroke-width="1.2"/>')
for block in targets[0]:
    rows, columns = (block // 16).tolist(), (block % 16).tolist()
    el.append(f'<rect x="{GX + min(columns) * CELL}" y="{GY + min(rows) * CELL}" '
              f'width="{(max(columns) - min(columns) + 1) * CELL}" height="{(max(rows) - min(rows) + 1) * CELL}" '
              f'fill="none" stroke="#E65100" stroke-width="1.6"/>')
arrow([(GX - 4, GY + 52), (875, GY + 52)], **({} if RECON else dict(label='vị trí', lx=893, ly=GY + 44, lsize=11)))
for k, (fill, label) in enumerate((('#9FC5E8', f'ngữ cảnh {P1["ijepa_context_scale"][0]:g}–{P1["ijepa_context_scale"][1]:g}'
                                                f' khung, đã cắt đích'),
                                   ('#F6B26B', f'{P1["ijepa_targets"]} khối đích {P1["ijepa_target_scale"][0]:g}–'
                                               f'{P1["ijepa_target_scale"][1]:g} khung'))):
    y = GY + 16 * CELL + 16 + 16 * k
    el.append(f'<rect x="{GX}" y="{y - 9}" width="10" height="10" fill="{fill}" stroke="#546E7A" stroke-width="0.8"/>')
    text(GX + 15, y, label.replace('.', ','), size=11, color='#37474F', anchor='start')
text(GX + 8 * CELL, GY - 6, 'mask thật · 16 × 16 token' + (' → vị trí' if RECON else ''), size=11, weight='bold',
     color='#546E7A')
text(GX + 16 * CELL + 10, GY + 40, 'IMU: cùng cách,', size=11, color='#546E7A', anchor='start')
text(GX + 16 * CELL + 10, GY + 54, 'đoạn trên 8 token', size=11, color='#546E7A', anchor='start')
if RECON:
    # p48: the reconstruction head reads ZI of the whole noisy frame and redraws the clean one.
    head = node(705, 736, 170, 72, 'p1', 'Đầu tái tạo (p48)', ['ZI cả khung nhiễu → ảnh',
                                                             f'SẠCH (Y) · {P1["ijepa_reconstruction_width"]} kênh · ×16',
                                                             'không skip: chỉ qua latent'], title_size=14)
    badge(715, 812, 'bỏ sau phase 1', color='#78909C')
    dot(829, 573, '#455A64')
    el.append('<path d="M 829 573 L 893 573 L 893 645 A 7 7 0 0 1 893 659 L 893 772 L 877 772" fill="none" '
              'stroke="#455A64" stroke-width="2.2" marker-end="url(#ah)"/>')
    arrow([mid_left(head), (540, 772), (540, 642 + OY)], label='tái tạo', lx=622, ly=766, lsize=12)

# ---------------- phase 2 losses: what the config turns on
perceptual = f'{float(P2.get("perceptual_weight", 0.0)):g}'.replace('.', ',')
if RESTORE:
    brighten = P2.get('target_brighten')
    terms = [f'ảnh: L1 · chi tiết QWT của Y · VGG16 × {perceptual}',
             f'phổ: L1 của FFT phức ảnh ra × {float(P2["restore_fft_weight"]):g}'.replace('.', ','),
             f'tone: L1 ảnh sau tone ↔ ảnh sạch ở 1/8 × {float(P2["restore_tone_weight"]):g}'.replace('.', ','),
             (f'đích quá tối (< {brighten["below"]:g}) nâng lên {brighten["target"]:g}'.replace('.', ',')
              if brighten else ''),
             f'ảnh: chấm trên ô {CROP}² · tone: trên cả ảnh' if CROP else '']
else:
    terms = ['ảnh: L1 · chi tiết QWT của Y · VGG16 (perceptual)', 'màu: L1 Cb/Cr + thống kê màu từng ảnh',
         'nét: L1 · độ dốc · FFT phức · 128², 64² · mượt',
         'tone: L1 giữa J và ảnh sạch ở 32²' if GRID or P2.get('split_light_branch', False) else '',
         'stop: L1 giữa S và stop thật của bước nhiễu' if RELIGHT else '',
         'cổng tone: BCE với nhãn "ánh sáng bị đổi"' if GATE else '',
         'lóe: L1 lớp lóe với (ảnh lóe − ảnh sạch)' if FLARE and float(P2.get('split_flare_weight', 0)) > 0 else '']
if LOWFREQ:
    terms.append(f'độ sáng: MSE ảnh trung bình khối 8 px × {LOWFREQ:g}'.replace('.', ','))
terms.append('IMU: L1 · chi tiết Haar · rung · rung thừa · gia số')
if KIND_BALANCE:
    terms.append('cân bằng: mỗi loại ảnh nặng ngang nhau (kind_balance)')
terms = [t for t in terms if t]
loss2 = node(1150, 562, 350, 30 + 17 * len(terms) + 34, 'loss', 'Loss phase 2', terms)
recipe = (f'phạt nét ×1,5 · cắt gradient ở {P2["gradient_clip_norm"]:g} · '.replace('.', ',')
          + f'lr {P2["learning_rate"]:.0e}'.replace('e-0', 'e-')
          + (f' (Restormer × {float(P2["restore_pretrained_lr_scale"]):g})'.replace('.', ',') if RESTORMER else ''))
text(1325, 562 + 30 + 17 * len(terms) + 26, recipe, size=12, weight='bold', color='#D81B60')
arrow([(1325, 562), (1325, 520)], color='#D81B60', dash='6 4', width=2)
text(1333, 538, 'train decoder', size=12, color='#D81B60', anchor='start', style='font-style="italic"')
text(1333, 553, '25% cuối thêm backbone', size=12, color='#D81B60', anchor='start', style='font-style="italic"')

# ---------------- band 4: the flare stage (p42), before phase 2
if STAGE:
    BY = 800 + RH
    region(20, BY, 1485, 170, 'flare', '④ Giai đoạn lóe — train-flare, chạy SAU phase 1 và TRƯỚC phase 2: chỉ train nhánh '
           'khử lóe, rồi phase 2 nạp và đóng băng nó')
    lens = STAGE['lens']
    kinds = STAGE['kinds']
    optics = round(100 * float(lens['lens_optics_probability']))
    real = round(100 * float(lens['lens_flare_real_fraction']))
    wu = round(100 * float(lens.get('lens_flare_wu_fraction', 0.0)))
    if wu:
        ghosts = float(lens.get('lens_wu_ghost_boost', [0, 0])[1]) > 0
        s1 = node(32, BY + 36, 258, 50, 'flare', f'Lóe: Wu 2021 ban ngày {wu}%',
                  ['+ bóng ma lục giác/tròn · halo' if ghosts else
                   f'+ Flare7K++ {100 - wu}% · không bóng ma' if not optics else
                   f'+ Flare7K++ {100 - wu}% · quang học {optics}%'], title_size=13)
    else:
        s1 = node(32, BY + 36, 258, 50, 'flare', 'Lớp lóe Flare7K++',
                  [f'{real}% chụp thật Flare-R · quang học {optics}%'], title_size=13)
    pair_bg = round(100 * float(STAGE['background_pair_fraction']))
    scene_bg = float(STAGE.get('background_scene_fraction', 0.0))
    if scene_bg:
        # flare_stage._background: drone first, then FlareImage's gt, else TartanAir.
        pair_share = (1 - scene_bg) * float(STAGE['background_pair_fraction'])
        s2 = node(32, BY + 92, 258, 50, 'data', f'Ảnh nền sạch: drone {scene_bg:.0%}',
                  [f'gt FlareImage {pair_share:.0%} · TartanAir {1 - scene_bg - pair_share:.0%}'], title_size=13)
    else:
        s2 = node(32, BY + 92, 258, 50, 'data', 'Ảnh nền sạch',
                  [f'TartanAir {100 - pair_bg}% · gt FlareImage {pair_bg}%'], title_size=13)
    s3 = node(310, BY + 40, 260, 96, 'flare', 'Ghép (ánh sáng tuyến tính)',
              ['vào = nền + lóe', 'đích = nền + nguồn sáng (giữ đèn)', 'mờ: cả hai · hạt: chỉ ảnh vào',
               'mỗi mẫu một lần rút MỚI'], title_size=14)
    share = lambda k: f'{round(100 * float(kinds[k]) / sum(float(v) for v in kinds.values()))}%'
    widths = '·'.join(str(w) for w in STAGE['widths'])
    precision = 'fp16' if STAGE['precision'] == 'amp_fp16' else 'fp32'
    every = f'{STAGE["validation_every_updates"]:,}'.replace(',', ' ')
    if VEIL:
        s4 = node(615, BY + 40, 250, 96, 'flare', 'UFORMER → LƯỚI MÀN SÁNG', [
            'Uformer: best.pt run trước' + (', đóng băng' if VEIL_ONLY else ''),
            f'lưới song phương {VEIL}×{VEIL} ô × 8 mức (mới)',
            'cả khung 64² + bối cảnh toàn ảnh',
            'khởi tạo 0 → lớp lóe đoán'], title_size=15)
    elif UFORMER:
        s4 = node(615, BY + 40, 250, 96, 'flare', 'UFORMER FLARE7K++', [
                                                                       'tiếp từ flare/best.pt run trước'
                                                                       if STAGE.get('uformer_init') == 'stage'
                                                                       else 'khởi tạo: trọng số công bố',
                                                                       f'ảnh ×{float(STAGE["uformer_scale"]):g} → 512²'.replace('.', ','),
                                                                       'checkpointing · → lớp lóe đoán'], title_size=15)
    else:
        s4 = node(615, BY + 40, 250, 96, 'flare', 'NHÁNH KHỬ LÓE', [f'U-Net 256² [{widths}]', f'{STAGE["blocks"]} khối · không đọc ZI',
                                                                  '→ lớp lóe đoán'], title_size=15)
    s5 = node(910, BY + 40, 250, 96, 'loss', 'Loss L1 lớp lóe' + (' (chỉ lưới)' if VEIL_ONLY else
                                                                  ' (fine-tune)' if UFORMER else ''),
              ['đích: vào (trước hạt) − đích', f'{STAGE["max_successful_updates"]:,} bước × {STAGE["batch_size"]}'.replace(',', ' '),
               f'AdamW {STAGE["learning_rate"]:.0e} · {precision} · DDP 2 GPU'.replace('e-0', 'e-')], title_size=15)
    if STAGE.get('select_best', False):
        both = STAGE.get('select_on') in ('real_and_composite', 'real_composite_scene')
        scene = STAGE.get('select_on') == 'real_composite_scene'
        s6 = node(1205, BY + 40, 285, 96, 'data', 'Validation → giữ best.pt',
                  ['FlareImage + lóe ghép + cặp drone valid' if scene else
                   'cặp FlareImage + lóe ghép valid' if both else 'cặp FlareImage valid (ảnh thật)',
                   f'bước 0 (chưa fine-tune) + mỗi {every} bước',
                   'PSNR lời TB cao nhất → best.pt' if both else 'PSNR cả ô cắt cao nhất → best.pt',
                   'không hơn bước 0 → giữ bản run trước' if STAGE.get('uformer_init') == 'stage'
                   else 'không hơn bước 0 → giữ Uformer gốc'], title_size=14)
    else:
        s6 = node(1205, BY + 40, 285, 96, 'data', 'Validation (chưa thấy)',
                  ['lóe ghép valid: % lóe còn lại', 'cặp FlareImage valid (ảnh thật):', 'PSNR vùng lóe · ngoài lóe',
                   f'mỗi {every} bước'], title_size=14)
    arrow([mid_right(s1), (300, BY + 61), (300, BY + 75), (310, BY + 75)], color='#C62828')
    arrow([mid_right(s2), (300, BY + 117), (300, BY + 101), (310, BY + 101)], color='#C62828')
    arrow([mid_right(s3), mid_left(s4)], color='#C62828')
    arrow([mid_right(s4), mid_left(s5)], color='#C62828')
    arrow([mid_right(s5), mid_left(s6)], color='#90A4AE', dash='5 4')
    text(760, BY + 160, f'mẫu: {share("composite")} lóe ghép · {share("pair")} cặp FlareImage thật · '
         + (f'{share("scene")} cặp drone có lóe sẵn · ' if 'scene' in kinds else '')
         + f'{share("plain")} ảnh TartanAir không lóe (lớp lóe đích = 0, để yên ảnh không lóe)', size=12,
         color='#C62828', style='font-style="italic"')
el.append('</g>')

markers = ''.join(
    f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in ('#2E7D32', '#EF6C00', '#F9A825', '#90A4AE', '#1E88E5', '#8E24AA', '#D81B60', '#C62828'))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="#455A64"/></marker>{markers}</defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + ''.join(el) + '</svg>')
open(sys.argv[1], 'w', encoding='utf-8').write(svg)
