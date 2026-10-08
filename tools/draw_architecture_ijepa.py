"""Architecture figure of the run the notebook trains (configs/kaggle_relight.yaml, p32_relight;
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

Usage: python3 tools/draw_architecture_ijepa.py docs/kien_truc_ijepa.svg [config]
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.config import load_config  # noqa: E402
from qjepa.models.ijepa import multiblock_masks  # noqa: E402

CONFIG_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "configs/kaggle_relight.yaml"
CONFIG = load_config(CONFIG_PATH)
P1, P2, MODEL = CONFIG["phase1"], CONFIG["phase2"], CONFIG["model"]
RUN = Path(CONFIG["runtime"]["output_dir"]).name
GRID = bool(P2.get("split_tone_grid", False))
RELIGHT = bool(P2.get("split_relight", False))
EXPOSURE = bool(P2.get("split_exposure_stats", False))
LOWFREQ = float(P2.get("lowfreq_mse_weight", 0.0))
# A typical draw (context 40% of the tokens; median 42%, p10-p90 33-51% over 400 draws) whose four
# target blocks barely overlap (6%; median 32%), so each block can be seen.
MASK_SEED = 382

W, H = 1520, 790
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


def mid_right(b): x, y, w, h = b; return (x + w, y + h / 2)
def mid_left(b):  x, y, w, h = b; return (x, y + h / 2)
def mid_top(b):   x, y, w, h = b; return (x + w / 2, y)
def mid_bot(b):   x, y, w, h = b; return (x + w / 2, y + h)


# ---------------- header
text(20, 28, f'{RUN} — kiến trúc đang train · phase 1 I-JEPA (backbone ViT trên hệ số wavelet) · phase 2 khôi phục'
     f'{" với bản đồ stop + lưới song phương" if RELIGHT else " với tầng tone lưới song phương" if GRID else ""}',
     size=16, weight='bold', color='#D81B60', anchor='start')
text(20, 48, 'MỘT ViT chung cho ảnh và IMU (attention giữa hai loại token là fusion) · ngữ cảnh NHIỄU, teacher EMA SẠCH'
     ' · khôi phục đọc ĐẦU RA CONTEXT ENCODER (đóng băng, mọi token nhiễu) · teacher + predictor bỏ sau phase 1',
     size=12, color='#AD1457', anchor='start')

# ---------------- regions
region(190, 150, 490, 320, 'bb', '① Context encoder — học ở phase 1, đóng băng ở phase 2')
region(842, 88, 663, 432, 'edge', '③ Phase 2 — khôi phục (backbone đóng băng)', right=RELIGHT)   # ảnh nhiễu rơi ở x 1113
OY = 60                                                   # phase 1 sits below phase 2's taller region
region(190, 505 + OY, 940, 200, 'p1', '② Phase 1 — I-JEPA')

# ---------------- inputs and backbone
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

# ---------------- phase 2: the tone stage
el.append('<rect x="856" y="116" width="636" height="110" rx="10" fill="#FFFFFF" fill-opacity="0.8" '
          'stroke="#EF6C00" stroke-width="1.5"/>')
if RELIGHT:
    cells, bins, size = P2['split_tone_grid_size'], P2['split_tone_grid_bins'], P2['split_relight_size']
    text(868, 136, 'TẦNG TONE', size=14, weight='bold', color='#EF6C00', anchor='start')   # ảnh nhiễu rơi vào x 1113
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
line([(95, 200), (95, 76), (1113 if RELIGHT else 1418, 76)], '#2E7D32', width=3)
text(470, 68, 'ảnh nhiễu RGB 256² — chỉ đi qua tầng tone (đường duy nhất mang MÀU)', size=13, color='#2E7D32',
     style='font-style="italic"')
arrow([(850, 76), (850, 170), (866, 170)], color='#2E7D32', width=2.4)
if RELIGHT:                                              # the grid reads the relit frame
    arrow([(1113, 76), (1113, 146)], color='#2E7D32', width=2.4)
else:
    arrow([(1266, 76), (1266, 146)], color='#2E7D32', width=2.4)
    arrow([(1418, 76), (1418, 146)], color='#2E7D32', width=2.4)
# ZI into the coefficient net.
arrow([(760, 205), (760, 194), (866, 194)], color='#8E24AA', width=2.4, label='ZI', lx=800, ly=188, lsize=12)

# ---------------- phase 2: colour and edge branches on J, IMU decoder
FX = 905
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
            ['tone chung (+ phân vị) → từng vùng' if EXPOSURE else 'tone/màu cả ảnh → từng vùng'], title_size=14)
edgb = unet_block(FX, EDGE_Y, 'edge', 'Nhánh ĐƯỜNG NÉT (Y của J) · NAFNet + 4 WF')
for x in (FX + 4, FX + 198):
    el.append(f'<rect x="{x}" y="{EDGE_Y - 48}" width="30" height="15" rx="4" fill="#FCE4EC" stroke="#D81B60" '
              f'stroke-width="1.6"/>')
    text(x + 15, EDGE_Y - 37, 'WF', size=10, weight='bold', color='#D81B60')
    arrow([(x + 15, EDGE_Y - 33), (x + 15, EDGE_Y - 24)], color='#D81B60', width=1.4)
refine = node(1160, EDGE_Y - 24, 118, 48, 'edge', 'CNN làm nét', ['+ mượt · 4 khối'], rx=8, title_size=13)
join = node(1292, 300, 58, 58, 'out', 'Ghép', [], rx=29, title_size=14)
img_out = node(1392, 290, 100, 78, 'out', 'Ảnh', ['phục hồi'])
imu_dec = node(FX, 458, 232, 44, 'edge', 'Decoder IMU', ['Haar · không skip từ encoder'], title_size=14)
imu_ref = node(1160, 452, 190, 58, 'edge', 'CNN làm mượt IMU', ['1-D · dilation 1-2-4-8 · 0,65 s',
                                                              'đọc cả IMU nhiễu'], rx=8, title_size=13)
imu_out = node(1392, 452, 100, 58, 'out', 'IMU', ['phục hồi'])
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
arrow([mid_right(zu), (850, 400), (850, 480), (FX, 480)])
arrow([mid_right(imu_dec), mid_left(imu_ref)])
arrow([mid_right(imu_ref), mid_left(imu_out)])

# ---------------- phase 1: I-JEPA
clean = node(20, 565 + OY, 150, 70, 'data', 'Ảnh + IMU', ['SẠCH'])
teach = node(205, 548 + OY, 180, 94, 'p1', 'Teacher EMA', ['bản EMA của encoder', 'cặp SẠCH → LayerNorm',
                                                        '= đích ở khối đích'])
badge(215, 646 + OY, 'bỏ sau phase 1', color='#78909C')
loss1 = node(420, 548 + OY, 240, 94, 'loss', 'Loss I-JEPA', ['smooth L1(đoán, đích)',
                                                            f'ảnh ½ + IMU ½ · batch {P1["batch_size"]}',
                                                            'không VICReg · neo · Jacobian'])
pred = node(705, 548 + OY, 170, 94, 'p1', 'Predictor chung',
            [f'ViT hẹp · {P1["ijepa_predictor_dim"]} kênh · {P1["ijepa_predictor_depth"]} khối',
             'ngữ cảnh ảnh + IMU', '+ mask token ở vị trí đích'], title_size=15)
badge(715, 646 + OY, 'bỏ sau phase 1', color='#78909C')
arrow([mid_right(clean), mid_left(teach)])
arrow([mid_right(teach), mid_left(loss1)], label='đích', lx=402, ly=586 + OY)
arrow([mid_left(pred), mid_right(loss1)], label='đoán', lx=683, ly=586 + OY)
arrow([mid_bot(zu), (760, 548 + OY)], label='ZU', lx=752, ly=500 + OY, lanchor='end')
el.append(f'<path d="M 815 262 L 829 262 L 829 393 A 7 7 0 0 1 829 407 L 829 {548 + OY}" '
          'fill="none" stroke="#455A64" stroke-width="2.2" marker-end="url(#ah)"/>')
text(823, 470 + OY, 'ZI', size=13, color='#455A64', anchor='end', style='font-style="italic"')
text(837, 538, 'lúc train: encoder chỉ chạy trên token NGỮ CẢNH', size=12, weight='bold', color='#455A64',
     anchor='start')
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
arrow([(GX - 4, GY + 52), (875, GY + 52)], label='vị trí', lx=893, ly=GY + 44, lsize=11)
for k, (fill, label) in enumerate((('#9FC5E8', f'ngữ cảnh {P1["ijepa_context_scale"][0]:g}–{P1["ijepa_context_scale"][1]:g}'
                                                f' khung, đã cắt đích'),
                                   ('#F6B26B', f'{P1["ijepa_targets"]} khối đích {P1["ijepa_target_scale"][0]:g}–'
                                               f'{P1["ijepa_target_scale"][1]:g} khung'))):
    y = GY + 16 * CELL + 16 + 16 * k
    el.append(f'<rect x="{GX}" y="{y - 9}" width="10" height="10" fill="{fill}" stroke="#546E7A" stroke-width="0.8"/>')
    text(GX + 15, y, label.replace('.', ','), size=11, color='#37474F', anchor='start')
text(GX + 8 * CELL, GY - 6, 'mask thật · 16 × 16 token', size=11, weight='bold', color='#546E7A')
text(GX + 16 * CELL + 10, GY + 40, 'IMU: cùng cách,', size=11, color='#546E7A', anchor='start')
text(GX + 16 * CELL + 10, GY + 54, 'đoạn trên 8 token', size=11, color='#546E7A', anchor='start')

# ---------------- phase 2 losses: what the config turns on
terms = ['ảnh: L1 · chi tiết QWT của Y · VGG16 (perceptual)', 'màu: L1 Cb/Cr + thống kê màu từng ảnh',
         'nét: L1 · độ dốc · FFT phức · 128², 64² · mượt',
         'tone: L1 giữa J và ảnh sạch ở 32²' if GRID or P2.get('split_light_branch', False) else '',
         'stop: L1 giữa S và stop thật của bước nhiễu' if RELIGHT else '']
if LOWFREQ:
    terms.append(f'độ sáng: MSE ảnh trung bình khối 8 px × {LOWFREQ:g}'.replace('.', ','))
terms.append('IMU: L1 · chi tiết Haar · rung · rung thừa · gia số')
terms = [t for t in terms if t]
loss2 = node(1150, 562, 350, 30 + 17 * len(terms) + 34, 'loss', 'Loss phase 2', terms)
recipe = (f'phạt nét ×1,5 · cắt gradient ở {P2["gradient_clip_norm"]:g} · '.replace('.', ',')
          + f'lr {P2["learning_rate"]:.0e}'.replace('e-0', 'e-'))
text(1325, 562 + 30 + 17 * len(terms) + 26, recipe, size=12, weight='bold', color='#D81B60')
arrow([(1325, 562), (1325, 520)], color='#D81B60', dash='6 4', width=2)
text(1333, 538, 'train decoder', size=12, color='#D81B60', anchor='start', style='font-style="italic"')
text(1333, 553, '25% cuối thêm backbone', size=12, color='#D81B60', anchor='start', style='font-style="italic"')

markers = ''.join(
    f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in ('#2E7D32', '#EF6C00', '#F9A825', '#90A4AE', '#1E88E5', '#8E24AA', '#D81B60'))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="#455A64"/></marker>{markers}</defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + ''.join(el) + '</svg>')
open(sys.argv[1], 'w', encoding='utf-8').write(svg)
