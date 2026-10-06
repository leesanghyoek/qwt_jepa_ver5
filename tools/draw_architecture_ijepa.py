"""Architecture figure of p26_ijepa (configs/kaggle_ijepa.yaml, qwt-jaco-jepa-ijepa.ipynb), as plain SVG.

Same visual language as draw_architecture_imu.py (whose --local figure is p25_local), with
what p26 changes: the backbone is ONE ViT over the image's and the IMU's wavelet
coefficients together -- the attention between the two kinds of token is the fusion, so
there is no fusion module and ZI = FI, ZU = FU -- and phase 1 is I-JEPA alone: the context
encoder (the online ViT, on the NOISY pair's context tokens only), one narrow ViT predictor
for the targets of both, and the EMA teacher on the CLEAN pair. The mask drawn in phase 1 is a real one: multiblock_masks with the
config's settings. Phase 2 is p25's, less the encoder stages into NAFNet and the
predictor's guess into ZI (a ViT has one resolution; I-JEPA's predictor needs masks).

Usage: python3 tools/draw_architecture_ijepa.py docs/kien_truc_ijepa.svg
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.config import load_config  # noqa: E402
from qjepa.models.ijepa import multiblock_masks  # noqa: E402

CONFIG = load_config(REPO / "configs/kaggle_ijepa.yaml")
P1, MODEL = CONFIG["phase1"], CONFIG["model"]
# A typical draw (context 40% of the tokens; median 42%, p10-p90 33-51% over 400 draws) whose four
# target blocks barely overlap (6%; median 32%), so each block can be seen.
MASK_SEED = 382

W, H = 1520, 720
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


def badge(x, y, label):
    """A small tag for what this run adds."""
    width = 12 + 7 * len(label)
    el.append(f'<rect x="{x}" y="{y}" width="{width}" height="18" rx="9" fill="#D81B60"/>')
    text(x + width / 2, y + 13, label, size=11, weight='bold', color='#FFFFFF')


def mid_right(b): x, y, w, h = b; return (x + w, y + h / 2)
def mid_left(b):  x, y, w, h = b; return (x, y + h / 2)
def mid_top(b):   x, y, w, h = b; return (x + w / 2, y)
def mid_bot(b):   x, y, w, h = b; return (x + w / 2, y + h)


# ---------------- regions
region(190, 150, 490, 320, 'bb', '① Backbone ViT — train ở phase 1, đóng băng ở phase 2')
region(842, 110, 663, 360, 'edge', '③ Phase 2 — decoder khôi phục (như p25)', right=True)
region(190, 505, 940, 200, 'p1', '② Phase 1 — I-JEPA')

text(W - 20, 26, 'p26_ijepa — phase 1 là I-JEPA, không có gì khác · backbone ViT trên hệ số QWT',
     size=15, weight='bold', color='#D81B60', anchor='end')
text(W - 20, 44, 'MỘT ViT chung cho ảnh và IMU: attention giữa hai loại token là fusion · ngữ cảnh NHIỄU,'
     ' teacher EMA SẠCH · nhiễu và phase 2 như p25_local', size=12, color='#AD1457', anchor='end')

# ---------------- inputs and backbone
img_in = node(20, 200, 150, 80, 'data', 'Ảnh mờ + tối', ['3 × 256 × 256', 'tối theo vùng'])
imu_in = node(20, 365, 150, 70, 'data', 'IMU nhiễu', ['6 × 128'])
qwt = node(210, 205, 135, 70, 'tf', 'QWT Hilbert', ['kênh sáng Y', '16 × 128 × 128'])
haar = node(210, 365, 135, 70, 'tf', 'Haar', ['12 × 64'])
depth, heads, dim = MODEL["vit_depth"], MODEL["vit_heads"], MODEL["embedding_dim"]
vit = node(370, 195, 200, 250, 'bb', 'ViT CHUNG', ['ảnh: patch 8 = 16 px', '→ 256 token', '',
                                                  'IMU: patch 8 = 16 mẫu', '→ 8 token', '',
                                                  f'264 token × {dim}', f'{depth} khối · {heads} head',
                                                  '+ loại token + vị trí'])
badge(478, 186, 'mới · ViT chung')
text(637, 305, 'attention', size=12, weight='bold', color='#1E88E5')
text(637, 320, 'ảnh ↔ IMU', size=12, weight='bold', color='#1E88E5')
text(637, 335, '= fusion', size=12, weight='bold', color='#1E88E5')
zi = node(705, 205, 110, 70, 'lat', 'ZI = FI', ['128 × 16 × 16'])
zu = node(705, 365, 110, 70, 'lat', 'ZU = FU', ['128 × 8'])
el.append('<circle cx="190" cy="240" r="13" fill="#E8EAF6" stroke="#3949AB" stroke-width="2"/>')
text(190, 245, 'Y', size=14, weight='bold', color='#3949AB')
text(190, 196, 'tách màu', size=11, weight='bold', color='#3949AB')

# ---------------- phase 2 (p25's): light branch, colour ResNet, edge NAFNet + WF, IMU decoder + smoother
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
    el.append(f'<path d="M {x + 80} {yc} L {x + 86} {yc}" stroke="{stroke}" stroke-width="2"/>')
    el.append(f'<path d="M {x + 146} {yc} L {x + 152} {yc}" stroke="{stroke}" stroke-width="2"/>')
    el.append(f'<path d="M {x + 40} {yc + 24} C {x + 56} {yc + 54}, {x + 176} {yc + 54}, {x + 192} {yc + 24}" '
              f'fill="none" stroke="#90A4AE" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ah-90A4AE)"/>')
    text(x + 116, yc + 53, 'skip connection', size=10, color='#78909C', style='font-style="italic"')
    text(x + 116, yc + 71, title, size=13, weight='bold', color=stroke)
    return (x, yc - 30, 232, 60)


el.append('<rect x="860" y="125" width="285" height="272" rx="10" fill="#FFFFFF" fill-opacity="0.7" stroke="#2E7D32" stroke-width="1.5"/>')
FX = 905
colb = node(FX, 168, 232, 64, 'color', 'Nhánh MÀU · ResNet 128²', ['tone/màu cả ảnh → từng vùng'], title_size=14)
edgb = unet_block(FX, 320, 'edge', 'Nhánh ĐƯỜNG NÉT (Y) · NAFNet + 4 WF')
for x in (FX + 4, FX + 198):
    el.append(f'<rect x="{x}" y="266" width="30" height="17" rx="4" fill="#FCE4EC" stroke="#D81B60" stroke-width="1.6"/>')
    text(x + 15, 279, 'WF', size=10, weight='bold', color='#D81B60')
    arrow([(x + 15, 283), (x + 15, 296)], color='#D81B60', width=1.4)
refine = node(1160, 296, 118, 48, 'edge', 'CNN làm nét', ['+ mượt · 4 khối'], rx=8, title_size=13)
join = node(1292, 231, 58, 58, 'out', 'Ghép', [], rx=29, title_size=14)
img_out = node(1392, 221, 100, 78, 'out', 'Ảnh', ['phục hồi'])
imu_dec = node(FX, 406, 232, 44, 'edge', 'Decoder IMU', ['Haar · không skip từ encoder'], title_size=14)
imu_out = node(1392, 398, 100, 60, 'out', 'IMU', ['phục hồi'])
imu_ref = node(1160, 400, 190, 58, 'edge', 'CNN làm mượt IMU', ['1-D · dilation 1-2-4-8 · 0,65 s',
                                                              'đọc cả IMU nhiễu'], rx=8, title_size=13)
text(1490, 152, 'ViT một độ phân giải: bỏ tầng mịn encoder → NAFNet', size=11, color='#2E7D32', anchor='end',
     style='font-style="italic"')
text(1490, 167, 'predictor I-JEPA cần mask: bỏ đoán của predictor → ZI', size=11, color='#2E7D32', anchor='end',
     style='font-style="italic"')
light = node(592, 50, 250, 58, 'color', 'Nhánh ÁNH SÁNG · U-Net 64²',
             ['nhìn cả khung · đọc ZI', 'J = (ảnh − sương V) × sáng g'], rx=8, title_size=13)

# ---------------- arrows: backbone
arrow([mid_right(img_in), mid_left(qwt)]); arrow([mid_right(imu_in), mid_left(haar)])
arrow([mid_right(qwt), (370, 240)]); arrow([mid_right(haar), (370, 400)])
arrow([(570, 240), mid_left(zi)]); arrow([(570, 400), mid_left(zu)])
# ---------------- arrows: phase 2
DX = FX + 116
el.append(f'<path d="M 815 240 L 845 240 L 845 260 L 878 260 A 7 7 0 0 1 892 260 L {DX} 260" '
          f'fill="none" stroke="#8E24AA" stroke-width="2.4"/>')
arrow([(DX, 260), (DX, 232)], color='#8E24AA', width=2.4)
arrow([(DX, 260), (DX, 307)], color='#8E24AA', width=2.4)
text(962, 254, 'ZI → cả hai nhánh', size=12, weight='bold', color='#8E24AA')
arrow([(95, 200), (95, 79), (592, 79)], color='#2E7D32', width=3,
      label='skip: ảnh mờ RGB 256 × 256 — đường duy nhất mang MÀU', lx=340, ly=70)
el.append('<path d="M 842 79 L 885 79 L 885 320 L 903 320" fill="none" stroke="#EF6C00" stroke-width="3" '
          'marker-end="url(#ah-EF6C00)"/>')
text(892, 100, 'J (tuyến tính → sRGB)', size=11, weight='bold', color='#EF6C00', anchor='start')
arrow([(885, 200), (FX, 200)], color='#EF6C00', width=3)
arrow([mid_right(colb), (1272, 200), (1272, 251), (1292, 251)])
arrow([mid_right(edgb), mid_left(refine)])
arrow([mid_right(refine), (1286, 320), (1286, 269), (1292, 269)])
arrow([mid_right(join), mid_left(img_out)])
arrow([mid_right(zu), (850, 400), (850, 428), (FX, 428)])
arrow([mid_right(imu_dec), mid_left(imu_ref)])
arrow([mid_right(imu_ref), mid_left(imu_out)])

# ---------------- phase 1: I-JEPA
clean = node(20, 565, 150, 70, 'data', 'Ảnh + IMU', ['SẠCH'])
teach = node(205, 548, 180, 94, 'p1', 'Teacher EMA', ['ViT chung · cặp SẠCH', 'LayerNorm → đích', 'ở các khối đích'])
loss1 = node(420, 548, 240, 94, 'loss', 'Loss I-JEPA', ['smooth L1(đoán, đích)', 'ảnh ½ + IMU ½',
                                                       'không VICReg · neo · Jacobian'])
badge(574, 539, 'mới · I-JEPA')
pred = node(705, 548, 170, 94, 'p1', 'Predictor chung',
            [f'ViT hẹp · {P1["ijepa_predictor_dim"]} kênh · {P1["ijepa_predictor_depth"]} khối',
             'ngữ cảnh ảnh + IMU', '+ mask token ở vị trí đích'], title_size=15)
arrow([mid_right(clean), mid_left(teach)])
arrow([mid_right(teach), mid_left(loss1)], label='đích', lx=402, ly=586)
arrow([mid_left(pred), mid_right(loss1)], label='đoán', lx=683, ly=586)
# The context tokens: the same online ViT, run on the noisy pair's context tokens only.
arrow([mid_bot(zu), (760, 548)], label='ZU', lx=752, ly=500, lanchor='end')
el.append('<path d="M 815 262 L 829 262 L 829 393 A 7 7 0 0 1 829 407 L 829 548" '
          'fill="none" stroke="#455A64" stroke-width="2.2" marker-end="url(#ah)"/>')
text(823, 470, 'ZI', size=13, color='#455A64', anchor='end', style='font-style="italic"')
text(837, 482, 'token NGỮ CẢNH (ảnh + IMU) của cặp NHIỄU:', size=12, weight='bold', color='#455A64',
     anchor='start')
text(837, 497, 'ViT chung chỉ chạy trên các token này', size=12, color='#455A64', anchor='start')
# EMA: the teacher follows the online ViTs.
el.append('<path d="M 470 445 L 470 522 L 375 522 L 375 548" fill="none" stroke="#F9A825" stroke-width="2" '
          'stroke-dasharray="6 4" marker-end="url(#ah-F9A825)"/>')
text(478, 490, f'EMA: m {P1["teacher_momentum_start"]:g} → {P1["teacher_momentum_end"]:g} (tuyến tính)'.replace('.', ','),
     size=12, color='#B26A00', anchor='start', style='font-style="italic"')

# A real mask of the config: context (blue) and the target blocks (orange) on the 16 x 16 token grid.
context, targets = multiblock_masks(
    (16, 16), 1, MASK_SEED, targets=P1["ijepa_targets"], target_scale=tuple(P1["ijepa_target_scale"]),
    target_aspect=tuple(P1["ijepa_target_aspect"]), context_scale=tuple(P1["ijepa_context_scale"]),
    min_keep=P1["ijepa_image_min_keep"])
seen, hidden = set(context[0].tolist()), set(targets[0].flatten().tolist())
GX, GY, CELL = 912, 540, 6.5
for index in range(256):
    row, column = divmod(index, 16)
    fill = '#F6B26B' if index in hidden else '#9FC5E8' if index in seen else '#FFFFFF'
    el.append(f'<rect x="{GX + column * CELL}" y="{GY + row * CELL}" width="{CELL}" height="{CELL}" '
              f'fill="{fill}" stroke="#CFD8DC" stroke-width="0.5"/>')
el.append(f'<rect x="{GX}" y="{GY}" width="{16 * CELL}" height="{16 * CELL}" fill="none" stroke="#546E7A" stroke-width="1.2"/>')
for block in targets[0]:                        # each target block outlined: they may touch or overlap
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

# ---------------- phase 2 losses
loss2 = node(1150, 528, 350, 150, 'loss', 'Loss phase 2', [
    'ảnh: L1 · chi tiết QWT của Y · VGG16 (perceptual)',
    'màu: L1 Cb/Cr + thống kê màu từng ảnh',
    'nét: L1 · độ dốc · FFT phức · 128², 64² · mượt',
    'ánh sáng (H): L1 giữa J và ảnh sạch ở 32²', ''])
text(1325, 654, 'IMU: L1 · chi tiết Haar · rung · rung thừa · gia số', size=13, color='#37474F')
arrow([(1325, 528), (1325, 474)], color='#D81B60', dash='6 4', width=2)
text(1333, 498, 'train 2 decoder', size=12, color='#D81B60', anchor='start', style='font-style="italic"')
text(1333, 513, '25% cuối thêm backbone', size=12, color='#D81B60', anchor='start', style='font-style="italic"')

markers = ''.join(
    f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in ('#2E7D32', '#EF6C00', '#F9A825', '#90A4AE', '#1E88E5', '#8E24AA', '#D81B60'))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="#455A64"/></marker>{markers}</defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + ''.join(el) + '</svg>')
open(sys.argv[1], 'w', encoding='utf-8').write(svg)
