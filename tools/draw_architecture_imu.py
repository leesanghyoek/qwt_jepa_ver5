"""Architecture figure of the p16_imu_smooth run (qwt-jaco-jepa-imu.ipynb), as plain SVG.

p16's figure -- the image side of this run is p16's recipe -- plus what the run adds on
the IMU: (B) the 1-D refiner after the IMU decoder and (C) the excess-jitter loss term,
both tagged "mới"; with --fourier, also (D) the wavelet-Fourier blocks in the NAFNet edge
branch (the p16_fourier_imu run, qwt-jaco-jepa-fourier.ipynb). Kept apart from draw_architecture.py, which follows the shared
notebook's recipe, so neither run's figure moves when the other changes.

With --gray, the current recipe (p20_gray, configs/kaggle_gray.yaml): p16_infomax plus
p19_sharp's changes (a, b, c, e, f) plus (G) the backbone reading luminance only, the
colour reaching phase 2 through the blurry-image skip alone. B-F are no longer tagged
there: they are part of the architecture, and only G is new.

With --light, p22_light (configs/kaggle_light.yaml): p20_gray plus (H) the light branch in
phase 2 -- a whole-frame U-Net at 64x64 that takes the glare veil off and lifts the dark in
linear light before the colour and edge branches -- and its low-frequency loss. Only H is
tagged there.

Usage: python3 tools/draw_architecture_imu.py docs/kien_truc_imu.svg
       python3 tools/draw_architecture_imu.py docs/kien_truc_infomax.svg --infomax   (D + E + F)
       python3 tools/draw_architecture_imu.py docs/kien_truc_fourier.svg --fourier
       python3 tools/draw_architecture_imu.py docs/kien_truc_gray.svg --gray         (p19 + G)
       python3 tools/draw_architecture_imu.py docs/kien_truc_light.svg --light       (p20 + H)
       python3 tools/draw_architecture_imu.py docs/kien_truc_illum.svg --illum       (p22, nhiễu ánh sáng không đều)
       python3 tools/draw_architecture_imu.py docs/kien_truc_env.svg --env           (p22, nhiễu môi trường)
"""
import sys
from xml.sax.saxutils import escape

ENV = '--env' in sys.argv[2:]              # p24_env: p23's architecture, poor environment (fog) in the corruption
ILLUM = '--illum' in sys.argv[2:] or ENV   # p23_illum: p22_light's architecture, uneven light in the corruption
LIGHT = '--light' in sys.argv[2:] or ILLUM # p22_light: p20_gray + light branch in phase 2 (H)
GRAY = '--gray' in sys.argv[2:] or LIGHT   # p20_gray: p19_sharp + QWT/JEPA on luminance only (G)
INFOMAX = '--infomax' in sys.argv[2:] or GRAY  # p16_infomax: (E) phase-1 information terms, (F) stages
FOURIER = '--fourier' in sys.argv[2:] or INFOMAX

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
    if ILLUM or (LIGHT and label != 'mới · H') or (GRAY and not LIGHT and label != 'mới · G'):
        return                                  # p23 changes only the corruption: nothing in the figure is new
    width = 12 + 7 * len(label)
    el.append(f'<rect x="{x}" y="{y}" width="{width}" height="18" rx="9" fill="#D81B60"/>')
    text(x + width / 2, y + 13, label, size=11, weight='bold', color='#FFFFFF')

def mid_right(b): x, y, w, h = b; return (x + w, y + h / 2)
def mid_left(b):  x, y, w, h = b; return (x, y + h / 2)
def mid_top(b):   x, y, w, h = b; return (x + w / 2, y)
def mid_bot(b):   x, y, w, h = b; return (x + w / 2, y + h)

# ---------------- regions
region(190, 150, 490, 320, 'bb', '① Backbone — train ở phase 1, đóng băng ở phase 2')
region(842, 110, 663, 360, 'edge', '③ Phase 2 — decoder khôi phục', right=True)
region(190, 505, 810, 200, 'p1', '② Phase 1 — học latent (chỉ lúc train)')

# ---------------- inputs
img_in = node(20, 200, 150, 80, 'data', 'Ảnh mờ + tối', ['3 × 256 × 256', 'sáng/tối bất ổn' if ENV
                                                         else 'sáng/tối không đều' if ILLUM
                                                         else 'đèn lóe nhẹ' if LIGHT else 'nhiễu B'])
imu_in = node(20, 365, 150, 70, 'data', 'IMU nhiễu', ['6 × 128'])
# ---------------- backbone
qwt = node(210, 205, 135, 70, 'tf', 'QWT Hilbert', ['kênh sáng Y', '16 × 128 × 128'] if GRAY
           else ['48 × 128 × 128'])
haar = node(210, 365, 135, 70, 'tf', 'Haar', ['12 × 64'])
enc_i = node(370, 205, 135, 70, 'bb', 'Encoder ảnh', ['CNN 4 stage', 'centre norm (b)'] if GRAY
             else ['CNN 4 stage'])
enc_u = node(370, 365, 135, 70, 'bb', 'Encoder IMU', ['CNN 4 stage'])
fus = node(535, 280, 125, 80, 'bb', 'Fusion', ['có cổng'])
# ---------------- latent
zi = node(705, 205, 110, 70, 'lat', 'ZI', ['128 × 16 × 16'])
zu = node(705, 365, 110, 70, 'lat', 'ZU', ['128 × 8'])
# ---------------- phase 2: image decoder = colour ResNet + edge U-Net (encoder / bottleneck / decoder)
def unet_block(x, yc, kind, title, below=False):
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
    k = 1 if below else -1                                      # skip arc and title above or below
    el.append(f'<path d="M {x + 40} {yc + 24 * k} C {x + 56} {yc + 54 * k}, {x + 176} {yc + 54 * k}, {x + 192} {yc + 24 * k}" '
              f'fill="none" stroke="#90A4AE" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ah-90A4AE)"/>')
    text(x + 116, yc + (53 if below else -45), 'skip connection', size=10, color='#78909C', style='font-style="italic"')
    text(x + 116, yc + (71 if below else -59), title, size=13, weight='bold', color=stroke)
    return (x, yc - 30, 232, 60)

el.append('<rect x="860" y="125" width="285" height="272" rx="10" fill="#FFFFFF" fill-opacity="0.7" stroke="#2E7D32" stroke-width="1.5"/>')
FX = 905                                                       # funnel x of both branches
colb = node(FX, 168, 232, 64, 'color', 'Nhánh MÀU · ResNet 128²', ['tone/màu cả ảnh → từng vùng'], title_size=14)
edgb = unet_block(FX, 320, 'edge', 'Nhánh ĐƯỜNG NÉT (Y) · NAFNet + 4 WF' if GRAY
                 else 'Nhánh ĐƯỜNG NÉT · NAFNet + 4 WF' if FOURIER
                 else 'Nhánh ĐƯỜNG NÉT · NAFNet 30 khối', below=True)
if FOURIER:
    # (D) a wavelet-Fourier block after the encoder and the decoder stage at 256² and 128²:
    # the wide ends of the two trapezoids
    for x in (FX + 4, FX + 198):
        el.append(f'<rect x="{x}" y="266" width="30" height="17" rx="4" fill="#FCE4EC" '
                  f'stroke="#D81B60" stroke-width="1.6"/>')
        text(x + 15, 279, 'WF', size=10, weight='bold', color='#D81B60')
        arrow([(x + 15, 283), (x + 15, 296)], color='#D81B60', width=1.4)
    badge(FX + 236, 263, 'mới · D')
# p15: small full-resolution CNN on the edge map only (colour already split off)
refine = node(1160, 296, 118, 48, 'edge', 'CNN làm nét', ['+ mượt · 4 khối'], rx=8, title_size=13)
join = node(1292, 231, 58, 58, 'out', 'Ghép', [], rx=29, title_size=14)
img_out = node(1392, 221, 100, 78, 'out', 'Ảnh', ['phục hồi'])
imu_dec = node(FX, 406, 232, 44, 'edge', 'Decoder IMU', ['Haar · skip có cổng từ encoder IMU'], title_size=14)
imu_out = node(1392, 398, 100, 60, 'out', 'IMU', ['phục hồi'])
# (B) 1-D CNN after the IMU decoder: reads the restored and the noisy IMU, sees 65 samples
imu_ref = node(1160, 400, 190, 58, 'edge', 'CNN làm mượt IMU', ['1-D · dilation 1-2-4-8 · 0,65 s',
                                                              'đọc cả IMU nhiễu'], rx=8, title_size=13)
badge(1290, 380, 'mới · B')
if ENV:
    text(W - 20, 26, 'p24_env — môi trường chụp kém: thiếu sáng, vùng sáng/tối bất ổn, nhòe tối, lóe · nhánh ÁNH SÁNG (H)',
         size=15, weight='bold', color='#D81B60', anchor='end')
    text(W - 20, 44, 'kiến trúc như p22_light: H trừ lớp sương V, nhân sáng g theo vị trí · 20% ảnh môi trường trong'
         ' · ít mờ chuyển động', size=12, color='#AD1457', anchor='end')
elif ILLUM:
    text(W - 20, 26, 'p23_illum — nhiễu: sáng/tối ngẫu nhiên theo vùng, nhòe tối, lóe sáng · nhánh ÁNH SÁNG (H) cân bằng lại',
         size=15, weight='bold', color='#D81B60', anchor='end')
    text(W - 20, 44, 'kiến trúc như p22_light: H trước 2 nhánh (trừ sương V, nhân sáng g theo vị trí) · nền p20_gray (G)'
         ' + p19_sharp (a–f) · phase 2: 4000 update', size=12, color='#AD1457', anchor='end')
elif LIGHT:
    text(W - 20, 26, 'p22_light — nhánh ÁNH SÁNG (H) trước 2 nhánh: trừ lớp sương lóe, làm sáng chỗ tối, trên cả khung',
         size=15, weight='bold', color='#D81B60', anchor='end')
    text(W - 20, 44, 'nền p20_gray: Y vào backbone (G) · p19_sharp (a–f) · nhiễu: đèn ×4–20, trời không lóe'
         ' · phase 2: 4000 update', size=12, color='#AD1457', anchor='end')
elif GRAY:
    text(W - 20, 26, 'p20_gray — QWT, encoder và teacher chỉ thấy kênh sáng Y (G) · màu đi đường skip, ghép lại ở phase 2',
         size=15, weight='bold', color='#D81B60', anchor='end')
    text(W - 20, 44, 'nền p19_sharp: Jacobian hạt nhiễu (a) · centre norm (b) · đầu hư hỏng (c) · đoán TI → decoder (e)'
         ' · LP-FT (f)   ·   p16_infomax: WF (D) · log-det, InfoNCE (E) · tầng mịn (F) · IMU: B, C',
         size=12, color='#AD1457', anchor='end')
elif INFOMAX:
    text(W - 20, 30, 'p16_infomax — phase 1 ép latent chứa nhiều hơn (E) · tầng mịn JEPA vào NAFNet (F) · WF (D) · IMU: B, C',
         size=15, weight='bold', color='#D81B60', anchor='end')
    text(W - 20, 50, 'E: coding rate (log-det) · InfoNCE dày đặc · sàn độ nhạy chi tiết 2–4 px · đích JEPA 64²'
         '   ·   phase 2: 3000 update', size=12, color='#AD1457', anchor='end')
elif FOURIER:
    text(W - 20, 30, 'p16_fourier_imu — ảnh: p16 + khối wavelet–Fourier (D) · IMU: CNN làm mượt (B) + rung thừa (C)',
         size=15, weight='bold', color='#D81B60', anchor='end')
    text(W - 20, 50, 'WF: LayerNorm → 1×1 → tách Haar → FFT toàn ảnh → 1×1 · ReLU · 1×1 → iFFT → ghép Haar → 1×1'
         ' · sau stage encoder / decoder ở 256² và 128²', size=12, color='#AD1457', anchor='end')
else:
    text(W - 20, 34, 'p16_imu_smooth — ảnh như p16 · IMU: CNN làm mượt (B) + loss rung thừa (C)',
         size=15, weight='bold', color='#D81B60', anchor='end')
# ---------------- phase 1
clean = node(20, 565, 150, 70, 'data', 'Ảnh + IMU', ['SẠCH'])
teach = node(205, 555, 135, 80, 'p1', 'Teacher EMA', ['2 encoder · Y SẠCH' if GRAY else '2 encoder · đọc SẠCH',
                                                      'đích 16² · 32² · 64²' if INFOMAX else 'đích 16² + mịn 32²'])
if GRAY:
    loss1 = node(420, 546, 230, 112, 'loss', 'Loss phase 1', ['JEPA ảnh+IMU · mịn · thô · 64²',
                                                            'VICReg gộp · neo Y · Jacobian (a)',
                                                            'log-det · InfoNCE · sàn chi tiết',
                                                            'đầu hư hỏng từ ZI (c)'])
elif INFOMAX:
    loss1 = node(420, 546, 230, 98, 'loss', 'Loss phase 1', ['JEPA ảnh+IMU · mịn · thô · 64²',
                                                           'VICReg gộp · neo · Jacobian',
                                                           'log-det · InfoNCE · sàn chi tiết'])
    badge(592, 537, 'mới · E')
else:
    loss1 = node(420, 550, 230, 90, 'loss', 'Loss phase 1', ['JEPA ảnh + IMU · mịn · thô', 'VICReg gộp · neo · Jacobian'])
pred_u = node(705, 555, 110, 70, 'p1', 'Predictor', ['IMU · lân cận', 'che 25% token'])
pred_i = node(862, 555, 115, 84 if GRAY else 70, 'p1', 'Predictor',
              ['ảnh · lân cận 5×5', 'che 30% token'] + (['đọc hư hỏng (c)'] if GRAY else []))

# ---------------- arrows: backbone
arrow([mid_right(img_in), mid_left(qwt)]); arrow([mid_right(imu_in), mid_left(haar)])
arrow([mid_right(qwt), mid_left(enc_i)]); arrow([mid_right(haar), mid_left(enc_u)])
arrow([mid_right(enc_i), (520, 240), (520, 305), (535, 305)])
arrow([mid_right(enc_u), (520, 400), (520, 335), (535, 335)])
arrow([(660, 305), (685, 305), (685, 240), (705, 240)])
arrow([(660, 335), (685, 335), (685, 400), (705, 400)])
# ---------------- arrows: phase 2
# ZI into the bottom of both branches (hop over the blurry-image bus at x = 885)
DX = FX + 116
el.append(f'<path d="M 815 240 L 845 240 L 845 260 L 878 260 A 7 7 0 0 1 892 260 L {DX} 260" '
          f'fill="none" stroke="#8E24AA" stroke-width="2.4"/>')
arrow([(DX, 260), (DX, 232)], color='#8E24AA', width=2.4)
arrow([(DX, 260), (DX, 307)], color='#8E24AA', width=2.4)
text(962, 254, 'ZI → cả hai nhánh', size=12, weight='bold', color='#8E24AA')
# the blurry image into the funnel of both branches
if LIGHT:
    # (H) the light branch on the blurry-image bus: both branches read its corrected frame J
    light = node(592, 50, 250, 58, 'color', 'Nhánh ÁNH SÁNG · U-Net 64²',
                 ['nhìn cả khung · đọc ZI', 'J = (ảnh − sương V) × sáng g'], rx=8, title_size=13)
    badge(760, 31, 'mới · H')
    arrow([(95, 200), (95, 79), (592, 79)], color='#2E7D32', width=3,
          label='skip: ảnh mờ RGB 256 × 256 — đường duy nhất mang MÀU', lx=340, ly=70)
    el.append('<path d="M 842 79 L 885 79 L 885 320 L 903 320" fill="none" stroke="#EF6C00" stroke-width="3" '
              'marker-end="url(#ah-EF6C00)"/>')
    text(892, 100, 'J (tuyến tính → sRGB)', size=11, weight='bold', color='#EF6C00', anchor='start')
else:
    arrow([(95, 200), (95, 72), (885, 72), (885, 320), (FX, 320)], color='#2E7D32', width=3,
          label='skip: chính ảnh mờ RGB 256 × 256 — đường duy nhất mang MÀU, QWT chỉ thấy Y' if GRAY
          else 'skip: chính ảnh mờ 256 × 256 → cho biết cạnh nằm ở đâu', lx=520, ly=63)
arrow([(885, 200), (FX, 200)], color='#EF6C00' if LIGHT else '#2E7D32', width=3)
arrow([mid_right(colb), (1272, 200), (1272, 251), (1292, 251)])
arrow([mid_right(edgb), mid_left(refine)])
arrow([mid_right(refine), (1286, 320), (1286, 269), (1292, 269)])
arrow([mid_right(join), mid_left(img_out)])
arrow([mid_right(zu), (850, 400), (850, 428), (FX, 428)])
if INFOMAX:
    # (F) the image encoder's finer stages (1/2, 1/4, 1/8 of the frame) into NAFNet at the same size;
    # over the backbone, down just left of the edge branch, in under the blurry-image skip
    el.append('<path d="M 480 205 L 480 190 L 585 190 L 585 140 L 857 140 L 857 253 A 7 7 0 0 1 857 267 '
              'L 857 340 L 903 340" '
              'fill="none" stroke="#8E24AA" stroke-width="2.2" stroke-dasharray="7 4" marker-end="url(#ah-8E24AA)"/>')
    text(690, 133, 'tầng mịn encoder JEPA (1/2 · 1/4 · 1/8 khung) → NAFNet cùng cỡ', size=12,
         weight='bold', color='#8E24AA')
    badge(884, 124, 'mới · F')
if GRAY:
    # (G) colour split off before the QWT: only luminance enters the backbone
    el.append('<circle cx="190" cy="240" r="13" fill="#E8EAF6" stroke="#3949AB" stroke-width="2"/>')
    text(190, 245, 'Y', size=14, weight='bold', color='#3949AB')
    text(190, 196, 'tách màu', size=11, weight='bold', color='#3949AB')
    badge(284, 186, 'mới · G')
    # (e) the frozen phase-1 image predictor's clean-latent guess joins ZI before both branches
    el.append('<path d="M 862 614 L 838 614 L 838 407 A 7 7 0 0 1 838 393 L 838 245" fill="none" '
              'stroke="#8E24AA" stroke-width="2.2" stroke-dasharray="7 4" marker-end="url(#ah-8E24AA)"/>')
    text(845, 489, 'e: đoán TI của predictor (đóng băng) cộng vào ZI', size=12, weight='bold', color='#8E24AA',
         anchor='start')
arrow([mid_right(imu_dec), mid_left(imu_ref)])
arrow([mid_right(imu_ref), mid_left(imu_out)])
# ---------------- arrows: phase 1
arrow([mid_right(clean), mid_left(teach)])
arrow([mid_right(teach), (420, 595)], label='đích TI, TU', lx=380, ly=585)
arrow([mid_bot(zu), mid_top(pred_u)], label='ZU', lx=772, ly=500, lanchor='start')
el.append('<path d="M 815 262 L 829 262 L 829 393 A 7 7 0 0 1 829 407 L 829 590 L 862 590" '
          'fill="none" stroke="#455A64" stroke-width="2.2" marker-end="url(#ah)"/>')
text(823 if GRAY else 835, 500, 'ZI', size=13, color='#455A64', anchor='end' if GRAY else 'start',
     style='font-style="italic"')
arrow([mid_left(pred_u), (650, 590)], label='đoán TU', lx=678, ly=582)
arrow([mid_bot(pred_i), (919, 672), (535, 672), (535, loss1[1] + loss1[3] if GRAY else 640)], label='đoán TI', lx=740, ly=665)
arrow([(490, loss1[1] if GRAY else 555), (490, 435)], color='#F9A825', dash='6 4', width=2)
text(498, 500, 'Jacobian: nhạy cạnh, điếc hạt nhiễu (a)' if GRAY else 'Jacobian: nhạy với cạnh, điếc với nhiễu',
     size=12, color='#B26A00', anchor='start', style='font-style="italic"')
# ---------------- phase 2 losses (train only the two decoders)
loss2 = node(1150, 528, 350, 150 if LIGHT else 132, 'loss', 'Loss phase 2', [
    'ảnh: L1 · chi tiết QWT của Y · VGG16 (perceptual)' if GRAY else 'ảnh: L1 · chi tiết QWT · VGG16 (perceptual)',
    'màu: L1 Cb/Cr + thống kê màu từng ảnh',
    'nét: L1 · độ dốc · FFT phức · 128², 64² · mượt'] + (
    ['ánh sáng (H): L1 giữa J và ảnh sạch ở 32²', ''] if LIGHT else ['']))
# the IMU line, with this run's new term in its colour
if GRAY:
    text(1325, 654 if LIGHT else 636, 'IMU: L1 · chi tiết Haar · rung · rung thừa · gia số', size=13, color='#37474F')
else:
    el.append('<text x="1325" y="636" font-size="13" fill="#37474F" text-anchor="middle">'
              'IMU: L1 · chi tiết Haar · độ rung · '
              '<tspan fill="#D81B60" font-weight="bold">rung thừa</tspan></text>')
badge(1438, 520, 'mới · C')
arrow([(1325, 528), (1325, 474)], color='#D81B60', dash='6 4', width=2)
if GRAY:
    text(1317, 513, 'train 2 decoder · 25% cuối thêm backbone', size=12, color='#D81B60', anchor='end',
         style='font-style="italic"')
else:
    text(1333, 505, 'chỉ train 2 decoder', size=12, color='#D81B60', anchor='start', style='font-style="italic"')

markers = ''.join(
    f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in ('#2E7D32', '#EF6C00', '#F9A825', '#90A4AE', '#1E88E5', '#8E24AA', '#D81B60'))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="#455A64"/></marker>{markers}</defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + ''.join(el) + '</svg>')
open(sys.argv[1], 'w', encoding='utf-8').write(svg)
