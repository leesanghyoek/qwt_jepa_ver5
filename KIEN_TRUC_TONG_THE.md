# Sơ đồ kiến trúc QWT–JEPA

![Sơ đồ kiến trúc QWT–JEPA](docs/kien_truc.svg)

**Đọc theo ba vùng màu:**

1. **① Backbone (xanh dương).** Ảnh mờ đi qua **QWT Hilbert**, IMU nhiễu đi qua **Haar**. Mỗi bên
   có một encoder CNN. Fusion trộn hai bên thành latent **ZI** (ảnh) và **ZU** (IMU).
2. **② Phase 1 (vàng, chỉ lúc train).** Teacher EMA (bản sao của hai encoder) đọc bản **sạch**
   và cho ra hai đích: **TI** cho ảnh, **TU** cho IMU. Có **hai predictor riêng**: predictor ảnh
   nhận **ZI** để đoán TI, predictor IMU nhận **ZU** để đoán TU. Mỗi predictor **nhìn lân cận**
   (ảnh: 5×5 token), và một phần token bị **che** (ảnh 30%, IMU 25%): token bị che phải đoán từ
   lân cận. Ảnh còn có đích **mịn** 32×32 (stage trước của teacher) và đích **thô** 8×8. Loss
   JEPA, cùng VICReg, decoder neo và **Jacobian**, dùng để train backbone. Jacobian ép encoder
   nhạy với đường nét và bỏ qua nhiễu.
3. **③ Phase 2 (xanh lá).** Backbone **đóng băng**. Decoder ảnh nhận ZI và **chính ảnh mờ**
   (mũi tên skip), gồm hai nhánh, mỗi nhánh là một **U-Net** (phễu–loa): **Encoder** thu nhỏ ÷2 mỗi
   tầng (đặc trưng nông → sâu), **Bottleneck** 16×16 nhận **ZI**, **Decoder** phóng ×2 trở lại, **skip
   connection** mang đặc trưng nông từ Encoder sang Decoder. Nhánh **màu** ở 128×128, nhánh **đường
   nét** trên kênh sáng Y ở 256×256, rồi **ghép**. Decoder IMU nhận ZU. Recipe p8 hiện train mỗi nhánh
   bằng CNN một tầng; U-Net bật bằng `split_branch_arch: unet`.

| | Phase 1 | Phase 2 |
|---|---|---|
| Train | backbone 1,33 M (+ 2 predictor 0,24 M, bỏ sau phase 1) | decoder ảnh 0,79 M + decoder IMU 0,36 M |
| Loss | JEPA (+ mịn 0,5, thô 0,25) · VICReg (covariance gộp) · decoder neo 0,45 · Jacobian 0,05 | L1 ảnh · chi tiết QWT · màu · đường nét · độ dốc cạnh · IMU |

Chi tiết từng lớp: [README](README.md#kiến-trúc-chi-tiết). Vẽ lại hình:
`python3 tools/draw_architecture.py docs/kien_truc.svg`.
