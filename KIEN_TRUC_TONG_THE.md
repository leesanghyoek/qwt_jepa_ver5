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
   (mũi tên skip), gồm hai nhánh rồi **ghép**. Nhánh **màu** là ResNet ở 128×128 (như p8). Nhánh
   **đường nét** (kênh sáng Y, 256×256) là một **U-Net** 5 tầng: **Encoder** thu nhỏ ÷2 mỗi tầng, từ
   **đặc trưng nhỏ** (cạnh, vật nhỏ) tới **đặc trưng tổng thể** (bố cục, vật là gì, độ sáng chung);
   **Bottleneck** 16×16 nhận **ZI**; **Decoder** phóng ×2 trở lại; **skip connection** mang đặc trưng
   nhỏ từ Encoder sang Decoder (`split_branch_arch: unet_edge`, từ p11). Decoder IMU nhận ZU.

| | Phase 1 | Phase 2 |
|---|---|---|
| Train | backbone 1,33 M (+ 2 predictor 0,24 M, bỏ sau phase 1) | decoder ảnh 3,02 M (màu 0,14 M + đường nét U-Net 2,88 M) + decoder IMU 0,36 M |
| Loss | JEPA (+ mịn 0,5, thô 0,25) · VICReg (covariance gộp) · decoder neo 0,45 · Jacobian 0,05 | L1 ảnh · chi tiết QWT · màu · đường nét · độ dốc cạnh · IMU |

Chi tiết từng lớp: [README](README.md#kiến-trúc-chi-tiết). Vẽ lại hình:
`python3 tools/draw_architecture.py docs/kien_truc.svg`.
