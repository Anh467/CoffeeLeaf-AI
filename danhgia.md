# Đánh giá So sánh Thực nghiệm: EfficientNetV2-S vs RegNetY-3.2GF trong Nhận diện Bệnh hại Lá Cà phê

## 1. Tóm tắt

Hai kiến trúc mạng nơ-ron tích chập được huấn luyện theo phương pháp Transfer Learning (tinh chỉnh trên trọng số pretrained ImageNet) để phân loại 4 trạng thái lá cà phê: **Healthy, Miner, Phoma, Rust**. Cả hai mô hình dùng chung một pipeline dữ liệu (Stratified Split, Data Augmentation), cùng chiến lược tối ưu hóa (AdamW, Differential Learning Rate, Cosine Annealing, Label Smoothing, AMP) và được đánh giá độc lập trên tập Test 400 ảnh (100 ảnh/lớp) chưa từng xuất hiện trong quá trình huấn luyện.

**Kết luận nhanh:** EfficientNetV2-S vượt trội về độ chính xác tổng thể (95.25% so với 90.25%) và đặc biệt vượt trội trong việc phát hiện lớp Rust — điểm yếu chung của cả hai kiến trúc. Đổi lại, RegNetY-3.2GF nhẹ hơn, suy luận nhanh hơn và tiêu thụ ít VRAM hơn. Lựa chọn cuối cùng phụ thuộc vào việc ưu tiên độ chính xác chẩn đoán hay hiệu năng triển khai (xem Mục 5).

---

## 2. Bảng So sánh Tổng hợp

| Chỉ số | EfficientNetV2-S | RegNetY-3.2GF | Chênh lệch |
|---|---:|---:|---|
| **Test Accuracy (%)** | **95.25** | 90.25 | EfficientNetV2-S cao hơn 5.00 điểm % |
| Precision (Macro) | **0.9569** | 0.9159 | +0.0410 |
| Recall (Macro) | **0.9525** | 0.9025 | +0.0500 |
| F1-Score (Macro) | **0.9521** | 0.9002 | +0.0519 |
| Tổng số tham số (M) | 20.18 | **17.93** | RegNetY nhẹ hơn 11.2% |
| Tham số huấn luyện được (M) | 18.36 (91.0%) | 4.81 (26.8%) | Chiến lược fine-tune khác biệt rõ rệt |
| Dung lượng file trọng số (MB) | 77.86 | **68.83** | RegNetY nhỏ hơn 11.6% |
| Latency (ms/ảnh) | 15.46 | **14.61** | RegNetY nhanh hơn 5.5% |
| Throughput (FPS) | 64.7 | **68.4** | RegNetY cao hơn 5.7% |
| Max VRAM suy luận (MB) | 400.3 | **324.7** | RegNetY tiết kiệm hơn 18.9% |
| Thời gian huấn luyện (s) | 423.7 (~7.1 phút) | **379.1 (~6.3 phút)** | RegNetY nhanh hơn 10.5% |
| Delta Loss (Val − Train) | **0.0020** | 0.0172 | EfficientNetV2-S tổng quát hóa tốt hơn ~8.6 lần |

*(Nguồn: `model_comparison_summary.csv`, đo trên GPU NVIDIA RTX 5060 8GB, batch size 32, ảnh 300×300 cho EfficientNetV2-S và 224×224 cho RegNetY-3.2GF.)*

---

## 3. Phân tích Độ chính xác theo Từng lớp bệnh

### 3.1. EfficientNetV2-S

| Lớp | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Healthy | 0.9709 | **1.0000** | 0.9852 |
| Miner | 0.8772 | **1.0000** | 0.9346 |
| Phoma | 0.9796 | 0.9600 | 0.9697 |
| Rust | **1.0000** | 0.8500 | 0.9189 |

**Ma trận nhầm lẫn:** Healthy và Miner được nhận diện tuyệt đối chính xác (Recall 100%). Điểm yếu duy nhất nằm ở lớp **Rust** (Recall 85%): trong 100 ảnh Rust, 10 ảnh bị nhận nhầm thành Miner, 3 ảnh thành Healthy, 2 ảnh thành Phoma. Đây là lớp khó nhất đối với cả hai mô hình (xem 3.3).

### 3.2. RegNetY-3.2GF

| Lớp | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Healthy | 0.9320 | 0.9600 | 0.9458 |
| Miner | 0.7857 | **0.9900** | 0.8761 |
| Phoma | 0.9600 | 0.9600 | 0.9600 |
| Rust | **0.9859** | **0.7000** | 0.8187 |

**Ma trận nhầm lẫn:** RegNetY-3.2GF bộc lộ điểm yếu nghiêm trọng hơn ở lớp **Rust** — chỉ nhận đúng 70/100 ảnh, trong đó **24 ảnh (24%) bị nhận nhầm thành Miner**, 6 ảnh thành Healthy. Recall Miner đạt 99% nhưng Precision Miner chỉ 78.57% — hệ quả trực tiếp của việc mô hình có xu hướng "đổ dồn" các ca Rust khó phân biệt về nhãn Miner.

### 3.3. Điểm chung: Rust và Miner dễ gây nhầm lẫn

Cả hai kiến trúc đều thể hiện cùng một khuynh hướng nhầm lẫn: **Rust → Miner**. Điều này gợi ý rằng đặc trưng thị giác của các đốm rỉ sắt (Rust) ở một số giai đoạn phát triển có độ tương đồng cao với các vết đục do sâu vẽ bùa (Miner) — đây là hạn chế mang tính dữ liệu/đặc trưng hơn là lỗi huấn luyện. Tuy nhiên mức độ ảnh hưởng khác biệt rõ rệt: EfficientNetV2-S kiểm soát nhầm lẫn này tốt hơn nhiều (10% so với 24%), nhiều khả năng nhờ:

- Số tham số **được huấn luyện lại thực tế** cao hơn hẳn (18.36M so với 4.81M) — EfficientNetV2-S điều chỉnh tới ~91% mạng, trong khi RegNetY-3.2GF chỉ điều chỉnh ~27%, hạn chế khả năng học đặc trưng chuyên biệt cho bài toán.
- Độ phân giải ảnh đầu vào lớn hơn (300×300 so với 224×224), giữ lại nhiều chi tiết texture hơn — yếu tố quan trọng để phân biệt các dạng tổn thương lá có kích thước nhỏ như đốm rỉ sắt.

---

## 4. Phân tích Hiệu năng Phần cứng & Khả năng Tổng quát hóa

### 4.1. Tốc độ và tài nguyên

RegNetY-3.2GF chiếm ưu thế toàn diện về mặt hiệu năng vận hành:

- **Latency thấp hơn 5.5%** (14.61ms so với 15.46ms/ảnh) và **Throughput cao hơn 5.7%** (68.4 so với 64.7 FPS).
- **Tiết kiệm VRAM khi suy luận tới 18.9%** (324.7MB so với 400.3MB) — đáng kể nếu triển khai đồng thời nhiều tiến trình suy luận trên cùng một GPU.
- **Huấn luyện nhanh hơn 10.5%** (379.1s so với 423.7s cho 15 epoch) và **nhẹ hơn trên đĩa** (68.83MB so với 77.86MB).

Khoảng cách hiệu năng này chủ yếu đến từ độ phân giải ảnh đầu vào nhỏ hơn (224² so với 300², tương đương ~45% ít pixel hơn phải xử lý mỗi ảnh) chứ không hoàn toàn phản ánh sự khác biệt về độ phức tạp kiến trúc thuần túy.

### 4.2. Khả năng tổng quát hóa (Generalization Gap)

Chỉ số **Delta Loss** (chênh lệch |Val Loss − Train Loss| ở epoch cuối) cho thấy EfficientNetV2-S hội tụ ổn định và cân bằng hơn nhiều: **0.0020** so với **0.0172** của RegNetY-3.2GF (gấp ~8.6 lần). Quan sát này khớp với đường cong Loss/Accuracy: cả hai mô hình đều không có dấu hiệu overfitting nghiêm trọng (Val Acc bám sát hoặc vượt nhẹ Train Acc trong phần lớn quá trình huấn luyện), nhưng đường cong của EfficientNetV2-S mượt và hội tụ chặt hơn về cuối chu kỳ huấn luyện.

---

## 5. Kết luận & Khuyến nghị

| Tiêu chí ưu tiên | Lựa chọn khuyến nghị |
|---|---|
| **Độ chính xác chẩn đoán là ưu tiên hàng đầu** (ứng dụng thực tế hỗ trợ nông dân/chuyên gia ra quyết định) | **EfficientNetV2-S** — vượt trội 5 điểm % Accuracy, F1 Macro cao hơn 0.052, đặc biệt kiểm soát tốt hơn hẳn lớp Rust (Recall 85% so với 70%) — hạn chế bỏ sót ca bệnh (false negative), yếu tố quan trọng trong nông nghiệp chính xác |
| **Triển khai trên thiết bị biên (edge/mobile), tài nguyên hạn chế, cần xử lý số lượng lớn ảnh/giây** | **RegNetY-3.2GF** — nhẹ hơn, nhanh hơn, tiết kiệm VRAM hơn ~19%, đánh đổi bằng độ chính xác thấp hơn đáng kể và rủi ro bỏ sót ca Rust cao (30% Rust bị chẩn đoán sai) |

**Khuyến nghị chung:** Với bài toán chẩn đoán bệnh hại cây trồng — nơi chi phí của một lần bỏ sót bệnh (false negative) thường cao hơn nhiều so với chi phí tính toán tăng thêm — **EfficientNetV2-S là lựa chọn phù hợp hơn** cho hầu hết kịch bản triển khai thực tế, kể cả khi phải đánh đổi ~11% tham số, ~6% latency và ~19% VRAM. RegNetY-3.2GF chỉ nên được cân nhắc trong các kịch bản triển khai biên nghiêm ngặt về phần cứng, đi kèm khuyến nghị bổ sung dữ liệu huấn luyện hoặc mở băng (unfreeze) nhiều lớp hơn để cải thiện Recall của lớp Rust trước khi đưa vào sản xuất.

---

## 6. Phụ lục: Tệp kết quả liên quan

- `model_comparison_summary.csv` — bảng số liệu thô
- `classification_report_efficientnetv2_s.txt`, `classification_report_regnety_3.2gf.txt` — báo cáo chi tiết theo lớp
- `training_curves.png` — biểu đồ Loss/Accuracy qua 15 epoch
- `confusion_matrices.png` — ma trận nhầm lẫn trên tập Test
- `best_efficientnetv2_s_model.pth`, `best_regnety_3.2gf_model.pth` — trọng số tốt nhất của từng mô hình
