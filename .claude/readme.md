{
"thong_tin_du_an": {
"ten_de_tai": "Nghiên cứu so sánh hiệu năng giữa EfficientNetV2-S và RegNetY-4GF trong bài toán nhận diện bệnh hại lá cây cà phê",
"ten_tieng_anh": "A Comparative Study of EfficientNetV2-S and RegNetY-4GF for Coffee Leaf Disease Identification",
"linh_vuc": "Nông nghiệp thông minh / Thị giác máy tính",
"framework": "PyTorch",
"moi_truong_phan_cung": {
"cpu": "Intel Core Ultra 5 225",
"gpu": "NVIDIA GeForce RTX 5060 8GB VRAM",
"cong_nghe_tang_toc": "Automatic Mixed Precision (AMP FP16/FP32)"
}
},
"vai_tro_agent": {
"chuc_danh": "Chuyên gia Nghiên cứu AI và Thị giác Máy tính trong Nông nghiệp Thông minh",
"chuyen_mon": [
"Học sâu (Deep Learning)",
"Mạng Nơ-ron Tích phân (CNN)",
"Học chuyển giao & Tinh chỉnh mô hình (Transfer Learning & Fine-Tuning)",
"Tối ưu hóa & Đo đạc hiệu năng mô hình (Model Optimization & Benchmarking)"
],
"vong_phong_xuat_ban": "Học thuật, chặt chẽ, chuẩn mực IEEE/Springer, chính xác về thuật ngữ chuyên ngành"
},
"boi_canh": {
"vấn_de_nghien_cuu": "Chẩn đoán sớm các bệnh hại lá cà phê (như Rỉ sắt, Đốm mắt cua, Nấm hồng...) là yếu tố cốt lõi trong nông nghiệp chính xác. Bài toán đòi hỏi sự cân bằng giữa độ chính xác chẩn đoán và hiệu năng vận hành trên thiết bị phần cứng.",
"muc_tieu": "Thực hiện phân tích thực nghiệm so sánh đối đầu giữa hai kiến trúc EfficientNetV2-S (Google) và RegNetY-4GF (Meta AI) trên cả hai khía cạnh: Độ chính xác chẩn đoán và Hiệu năng tiêu thụ tài nguyên phần cứng.",
"tap_du_lieu": {
"ten": "Tập dữ liệu bệnh hại lá cà phê (Coffee Leaf Disease Dataset)",
"so_luong_lop_benh": 5,
"phuong_phap_chia": "Phân chia ngẫu nhiên phân tầng (Stratified Random Sampling)",
"ty_le_chia": {
"tap_train": 0.70,
"tap_val": 0.15,
"tap_test": 0.15
}
}
},
"quy_trinh_huan_luyen_tung_buoc": [
{
"buoc_so": 1,
"ten_buoc": "Pipeline Dữ liệu & Tăng cường Dữ liệu (Data Augmentation)",
"cac_thao_tac": [
"Phân chia tập dữ liệu bằng phương pháp Lấy mẫu Phân tầng để giữ nguyên phân phối xác suất các lớp nhãn P(Y).",
"Áp dụng biến đổi hình học: Lật ngang ngẫu nhiên (p=0.5), Lật dọc ngẫu nhiên (p=0.5), Xoay ảnh ngẫu nhiên (-20 đến +20 độ).",
"Áp dụng biến đổi không gian màu: ColorJitter (độ sáng=0.2, độ tương phản=0.2, độ bão hòa=0.2) nhằm tránh mô hình học vẹt màu nền đất/bóng râm.",
"Đưa ảnh về kích thước chuẩn: EfficientNetV2-S (300x300), RegNetY-4GF (224x224).",
"Chuẩn hóa dữ liệu theo giá trị trung bình và độ lệch chuẩn của ImageNet: Mean [0.485, 0.456, 0.406] và Std [0.229, 0.224, 0.225]."
]
},
{
"buoc_so": 2,
"ten_buoc": "Thích ứng Cấu trúc Mô hình & Transfer Learning",
"cac_thao_tac": [
"Nạp trọng số huấn luyện trước (Pre-trained weights) từ tập dữ liệu ImageNet.",
"Chiến lược cho EfficientNetV2-S: Đóng băng (Freeze) các lớp trích xuất đặc trưng đầu, mở băng (Unfreeze) 2 khối Fused-MBConv cuối, thay thế lớp phân loại bằng Dropout(p=0.4) + Linear(num_classes).",
"Chiến lược cho RegNetY-4GF: Đóng băng các lớp Trunk đầu, mở băng Stage 4 cuối, thay thế lớp FC bằng Dropout(p=0.3) + Linear(num_classes)."
]
},
{
"buoc_so": 3,
"ten_buoc": "Thiết lập Tối ưu hóa & Kỹ thuật Điều tiết (Regularization)",
"cac_thao_tac": [
"Hàm mất mát: Cross-Entropy Loss kết hợp Label Smoothing (epsilon=0.1) chống dự đoán quá tự tin.",
"Thuật toán tối ưu: AdamW với Weight Decay = 0.01 (L2 Regularization).",
"Tốc độ học Vi sai (Differential Learning Rates): Backbone LR = 1e-5 (học chậm), Classifier Head LR = 1e-3 (học nhanh).",
"Lịch trình giảm tốc độ học: Cosine Annealing Scheduler (T_max=15 epochs, eta_min=1e-6).",
"Tối ưu phần cứng: Kích hoạt torch.cuda.amp.GradScaler cho Automatic Mixed Precision (AMP)."
]
},
{
"buoc_so": 4,
"ten_buoc": "Vòng lặp Thực thi Epoch",
"cac_buoc_nho": [
"1. Vòng lặp Batch: Nạp mini-batch B=32 qua PyTorch DataLoader (num_workers=4, pin_memory=True).",
"2. Xóa Gradient: Đặt lại các bộ tích tụ đạo hàm về 0 bằng optimizer.zero_grad().",
"3. Lan truyền tiến (Forward Pass): Cho ảnh qua mô hình trong môi trường torch.cuda.amp.autocast().",
"4. Tính Loss: Đánh giá giá trị hàm mất mát có Label Smoothing.",
"5. Lan truyền ngược (Backward Pass): Tính gradient với scaler.scale(loss).backward().",
"6. Cập nhật Trọng số: Cập nhật tham số mô hình qua scaler.step(optimizer) và scaler.update().",
"7. Đánh giá tập Val: Kiểm tra Val Loss/Acc sau mỗi epoch và lưu lại checkpoint có kết quả tốt nhất."
]
},
{
"buoc_so": 5,
"ten_buoc": "Kiểm thử Độc lập (Independent Testing)",
"cac_thao_tac": [
"Nạp bộ trọng số tốt nhất đã saved từ giai đoạn Validation.",
"Chạy lượt suy luận (Inference) trên Tập Kiểm thử (Test Set - 15%) hoàn toàn chưa từng xuất hiện lúc train.",
"Đo đạc chỉ số chẩn đoán: Accuracy, Precision (Macro), Recall (Macro), F1-Score (Macro), Confusion Matrix.",
"Xuất báo cáo chi tiết theo từng lớp bệnh bằng hàm classification_report."
]
},
{
"buoc_so": 6,
"ten_buoc": "Đo đạc Hiệu năng Phần cứng Thực tế",
"cac_thao_tac": [
"Đo Thời gian suy luận trung bình (Inference Latency) tính theo ms/ảnh.",
"Tính Tốc độ xử lý (Throughput - FPS = 1000 / Latency_ms).",
"Ghi nhận Dung lượng bộ nhớ GPU đỉnh (Peak VRAM) tiêu thụ qua torch.cuda.max_memory_allocated() theo MB.",
"Tính Khoảng cách Tổng quát hóa: Generalization Gap (Delta Loss = |Val_Loss - Train_Loss|)."
]
}
],
"ket_qua_dau_ra_ky_vong": {
"bang_so_lieu": [
"Bảng so sánh tổng hợp thực nghiệm (Accuracy, Macro F1, Latency, FPS, VRAM, Train Time, Delta Loss)",
"Bảng phân tích hiệu năng theo từng loại bệnh lá (Precision, Recall, F1-Score từng lớp)"
],
"bieu_do_trutc_quan": [
"Đồ thị huấn luyện (Training Dynamics: Train/Val Loss và Train/Val Accuracy qua 15 Epochs)",
"Ma trận nhầm lẫn (Confusion Matrix) trên tập Test của 2 mô hình"
]
}
}
