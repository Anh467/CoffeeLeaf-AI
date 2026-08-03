# CoffeeLeaf-AI

Pipeline MLOps phát hiện bệnh trên **ảnh chụp cả cây cà phê**. Hệ thống dùng hai
model nối tiếp nhưng chỉ cần một bộ annotation:

1. YOLO instance segmentation tách từng lá khỏi ảnh toàn cây.
2. EfficientNetV2-S hoặc RegNetY-3.2GF phân loại crop lá thành `healthy`,
   `miner`, `phoma`, `rust`.
3. DVC huấn luyện các ứng viên, lưu metrics, áp quality gate và đóng gói cặp
   model phù hợp nhất vào `deployment/`.

Model segmentation được train class-agnostic (`leaf`). Nhãn bệnh trong polygon
gốc chỉ được dùng để tạo crop classification, vì vậy inference không phụ thuộc
vào nhãn bệnh do detector dự đoán.

## Cấu trúc tối giản

```text
pipeline.py       # prepare, train, select, inference và doctor
params.yaml       # dữ liệu, hyperparameters, quality gates
dvc.yaml          # DAG: prepare -> train hai nhánh -> select
data/raw/         # dữ liệu gốc do DVC quản lý, không commit ảnh vào Git
metrics/          # JSON/CSV nhỏ để DVC so sánh thí nghiệm
models/           # checkpoint ứng viên do DVC cache
deployment/       # cặp model thắng + manifest/checksum do DVC cache
```

Notebook và checkpoint cũ chỉ giải quyết classification trên từng lá đã crop;
pipeline mới thay thế luồng đó để tránh nhầm accuracy ảnh một lá với hiệu năng
thực tế trên ảnh toàn cây.

## 1. Chuẩn bị môi trường

Python 3.10–3.12 được khuyến nghị.

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python pipeline.py doctor
```

PyTorch nên được cài bằng wheel CUDA phù hợp với máy trước khi cài phần còn lại
nếu muốn train bằng GPU.

Khởi tạo một DVC remote dùng chung cho nhóm, ví dụ S3:

```bash
dvc remote add -d storage s3://YOUR_BUCKET/coffee-leaf-ai
dvc remote modify storage region YOUR_REGION
```

Không commit access key. Dùng biến môi trường/credential profile của nhà cung
cấp cloud. Có thể thay S3 bằng SSH, Azure, GCS hoặc một remote DVC khác.

## 2. Dữ liệu và annotation

Ảnh phải mô phỏng đúng lúc triển khai: nhiều lá trên một cây, lá chồng lấp,
nền vườn tự nhiên, nhiều khoảng cách, ánh sáng và thiết bị chụp khác nhau.

Mỗi lá nhìn thấy đủ rõ được gán một polygon theo định dạng YOLO segmentation:

```text
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Tọa độ được chuẩn hóa về `[0, 1]`; class id theo thứ tự trong `params.yaml`:

| ID | Lớp |
|---:|---|
| 0 | healthy |
| 1 | miner |
| 2 | phoma |
| 3 | rust |

Đặt ảnh/nhãn ở bất kỳ thư mục con nào dưới `data/raw`, rồi tạo
`data/raw/manifest.csv`:

```csv
image,label,split,group_id
images/farm_a/tree_001_01.jpg,labels/farm_a/tree_001_01.txt,train,farm_a_tree_001
images/farm_a/tree_071_01.jpg,labels/farm_a/tree_071_01.txt,val,farm_a_tree_071
images/farm_b/tree_086_01.jpg,labels/farm_b/tree_086_01.txt,test,farm_b_tree_086
```

`group_id` nên đại diện cho cùng cây hoặc cùng phiên chụp. Pipeline sẽ dừng nếu
một group xuất hiện ở nhiều split, tránh rò rỉ dữ liệu giữa train và test.
`require_group_manifest: true` được bật mặc định; chỉ tắt cho thử nghiệm nhanh.

Đưa dữ liệu lên DVC:

```bash
dvc add data/raw
git add data/raw.dvc .gitignore
dvc push
```

Dataset không được tự động tải từ Kaggle vì ảnh một lá/nền kiểm soát không đại
diện đầy đủ cho ảnh toàn cây. Có thể trộn dataset ngoài vào train, nhưng test
phải gồm ảnh thực địa độc lập tại điều kiện triển khai.

## 3. Huấn luyện và chọn model

Chạy toàn bộ DAG:

```bash
dvc pull
dvc repro
dvc metrics show
dvc plots show
dvc push
```

Các stage:

- `prepare`: kiểm tra polygon/split, tạo nhãn segmentation một lớp `leaf`, mask
  nền ngoài polygon và sinh crop classification.
- `train_segmenters`: mặc định so sánh `yolo11n-seg` và `yolo11s-seg` trên validation.
- `train_classifiers`: mặc định so sánh EfficientNetV2-S và RegNetY-3.2GF.
- `select`: quality gate trước, sau đó tính điểm tổng hợp giữa chất lượng,
  latency và kích thước checkpoint.

Thay hyperparameter rồi chạy một DVC experiment:

```bash
dvc exp run -S segmentation.image_size=960 \
  -S classification.epochs=30
dvc exp show
dvc metrics diff
```

### Quy tắc lựa chọn mặc định

- Segmenter phải đạt `mAP50-95 >= 0.45` và `recall >= 0.65`.
- Classifier phải đạt `macro F1 >= 0.85` và recall của lớp yếu nhất `>= 0.70`.
- Chỉ metrics validation được dùng để chọn model; metrics test được ghi riêng
  để báo cáo độc lập, tránh tối ưu gián tiếp trên test set.
- Trong nhóm qua quality gate, điểm chất lượng có trọng số lớn hơn latency và
  dung lượng model. Có thể sửa toàn bộ ngưỡng/trọng số trong `params.yaml`.
- Nếu không ứng viên nào đạt gate, hệ thống vẫn đóng gói ứng viên tốt nhất để
  phân tích nhưng ghi `deploy_ready: false`; không nên đưa bundle đó lên production.

Kết quả quan trọng:

```text
metrics/segmenters.json
metrics/classifiers.json
metrics/selection.json
metrics/leaderboard.csv
deployment/leaf_segmenter.pt
deployment/leaf_classifier.pt
deployment/manifest.json
```

`manifest.json` lưu tên ứng viên, metrics, preprocessing, thresholds và SHA-256
của từng checkpoint. Đặt `deployment.export_onnx: true` nếu môi trường đã có
đủ dependency ONNX và cần export cho runtime khác.

## 4. Suy luận ảnh toàn cây

Sau `dvc pull`:

```bash
python pipeline.py predict \
  --source path/to/whole_coffee_tree.jpg \
  --output runs/predict
```

Đầu ra gồm:

- `runs/predict/annotated.jpg`: mask, box, lớp bệnh và confidence từng lá.
- `runs/predict/result.json`: kết quả từng instance, số lượng mỗi bệnh và tỉ lệ
  lá bệnh trong các dự đoán vượt ngưỡng.

Tỉ lệ lá bệnh chỉ là chỉ báo thị giác, không thay thế đánh giá nông học. Với
ảnh quá xa làm lá rất nhỏ, nên hướng dẫn người dùng tiến gần hơn hoặc chụp nhiều
ảnh bao phủ từng phần tán cây.

## 5. Kiểm tra nhanh

```bash
python -m py_compile pipeline.py
python pipeline.py doctor
python pipeline.py doctor --check-data
dvc dag
```

`doctor --check-data` xác minh file tồn tại, class id, polygon chuẩn hóa và group
split trước khi sử dụng GPU.
