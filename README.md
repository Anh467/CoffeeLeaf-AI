# CoffeeLeaf-AI

Pipeline MLOps phát hiện bệnh trên **ảnh chụp cả cây cà phê**. Hệ thống dùng hai
model nối tiếp và hai nguồn dữ liệu đúng với từng nhiệm vụ:

1. [BRACOT](https://data.mendeley.com/datasets/pmkbyjpf6k/1) train YOLO instance
   segmentation để tách từng lá khỏi ảnh cây.
2. [Coffee leaf diseases trên Kaggle](https://www.kaggle.com/datasets/badasstechie/coffee-leaf-diseases/data)
   so sánh EfficientNetV2-S, RegNetY-3.2GF, ConvNeXt-Tiny và DenseNet121 với ba
   đầu ra độc lập `miner`, `rust`, `phoma`; `healthy` được suy ra khi cả ba đều
   âm tính.
3. DVC huấn luyện các ứng viên, lưu metrics, áp quality gate và đóng gói cặp
   model phù hợp nhất vào `deployment/`.
4. MLflow trên DagsHub ghi lại experiment, đăng ký challenger và chỉ chuyển
   model tốt hơn sang `Production`/alias `champion`.

Model segmentation được train class-agnostic (`leaf`). Dữ liệu BRACOT không bị
gán nhãn bệnh giả; nhãn bệnh chỉ đến từ `train_classes.csv` và
`test_classes.csv` của dataset Kaggle. Đây là bài toán multi-label vì một lá có
thể đồng thời mang `miner` và `rust`.

## Cấu trúc tối giản

```text
notebooks/train_leaf_segmentation.ipynb      # training YOLO segmentation
notebooks/train_disease_classification.ipynb # training classifier
pipeline.py       # prepare, select, publish, inference, doctor và model contract
params.yaml       # dữ liệu, hyperparameters, quality gates
dvc.yaml          # DAG: prepare -> hai notebook train -> select -> publish
data/raw/segmentation/   # BRACOT + VIA JSON, do DVC quản lý
data/raw/classification/ # Kaggle images + masks + CSV, do DVC quản lý
metrics/          # JSON/CSV nhỏ để DVC so sánh thí nghiệm
models/           # checkpoint ứng viên do DVC cache
deployment/       # cặp model thắng + manifest/checksum do DVC cache
```

Hai notebook chứa training loop thật và có thể mở bằng JupyterLab để theo dõi.
DVC chạy chính các notebook này bằng Papermill ở chế độ headless, vì vậy notebook
không phải bước thủ công nằm ngoài pipeline và không làm mất khả năng tái lập.

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

Tạo/import repository `Anh467/CoffeeLeaf-AI` trên DagsHub, lấy access token rồi
cấu hình một lần:

```bash
export DAGSHUB_USER_TOKEN="..."          # dùng secret của runner, không ghi vào Git
python pipeline.py setup-dagshub
```

Lệnh này nối MLflow Tracking và DVC remote với DagsHub. Nếu owner/repository
khác giá trị mặc định trong `params.yaml`, đặt `DAGSHUB_REPO_OWNER` và
`DAGSHUB_REPO_NAME`. Credential chỉ nằm trong môi trường/cấu hình local; tuyệt
đối không commit token hay `.dvc/config.local`.

## 2. Chuẩn bị hai dataset

### 2.1. Segmentation: BRACOT

Tải `BRACOT-data.zip` từ Mendeley, giải nén toàn bộ vào:

```text
data/raw/segmentation/bracot/
```

Không đổi tên ảnh hoặc JSON. Pipeline tìm tự động mọi JSON do VGG Image
Annotator (VIA) tạo, ghép annotation với ảnh, chuẩn hóa polygon và sinh nhãn YOLO
segmentation một lớp `leaf`.

Nếu không có manifest, BRACOT được chia ổn định theo seed thành 70% train, 15%
validation và 15% test. Muốn kiểm soát split theo cây/phiên chụp, tạo file
`data/raw/segmentation/bracot/manifest.csv`:

```csv
image,split,group_id
BRACOT-data/images/001.jpg,train,tree_001
BRACOT-data/images/002.jpg,val,tree_002
BRACOT-data/images/003.jpg,test,tree_003
```

Đường dẫn `image` phải tính từ thư mục `bracot`. Pipeline dừng nếu manifest thiếu
ảnh VIA, chứa ảnh dư, hoặc cùng `group_id` xuất hiện ở nhiều split.

### 2.2. Classification: Coffee leaf diseases trên Kaggle

Tải và giải nén dataset vào:

```text
data/raw/classification/coffee-leaf-diseases/
└── coffee-leaf-diseases/
    ├── train/
    │   ├── images/       # 1264 JPG
    │   └── masks/        # 1264 PNG
    ├── test/
    │   ├── images/       # 400 JPG
    │   └── masks/        # 400 PNG
    ├── train_classes.csv
    ├── test_classes.csv
    └── mask_colors.csv
```

Có thể giữ nguyên thư mục trung gian do Kaggle tạo; pipeline tìm duy nhất ba CSV
theo tên. CSV nhãn phải có schema `id,miner,rust,phoma`, và mỗi `id` phải có một
ảnh cùng một mask. Pipeline giữ nguyên 400 mẫu test chính thức, rồi lấy 15% phần
train làm validation bằng cách stratify theo toàn bộ tổ hợp nhãn. Mask màu được
dùng để loại background trong notebook; không dùng mask như nhãn đầu ra của
classifier.

Classifier có ba sigmoid output và dùng `BCEWithLogitsLoss`, không phải softmax
bốn lớp. Một mẫu `miner=1,rust=1,phoma=0` được giữ là hai bệnh đồng thời;
`healthy` chỉ là trạng thái dẫn xuất khi cả ba nhãn bằng 0.

### 2.3. Kiểm tra và quản lý bằng DVC

Tại repository root:

```bash
python pipeline.py doctor --check-data
dvc add data/raw/segmentation data/raw/classification
git add data/raw/segmentation.dvc data/raw/classification.dvc .gitignore
dvc push
```

`prepare` tạo hai đầu ra độc lập mà notebook đang dùng:

```text
data/processed/segmentation/     # YOLO images/labels + dataset.yaml
data/processed/classification/   # image/mask theo split + labels.csv
```

Kaggle chỉ dùng để train classifier; BRACOT chỉ dùng để train segmenter. Khi
inference, segmenter tách lá trên ảnh cây rồi classifier dự đoán bệnh cho từng
crop. Để đánh giá triển khai nghiêm túc, nên bổ sung một test set ảnh thực địa
riêng vì domain ảnh lá Kaggle có thể khác crop do segmenter tạo ra.

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

- `prepare`: chuyển VIA polygon của BRACOT thành YOLO một lớp `leaf`; kiểm tra
  cặp image/mask và CSV multi-label của Kaggle, giữ test gốc và sinh validation.
- `train_segmenters`: Papermill thực thi `train_leaf_segmentation.ipynb`, mặc
  định so sánh `yolo11n-seg` và `yolo11s-seg` trên validation.
- `train_classifiers`: Papermill thực thi
  `train_disease_classification.ipynb`, mặc định so sánh EfficientNetV2-S,
  RegNetY-3.2GF, ConvNeXt-Tiny và DenseNet121. Notebook áp augmentation nhẹ,
  xuất EDA, training curves, confusion matrices và biểu đồ metric sau training.
- `select`: quality gate trước, sau đó tính điểm tổng hợp giữa chất lượng,
  latency và kích thước checkpoint.
- `publish`: log bundle/params/metrics/fingerprint lên DagsHub MLflow, so sánh
  challenger với model `Production`, rồi promote theo chính sách trong
  `params.yaml`.

Muốn xem và chạy từng training loop tương tác, mở notebook từ repository root:

```bash
jupyter lab notebooks
```

Trước khi chạy notebook thủ công, hãy chạy `dvc repro prepare`. Cách được khuyến
nghị để ghi đúng cache/dependency vẫn là `dvc repro train_segmenters`,
`dvc repro train_classifiers` hoặc toàn bộ `dvc repro`.

Thay hyperparameter rồi chạy một DVC experiment:

```bash
dvc exp run -S segmentation.image_size=960 \
  -S classification.epochs=30
dvc exp show
dvc metrics diff
```

Thêm ảnh hoặc sửa annotation trong một trong hai thư mục `data/raw`, đổi danh
sách candidate hay hyperparameter trong `params.yaml`, rồi chạy lại `dvc repro`.
DVC chỉ chạy lại những stage train bị ảnh hưởng; `publish` luôn kiểm tra lại trạng thái Registry
và nhánh Git, nhưng nhận diện bundle theo SHA-256 nên không tạo model version
trùng. Nhờ vậy candidate được train ở feature branch có thể được promote sau khi
merge và chạy pipeline trên `main`.

Tên key của classifier là tên experiment; trường `architecture` chọn backbone
được hỗ trợ (`efficientnet_v2_s`, `regnet_y_3_2gf`, `convnext_tiny` hoặc
`densenet121`). Vì vậy có thể khai báo nhiều cấu hình của cùng một backbone với
image size/dropout khác nhau mà không sửa code. `batch_size` có thể đặt riêng
cho từng candidate; cấu hình ConvNeXt mặc định dùng batch 4 để vừa GPU 6 GB.
Candidate YOLO nhận trực tiếp tên/path weight Ultralytics.

### Quy tắc lựa chọn mặc định

- Segmenter phải đạt `mAP50-95 >= 0.45` và `recall >= 0.65`.
- Classifier phải đạt `macro F1 >= 0.85` và recall của lớp yếu nhất `>= 0.70`.
- Chỉ metrics validation được dùng để chọn model; metrics test được ghi riêng
  để báo cáo độc lập, tránh tối ưu gián tiếp trên test set.
- Trong nhóm qua quality gate, điểm chất lượng có trọng số lớn hơn latency và
  dung lượng model. Có thể sửa toàn bộ ngưỡng/trọng số trong `params.yaml`.
- Nếu không ứng viên nào đạt gate, hệ thống vẫn đóng gói ứng viên tốt nhất để
  phân tích nhưng ghi `deploy_ready: false`; DagsHub chỉ log experiment và không
  tạo model version.

### Champion–challenger trên DagsHub

Điểm dùng để so sánh **giữa các lần train** là `deployment_score` ổn định, chỉ
dùng validation metrics:

| Metric | Trọng số mặc định |
|---|---:|
| Segmenter mAP50-95 | 0.35 |
| Segmenter recall | 0.15 |
| Classifier macro F1 | 0.35 |
| Classifier recall của lớp yếu nhất | 0.15 |

Bundle mới chỉ thành champion khi đồng thời:

- cả segmenter và classifier qua quality gate;
- chạy từ nhánh `main` (`required_git_branch`);
- dùng cùng `evaluation_fingerprint` với champion hiện tại;
- `deployment_score` tăng ít nhất `0.005`.

Model đủ gate nhưng chưa tốt hơn vẫn được đăng ký làm candidate để audit. Model
đầu tiên qua gate được promote nếu `allow_first_model: true`. Nếu validation data
hoặc preprocessing validation thay đổi, fingerprint cũng đổi và auto-promotion
dừng với trạng thái `evaluation_set_changed`; hãy giữ một validation benchmark
cố định hoặc đánh giá lại baseline trước khi chủ động bật
`allow_evaluation_change`.

Kết quả quan trọng:

```text
metrics/segmenters.json
metrics/classifiers.json
metrics/classification_eda.json
metrics/classification_eda.png
metrics/classifier_training_curves.png
metrics/classifier_comparison.png
metrics/classifier_confusion_matrices.png
metrics/selection.json
metrics/publish.json
metrics/leaderboard.csv
deployment/leaf_segmenter.pt
deployment/leaf_classifier.pt
deployment/manifest.json
```

`manifest.json` lưu tên ứng viên, metrics, dataset/evaluation fingerprint,
`deployment_score`, preprocessing, thresholds và SHA-256 của checkpoint/bundle.
Đặt `deployment.export_onnx: true` nếu môi trường đã có đủ dependency ONNX và
cần export cho runtime khác.

Trên DagsHub, mỗi run chứa bundle và các file tái lập (`params.yaml`, `dvc.yaml`,
metrics). Thư mục artifact `analysis/` chứa EDA, training curves, confusion
matrices, CSV so sánh và report/history của tất cả classifier candidate; weight
của mọi candidate vẫn do DVC quản lý. Model Registry dùng stage `Production` và
alias `champion` làm hợp đồng triển khai. Việc promote này không tự tạo inference
server; service triển khai có thể tải đúng bundle production bằng MLflow:

```python
import dagshub
import mlflow

dagshub.init(repo_owner="Anh467", repo_name="CoffeeLeaf-AI", mlflow=True)
bundle = mlflow.artifacts.download_artifacts(
    artifact_uri="models:/CoffeeLeaf-AI-Pipeline/Production"
)
print(bundle)
```

## 4. Suy luận ảnh toàn cây

Sau `dvc pull`:

```bash
python pipeline.py predict \
  --source path/to/whole_coffee_tree.jpg \
  --bundle deployment \
  --output runs/predict
```

`--bundle` cũng nhận đường dẫn do `mlflow.artifacts.download_artifacts` trả về,
nhờ vậy runtime luôn có thể lấy champion từ DagsHub Registry.

Đầu ra gồm:

- `runs/predict/annotated.jpg`: mask, box, một hoặc nhiều bệnh và confidence của
  từng lá.
- `runs/predict/result.json`: `predicted_labels`, xác suất từng bệnh, số lượng
  mỗi nhãn và tỉ lệ lá bệnh trong các dự đoán vượt ngưỡng. Vì đây là multi-label,
  tổng `class_counts` có thể lớn hơn số lá nếu một lá mang nhiều bệnh.

Tỉ lệ lá bệnh chỉ là chỉ báo thị giác, không thay thế đánh giá nông học. Với
ảnh quá xa làm lá rất nhỏ, nên hướng dẫn người dùng tiến gần hơn hoặc chụp nhiều
ảnh bao phủ từng phần tán cây.

## 5. Kiểm tra nhanh

```bash
python -m py_compile pipeline.py
python -c "import json; json.load(open('notebooks/train_leaf_segmentation.ipynb', encoding='utf-8'))"
python -c "import json; json.load(open('notebooks/train_disease_classification.ipynb', encoding='utf-8'))"
python pipeline.py doctor
python pipeline.py doctor --check-data
dvc dag
```

`doctor --check-data` xác minh VIA polygon của BRACOT; đồng thời kiểm tra schema
CSV, cặp image/mask, kích thước ảnh, các nhãn dương/healthy và split của Kaggle
trước khi sử dụng GPU. `doctor` chỉ báo token là `missing`, không in giá trị
secret. Publish cần `DAGSHUB_USER_TOKEN`; nếu không có, stage sẽ dừng thay vì giả
vờ đã triển khai.
