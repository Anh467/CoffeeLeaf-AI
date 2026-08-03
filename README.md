# CoffeeLeaf-AI

Pipeline two-stage cho ảnh chụp tổng quát:

```text
Ảnh cây → YOLO phát hiện lá → loại crop quá nhỏ → EfficientNetV2-S phân loại bệnh
```

## Dataset được chọn

| Mục đích | Kaggle dataset | Vai trò |
|---|---|---|
| Classification | `coffeedisease/coffee-leaves-disease` | Healthy, Miner, Phoma, Rust |
| Classification | `nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset` | Bổ sung ảnh Robusta ngoài thực địa; chỉ nhãn trùng mới được nhập |
| Detection | `alexo98/leaf-detection` | 1,140 ảnh và 5,346 bounding box lá; train YOLO class `leaf` |

Hai URL ban đầu là **Kaggle notebooks**, không phải dataset slug. Pipeline dùng dataset công khai gắn với notebook và từ chối thư mục nhãn không có trong `LABEL_ALIASES` ở `train.py`; vì vậy nhãn khác nghĩa không bị gộp nhầm. Detector tổng quát cần được fine-tune thêm bằng ảnh cây cà phê thật nếu dùng ngoài demo.

## Chuẩn bị dữ liệu

```text
data/classification/
├── source_1/.../Healthy/*.jpg
└── source_2/.../rust/*.jpg

data/detection/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Không commit dữ liệu hoặc model. Có thể tải bằng Kaggle CLI theo ba slug trong bảng; nếu cấu trúc detection tải về chưa theo YOLO, cần dùng annotation kèm dataset để convert một lần trước khi train.

## Training

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py classify --data data/classification --model efficientnet_v2_s --epochs 20 \
  --dataset-sources "coffee-leaves-disease,rocole-v1"

python train.py prepare-detection --data data/detection --output data/detection_yolo
python train.py detect --data data/detection_yolo/data.yaml --model yolov8n.pt --epochs 30 \
  --dataset-sources "alexo98/leaf-detection"
```

Kết quả local nằm trong `artifacts/`. Mỗi run log params, metrics và model vào MLflow/DagsHub nếu `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` đã được đặt.

## Inference hai model

```bash
python predict.py --source sample.jpg \
  --detector artifacts/leaf_detector/weights/best.pt \
  --classifier artifacts/classifier.pt
```

Crop nhỏ hơn 96 px trả `TOO_FAR`; confidence classification dưới 0.60 trả `UNCERTAIN`. Hai ngưỡng chỉnh được qua CLI.

## MLOps

GitHub Actions chỉ validate code trên PR. Training chỉ chạy thủ công bằng **Actions → MLOps training → Run workflow** để tránh tốn GPU/phút ngoài ý muốn.

Secrets cần tạo:

- `KAGGLE_JSON`
- `MLFLOW_TRACKING_URI` (DagsHub: `https://dagshub.com/Anh467/CoffeeLeaf-AI.mlflow`)
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD` (DagsHub token)

Classification được so bằng `val_f1_macro`; detection bằng `map50_95`. Không tự promote model chỉ từ test metric. Sau khi có bộ ảnh `near/medium/far`, nên thêm gate theo `f1_far` trước khi tự động deploy.

## EDA

Mở `eda.ipynb`, đặt `DATA_ROOT`, rồi Run All để xem số ảnh theo nguồn/lớp, kích thước ảnh và sample. Notebook không train model.
