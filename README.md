# CoffeeLeaf-AI

So sánh EfficientNetV2-S và RegNetY-3.2GF cho bài toán phân loại bốn trạng thái lá
cà phê: `Healthy`, `Miner`, `Phoma`, `Rust`.

PR MLOps này giữ nguyên training loop trong `coffee_leaf_experiment.ipynb`. Notebook
chỉ nhận thêm cấu hình qua environment variables và xuất một file JSON dùng làm
contract; toàn bộ orchestration nằm ở các script bên ngoài.

## Luồng tối giản

```text
DVC dataset -> validate -> execute notebook -> MLflow run/artifacts
            -> quality gate -> optional champion promotion
```

GitHub Actions chỉ chạy kiểm tra nhanh. Training vẫn chạy trên máy có GPU, tránh tải
dataset và huấn luyện hai model trong mỗi pull request.

## 1. Cài công cụ MLOps

Kích hoạt môi trường `agriviet-ai` đang chạy được notebook, sau đó:

```powershell
python -m pip install -r requirements-mlops.txt
```

## 2. Đưa dataset vào DVC

Copy dữ liệu theo cấu trúc trong `data/README.md`, rồi chạy:

```powershell
dvc add data/raw
git add data/raw.dvc .gitignore
```

Remote miễn phí trên máy khác hoặc ổ đĩa khác:

```powershell
dvc remote add --local -d storage D:/MLOps/coffeeleaf-dvc
dvc push
```

Không commit credential của Azure hoặc remote khác. Nếu cần credential cục bộ, dùng
environment variables hoặc `.dvc/config.local`.

## 3. Chạy pipeline

Chạy toàn bộ pipeline DVC:

```powershell
dvc repro
```

Hoặc chạy notebook wrapper trực tiếp, ví dụ smoke run một epoch:

```powershell
python scripts/run_pipeline.py --config params.yaml --epochs 1 --batch-size 8
python scripts/quality_gate.py --config params.yaml
```

Kết quả chính:

- `outputs/latest/executed.ipynb`: notebook đã thực thi, không ghi đè notebook nguồn.
- `outputs/latest/*.pth`: checkpoint sinh trong run mới.
- `metrics/candidate.json`: metric validation/test, Git SHA, data version và MLflow run ID.
- `metrics/promotion.json`: kết quả từng điều kiện của quality gate.
- `mlflow.db` và `mlruns/`: experiment tracking cục bộ.

Mở MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## 4. Quality gate và promotion

Gate dùng `best_val_accuracy`, không dùng Test F1 để quyết định promotion:

```text
candidate validation accuracy >= 0.90
candidate validation accuracy > champion validation accuracy
checkpoint tồn tại và không rỗng
evaluation version không thay đổi
```

`metrics/champion.json` khởi tạo từ EfficientNetV2-S của commit hiện tại với best
validation accuracy `99.58%`. Nếu `metrics/promotion.json` có `approved: true`, promote:

```powershell
python scripts/quality_gate.py --config params.yaml --promote
git add metrics/champion.json
```

Promotion chỉ cập nhật manifest trỏ tới artifact của MLflow run; nó không tự deploy.
Do notebook hiện chia validation bằng `train_test_split`, thay đổi dataset sẽ làm đổi
evaluation version và gate sẽ dừng so sánh tự động. Khi đó cần review/rebaseline thay
vì so sánh hai metric trên hai tập validation khác nhau.

## Thay đổi tối thiểu trong notebook

Notebook chỉ thay hai điểm:

1. Đọc dataset path và hyperparameters từ environment variables, vẫn giữ nguyên giá
   trị mặc định cũ khi chạy thủ công.
2. Xuất model candidate và validation metric vào `MLOPS_METRICS_PATH`.

Model architecture, augmentation, optimizer, training loop, checkpoint format và báo
cáo cũ không bị refactor.
