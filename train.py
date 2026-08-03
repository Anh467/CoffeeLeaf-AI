"""Train coffee-leaf classification or leaf detection with MLflow tracking.

Examples:
  python train.py classify --data data/classification --model efficientnet_v2_s
  python train.py detect --data data/detection/data.yaml --model yolov8n.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_ALIASES = {
    "healthy": "healthy", "health": "healthy", "normal": "healthy",
    "miner": "miner", "leaf_miner": "miner", "leaf miner": "miner",
    "phoma": "phoma", "phoma_leaf_spot": "phoma",
    "rust": "rust", "leaf_rust": "rust", "coffee_leaf_rust": "rust",
}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def canonical_label(value: str) -> str | None:
    key = value.lower().strip().replace("-", "_")
    return LABEL_ALIASES.get(key, LABEL_ALIASES.get(key.replace("_", " ")))


def discover_images(root: Path) -> list[tuple[Path, str]]:
    """Discover class-folder images across multiple dataset source folders."""
    records, seen_hashes = [], set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            label = canonical_label(path.parent.name)
            if label:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest not in seen_hashes:
                    records.append((path, label)); seen_hashes.add(digest)
    if not records:
        raise ValueError(f"No supported class-folder images found below {root}")
    return records


class CoffeeDataset(Dataset):
    def __init__(self, records, classes, transform):
        self.records, self.transform = records, transform
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), self.class_to_idx[label]


def build_classifier(name: str, num_classes: int):
    if name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        size = 300
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        size = 224
    else:
        raise ValueError(f"Unsupported classifier: {name}")
    return model, size


def evaluate(model, loader, device):
    model.eval(); ys, preds = [], []
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            ys.extend(y.numpy()); preds.extend(pred)
    return np.asarray(ys), np.asarray(preds)


def train_classification(args):
    records = discover_images(Path(args.data))
    classes = sorted({label for _, label in records})
    counts = Counter(label for _, label in records)
    too_small = [name for name, count in counts.items() if count < 7]
    if too_small: raise ValueError(f"Classes require >=7 unique images for stratified 70/15/15 split: {too_small}")

    paths = np.arange(len(records)); labels = [label for _, label in records]
    train_idx, temp_idx = train_test_split(paths, test_size=0.30, stratify=labels, random_state=args.seed)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_labels, random_state=args.seed)
    pick = lambda ids: [records[int(i)] for i in ids]

    model, size = build_classifier(args.model, len(classes))
    train_tf = transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), transforms.ColorJitter(.2, .2, .2), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    loader = lambda recs, tf, shuffle=False: DataLoader(CoffeeDataset(recs, classes, tf),
        batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=True)
    train_loader, val_loader, test_loader = loader(pick(train_idx), train_tf, True), loader(pick(val_idx), eval_tf), loader(pick(test_idx), eval_tf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    best_path, best_f1 = output / "classifier.pt", -1.0

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"classify-{args.model}"):
        mlflow.log_params({"task":"classification", "model":args.model, "epochs":args.epochs,
            "batch_size":args.batch_size, "learning_rate":args.lr, "seed":args.seed,
            "classes":json.dumps(classes), "dataset_sources":args.dataset_sources})
        for epoch in range(args.epochs):
            model.train(); total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y); loss.backward(); optimizer.step()
                total_loss += loss.item() * len(y)
            y_val, p_val = evaluate(model, val_loader, device)
            val_f1 = f1_score(y_val, p_val, average="macro")
            mlflow.log_metrics({"train_loss":total_loss/len(train_loader.dataset),
                "val_accuracy":accuracy_score(y_val,p_val), "val_f1_macro":val_f1}, step=epoch)
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({"state_dict":model.state_dict(), "model":args.model, "classes":classes, "image_size":size}, best_path)

        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        y_test, p_test = evaluate(model, test_loader, device)
        metrics = {"test_accuracy":accuracy_score(y_test,p_test),
            "test_f1_macro":f1_score(y_test,p_test,average="macro"),
            "test_f1_weighted":f1_score(y_test,p_test,average="weighted")}
        report_path = output / "classification_report.json"
        report_path.write_text(json.dumps(classification_report(y_test,p_test,target_names=classes,output_dict=True),indent=2),encoding="utf-8")
        mlflow.log_metrics(metrics); mlflow.log_artifact(str(best_path), "model"); mlflow.log_artifact(str(report_path), "evaluation")
        print(json.dumps(metrics, indent=2))


def train_detection(args):
    from ultralytics import YOLO
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"detect-{Path(args.model).stem}"):
        mlflow.log_params({"task":"detection", "model":args.model, "epochs":args.epochs,
            "batch_size":args.batch_size, "image_size":args.image_size,
            "dataset_sources":args.dataset_sources})
        result = YOLO(args.model).train(data=args.data, epochs=args.epochs, imgsz=args.image_size,
            batch=args.batch_size, project=args.output, name="leaf_detector", exist_ok=True)
        values = result.results_dict
        mapping = {"metrics/precision(B)":"precision", "metrics/recall(B)":"recall",
            "metrics/mAP50(B)":"map50", "metrics/mAP50-95(B)":"map50_95"}
        mlflow.log_metrics({out:float(values[key]) for key,out in mapping.items() if key in values})
        best = Path(result.save_dir) / "weights" / "best.pt"
        if best.exists(): mlflow.log_artifact(str(best), "model")


def prepare_detection(args):
    """Convert Pascal VOC XML leaf boxes to a reproducible YOLO split."""
    source, output = Path(args.data), Path(args.output)
    xml_files = sorted(source.rglob("*.xml"))
    if not xml_files:
        raise ValueError(f"No Pascal VOC .xml annotations found below {source}")
    samples = []
    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")
        candidates = ([xml_path.parent / filename] if filename else []) + [
            p for ext in IMAGE_EXTENSIONS for p in source.rglob(f"{xml_path.stem}{ext}")]
        image_path = next((p for p in candidates if p.exists()), None)
        if image_path: samples.append((image_path, xml_path))
    if not samples: raise ValueError("Annotations found, but matching images were not found")
    indexes = np.arange(len(samples))
    train_idx, val_idx = train_test_split(indexes, test_size=args.val_ratio, random_state=args.seed)
    split_map = {int(i): "train" for i in train_idx} | {int(i): "val" for i in val_idx}
    for index, (image_path, xml_path) in enumerate(samples):
        split = split_map[index]
        image_dir, label_dir = output/"images"/split, output/"labels"/split
        image_dir.mkdir(parents=True,exist_ok=True); label_dir.mkdir(parents=True,exist_ok=True)
        root = ET.parse(xml_path).getroot(); size = root.find("size")
        width, height = float(size.findtext("width")), float(size.findtext("height"))
        lines=[]
        for obj in root.findall("object"):
            box=obj.find("bndbox"); xmin,ymin,xmax,ymax=[float(box.findtext(k)) for k in ("xmin","ymin","xmax","ymax")]
            x=((xmin+xmax)/2)/width; y=((ymin+ymax)/2)/height; w=(xmax-xmin)/width; h=(ymax-ymin)/height
            if w>0 and h>0: lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        target_name=f"{index:06d}_{image_path.name}"
        shutil.copy2(image_path,image_dir/target_name)
        (label_dir/Path(target_name).with_suffix(".txt")).write_text("\n".join(lines),encoding="utf-8")
    yaml = f"path: {output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: leaf\n"
    (output/"data.yaml").write_text(yaml,encoding="utf-8")
    print(f"Prepared {len(samples)} images at {output}")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="task", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", required=True); common.add_argument("--model", required=True)
    common.add_argument("--epochs", type=int, default=20); common.add_argument("--batch-size", type=int, default=16)
    common.add_argument("--output", default="artifacts"); common.add_argument("--experiment", default="coffee-leaf-ai")
    common.add_argument("--dataset-sources", default="unknown", help="Dataset slugs/version for reproducibility")
    c = sub.add_parser("classify", parents=[common]); c.add_argument("--lr", type=float, default=1e-4)
    c.add_argument("--seed", type=int, default=42); c.add_argument("--workers", type=int, default=2)
    c.set_defaults(handler=train_classification, model="efficientnet_v2_s")
    d = sub.add_parser("detect", parents=[common]); d.add_argument("--image-size", type=int, default=640)
    d.set_defaults(handler=train_detection, model="yolov8n.pt")
    prep = sub.add_parser("prepare-detection")
    prep.add_argument("--data",required=True); prep.add_argument("--output",default="data/detection_yolo")
    prep.add_argument("--val-ratio",type=float,default=.2); prep.add_argument("--seed",type=int,default=42)
    prep.set_defaults(handler=prepare_detection)
    return root


if __name__ == "__main__":
    args = parser().parse_args(); seed_everything(getattr(args, "seed", 42))
    args.handler(args)
