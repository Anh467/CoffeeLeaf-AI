"""Detect leaves, reject tiny crops, then classify each usable leaf."""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms
from torch import nn
from ultralytics import YOLO


def load_classifier(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt["model"] == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(ckpt["classes"]))
    else:
        model = models.resnet50(weights=None); model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()
    tf = transforms.Compose([transforms.Resize((ckpt["image_size"],)*2), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return model, ckpt["classes"], tf


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(args.detector); classifier, classes, tf = load_classifier(args.classifier, device)
    image = Image.open(args.source).convert("RGB"); draw = ImageDraw.Draw(image)
    result = detector.predict(image, conf=args.det_conf, verbose=False)[0]
    for box in result.boxes.xyxy.cpu().tolist():
        x1,y1,x2,y2 = map(int, box); crop = image.crop((x1,y1,x2,y2))
        if min(crop.size) < args.min_crop: label = "TOO_FAR"
        else:
            with torch.inference_mode(): prob = classifier(tf(crop).unsqueeze(0).to(device)).softmax(1)[0]
            score, idx = prob.max(0); label = f"{classes[idx]} {score:.2f}" if score >= args.cls_conf else "UNCERTAIN"
        draw.rectangle((x1,y1,x2,y2),outline="red",width=3); draw.text((x1,y1),label,fill="red")
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); image.save(args.output)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--detector",required=True)
    p.add_argument("--classifier",required=True); p.add_argument("--output",default="artifacts/prediction.jpg")
    p.add_argument("--det-conf",type=float,default=.25); p.add_argument("--cls-conf",type=float,default=.60)
    p.add_argument("--min-crop",type=int,default=96); main(p.parse_args())
