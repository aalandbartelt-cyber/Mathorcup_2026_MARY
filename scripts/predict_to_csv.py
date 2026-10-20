#!/usr/bin/env python3
"""对测试集推理并生成提交格式 CSV: image_id,class_id,x_center,y_center,width,height
用法: python scripts/predict_to_csv.py --weights runs/exp/weights/best.pt --test-dir data/<数据集>/images/test --conf 0.25
"""
import argparse, csv
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True, help="best.pt 路径")
ap.add_argument("--test-dir", required=True, help="测试集图片目录")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", default="outputs/test_result.csv")
args = ap.parse_args()

model = YOLO(args.weights)
rows = []
for r in model.predict(source=args.test_dir, conf=args.conf, imgsz=args.imgsz, device=0, verbose=False):
    image_id = Path(r.path).name
    if r.boxes is None or len(r.boxes) == 0:
        rows.append([image_id, "", "", "", "", ""])  # 无检出,按赛题要求调整
        continue
    for b in r.boxes:
        cls = int(b.cls.item())
        x, y, w, h = b.xywhn[0].tolist()
        rows.append([image_id, cls, f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"])

out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["image_id", "class_id", "x_center", "y_center", "width", "height"])
    wr.writerows(rows)
print(f"写入 {len(rows)} 行 -> {out}")
