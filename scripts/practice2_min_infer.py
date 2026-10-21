#!/usr/bin/env python3
"""Practice 2 pipeline smoke test, NOT a trained COVID classifier.

ImageNet ResNet18 backbone + seeded, untrained binary head.
prediction: 0=NonCOVID, 1=COVID; probability: softmax of predicted class.
No accuracy or clinical meaning is implied by these outputs.
"""
import argparse
import csv
import json
import socket
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--test-dir', type=Path, default=Path('/tmp/p2/data/predict'))
    ap.add_argument('--out', type=Path, default=Path('outputs/practice2_pred.csv'))
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    if args.batch_size <= 0:
        ap.error('--batch-size must be positive')
    files = sorted(p for p in args.test_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not files:
        ap.error('No supported images in test directory')
    start = time.perf_counter()
    torch.manual_seed(42)
    socket.setdefaulttimeout(30)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise SystemExit(f'BLOCKED: weights URL={weights.url}; {type(exc).__name__}: {exc}') from exc
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(args.device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rows = []
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    infer_start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(files), args.batch_size):
            batch_files = files[offset:offset + args.batch_size]
            tensors = []
            for path in batch_files:
                with Image.open(path) as img:
                    tensors.append(transform(img.convert('RGB')))
            probs = model(torch.stack(tensors).to(args.device)).softmax(dim=1)
            confidence, labels = probs.max(dim=1)
            rows.extend((p.name, label, f'{prob:.8f}')
                        for p, label, prob in zip(batch_files, labels.cpu().tolist(), confidence.cpu().tolist()))
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - infer_start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'prediction', 'probability'])
        writer.writerows(rows)
    summary = {
        'smoke_test_only': True, 'head': 'untrained binary head, seed=42',
        'probability': 'predicted-class softmax; uncalibrated',
        'rows': len(rows), 'positive_count': sum(r[1] == 1 for r in rows),
        'positive_ratio': sum(r[1] == 1 for r in rows) / len(rows),
        'device': args.device, 'weights_url': weights.url,
        'inference_including_preprocessing_seconds': infer_seconds,
        'mean_seconds_per_image': infer_seconds / len(rows),
        'total_including_model_load_and_csv_seconds': time.perf_counter() - start,
    }
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
