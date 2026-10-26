"""Make review previews and check export dimensions; not a plotting backend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
FILES = [
    ("图1_生成质量与多样性.png", (8.4, 5.2)),
    ("图2_三风格感官画像.png", (8.4, 4.6)),
    ("图3_迁移收益与审计状态.png", (8.4, 4.8)),
]
QA = HERE / "qa"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contact(images: list[Image.Image], path: Path) -> None:
    width = max(im.width for im in images)
    height = sum(im.height for im in images) + 24 * (len(images) + 1)
    canvas = Image.new("RGB", (width + 48, height), "white")
    top = 24
    for im in images:
        canvas.paste(im, ((canvas.width - im.width) // 2, top))
        top += im.height + 24
    canvas.save(path, dpi=(300, 300))


def main() -> None:
    QA.mkdir(exist_ok=True)
    color_previews = []
    gray_previews = []
    report = {"scope": "visual_qa_only; source_figures_from_R_ggplot2", "figures": []}
    for filename, inches in FILES:
        png = HERE / filename
        svg = png.with_suffix(".svg")
        assert png.is_file() and svg.is_file()
        with Image.open(png) as original:
            image = original.convert("RGB")
            expected = tuple(round(dim * 300) for dim in inches)
            assert image.size == expected, (filename, image.size, expected)
            dpi = original.info.get("dpi", (0, 0))
            assert all(abs(x - 300) < 1 for x in dpi), (filename, dpi)
        assert svg.read_text(encoding="utf-8", errors="replace").lstrip().startswith("<?xml")
        reduced = image.resize((round(image.width * .7), round(image.height * .7)), Image.Resampling.LANCZOS)
        gray = ImageOps.grayscale(reduced).convert("RGB")
        reduced.save(QA / filename.replace(".png", "_70pct.png"), dpi=(300, 300))
        gray.save(QA / filename.replace(".png", "_70pct_gray.png"), dpi=(300, 300))
        color_previews.append(reduced)
        gray_previews.append(gray)
        report["figures"].append({
            "png": filename, "svg": svg.name, "pixels": image.size, "dpi": dpi,
            "png_sha256": sha256(png), "svg_sha256": sha256(svg),
        })
    contact(color_previews, QA / "Q3_彩色审图合页.png")
    contact(gray_previews, QA / "Q3_灰度审图合页.png")
    (HERE / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Q3 exports passed: 3 PNG at 300 dpi, 3 SVG, 70% and grayscale previews")


if __name__ == "__main__":
    main()
