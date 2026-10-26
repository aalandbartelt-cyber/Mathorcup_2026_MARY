"""Prepare publication previews and a traceable audit; no modelling is executed."""
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd
from PIL import Image, ImageOps
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path("/home/mary/mathorcup")
FIG = ROOT / "04_图表/Q2_关键化合物/v4"
QA = ROOT / "outputs/qa/q2_v4"
AUDIT = ROOT / "03_结果/Q2_关键化合物/publication_audit"
PAPER = ROOT / "05_论文/Q2_关键化合物"


def sheet(images, path, columns=2, width=1050, height=720):
    canvas = Image.new("RGB", (columns*width, ((len(images)+columns-1)//columns)*height), "white")
    for i, source in enumerate(images):
        image = Image.open(source).convert("RGB")
        image.thumbnail((width-30, height-30), Image.Resampling.LANCZOS)
        x = (i % columns)*width+(width-image.width)//2
        y = (i // columns)*height+(height-image.height)//2
        canvas.paste(image, (x,y))
    canvas.save(path)


def main():
    QA.mkdir(parents=True, exist_ok=True)
    paths = [next(FIG.glob(f"图{i}_*_v4.png")) for i in range(1,5)]
    previews, grays = [], []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = image.resize((round(image.width*.7),round(image.height*.7)),Image.Resampling.LANCZOS)
        preview = QA / (path.stem+"_70pct.png")
        gray = QA / (path.stem+"_70pct_gray.png")
        image.save(preview); ImageOps.grayscale(image).save(gray)
        previews.append(preview); grays.append(gray)
    sheet(previews, QA/"Q2_v4_color_contact.png")
    sheet(grays, QA/"Q2_v4_gray_contact.png")
    (FIG/"qa").mkdir(exist_ok=True)
    for name in ("Q2_v4_color_contact.png","Q2_v4_gray_contact.png"):
        shutil.copy2(QA/name,FIG/"qa"/name)
    bzd = CANONICAL / "docs/reference/BZD参考/AA论文/AA论文/figures"
    senior = next((CANONICAL/"docs/reference/学长重要参考").rglob("fig12_jaccard.png"))
    q1 = CANONICAL / "04_图表/Q1_感官评分/final_v4"
    pairs = [q1/"图1_模型对比.png",paths[1],q1/"图4_风格与难目标稳健性.png",paths[3],
             bzd/"fig_paper_q2a.png",paths[0],bzd/"fig_paper_q2b.png",paths[2],senior,paths[1]]
    sheet(pairs, QA/"internal_reference_comparison.png",height=700)
    (QA/"internal_reference_order.json").write_text(json.dumps([str(p) for p in pairs],ensure_ascii=False,indent=2),encoding="utf-8")
    for name in ("text_boxes.csv","text_overlaps.csv","text_outside_canvas.csv"):
        shutil.copy2(QA/name,AUDIT/("q2_v4_"+name))
    # Both new tables are deterministic aggregations / row-layout checks.
    manifest = json.loads((FIG/"source_data_manifest.json").read_text())
    manifest["tables"] = []
    for path in sorted((FIG/"source_data").glob("*.csv")):
        df = pd.read_csv(path)
        manifest["tables"].append({"name":path.name,"rows":len(df),"columns":len(df.columns),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (FIG/"source_data_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    expected = json.loads((AUDIT/"q2_v4_frozen_inputs.json").read_text())
    changed = [p for p,h in expected.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed, changed
    boxes = pd.read_csv(QA/"text_boxes.csv")
    overlaps = pd.read_csv(QA/"text_overlaps.csv")
    outside = pd.read_csv(QA/"text_outside_canvas.csv")
    assert len(overlaps)==len(outside)==0
    assert set(boxes.family)=={"SimSun"} and boxes["size"].min()>=10.5
    report = {"frozen_files":len(expected),"changed_frozen_files":changed,
        "text_boxes":len(boxes),"text_overlap_count":len(overlaps),"text_outside_count":len(outside),
        "families":list(boxes.family.unique()),"font_sizes_pt":sorted(boxes["size"].unique()),
        "scope_note":"Text bounding boxes do not establish general curve-label overlap; final rasters and grayscale are also visually inspected.",
        "figures":[{"name":p.name,"pixels":list(Image.open(p).size),"dpi":Image.open(p).info["dpi"],
            "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    (AUDIT/"q2_v4_visual_checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    pdfs=[PAPER/f"{stem}_v4.pdf" for stem in ("模型假设_Q2","问题分析_Q2","第二问_论文正文")]
    if all(p.exists() for p in pdfs):
        merged=pymupdf.open()
        for path in pdfs:
            with pymupdf.open(path) as source: merged.insert_pdf(source)
        merged.save(PAPER/"Q2_论文审计合订本_v4.pdf")
        page_images=[]
        for i,page in enumerate(merged):
            png=QA/f"audit_page_{i+1:02}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25)).save(png)
            page_images.append(png)
        sheet(page_images,QA/"audit_pdf_contact.png",columns=3,width=640,height=900)
        print("audit PDF pages",len(merged))
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    main()
