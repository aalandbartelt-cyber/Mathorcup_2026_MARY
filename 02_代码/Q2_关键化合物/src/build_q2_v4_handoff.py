"""Copy the accepted manuscript, replacing only figures and their descriptions.

No model, ranking, table, equation or non-figure paragraph is regenerated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "05_论文/Q2_关键化合物"
FIG = ROOT / "04_图表/Q2_关键化合物/v4"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

FIGURE_SENTENCE = (
    "图3将入选频率、留一风格保留率、预测层级与化学等级放在同一证据框架下。",
    "图3按化学证据等级分别展示样本扰动与风格扰动下的稳定性分布。",
)
CAPTION_CHANGE = (
    "图3 最终候选的重采样稳定性、跨风格保留率与证据等级",
    "图3 不同化学证据等级的样本与风格扰动稳定性",
)
NOTES = {
    1: "注：全量描述图；边数采用可包含零值的伪对数轴。虚线为既定阈值0.60，正负边数为264和2；全量描述图不参与折内监督验证。",
    2: "注：前三面板为10次重复的均值±标准差，短横线表示标准差。NNLS指感官组收缩NNLS，图ElasticNet指有符号图正则回归，ExtraTrees为独立树验证器，Spearman为边际关联Top-3预测器。点图坐标轴局部放大；MAE越低越好，R²与Spearman越高越好。末面板为正式Top-3与四路证据的平均Jaccard，不是预测性能。",
    3: "注：两面板均纳入全部150条关系。每格为对应等级内的关系数，底色为等级内比例；五个区间依次左闭右开，末区间包含1。A/B/C/D为既有化学核验等级，分箱只用于呈现，不改变等级或排序。",
    4: "注：左侧名次越小越靠前；右侧点为OAV中位数，横线为四分位区间。OAV为无量纲浓度/阈值比，采用含零伪对数轴，虚线为OAV=1；N/A表示单位或阈值不可比，0为零值，二者不等同。两面板关系按行对齐。4VG为4-乙烯基愈创木酚。",
}


def text(node):
    return "".join(node.xpath(".//w:t/text()", namespaces=NS))


def set_text(paragraph, value):
    runs = paragraph.findall("w:r", NS)
    template = runs[0] if runs else etree.Element(f"{{{NS['w']}}}r")
    for child in list(paragraph):
        if child.tag != f"{{{NS['w']}}}pPr":
            paragraph.remove(child)
    run = etree.SubElement(paragraph, f"{{{NS['w']}}}r")
    properties = template.find("w:rPr", NS)
    if properties is not None:
        run.append(copy.deepcopy(properties))
    etree.SubElement(run, f"{{{NS['w']}}}t").text = value


def main():
    source = PAPER / "第二问_论文正文_v3.docx"
    target = PAPER / "第二问_论文正文_v4.docx"
    with ZipFile(source) as archive:
        blobs = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    root = etree.fromstring(blobs["word/document.xml"])
    before_math = [etree.tostring(x) for x in root.xpath(".//m:oMath", namespaces=NS)]
    before_tables = [etree.tostring(x) for x in root.xpath(".//w:tbl", namespaces=NS)]
    rels = etree.fromstring(blobs["word/_rels/document.xml.rels"])
    relmap = {x.get("Id"): x.get("Target") for x in rels}
    drawings = root.xpath(".//w:drawing", namespaces=NS)
    figures = [next(FIG.glob(f"图{i}_*_v4.png")) for i in range(1, 5)]
    assert len(drawings) == 4
    for drawing, figure in zip(drawings, figures):
        blip = drawing.find(".//a:blip", NS)
        member = "word/" + relmap[blip.get(f"{{{NS['r']}}}embed")]
        blobs[member] = figure.read_bytes()
        width, height = Image.open(figure).size
        extent = drawing.find(".//wp:extent", NS)
        cy = round(int(extent.get("cx")) * height / width)
        extent.set("cy", str(cy))
        for node in drawing.findall(".//a:xfrm/a:ext", NS):
            node.set("cy", str(cy))

    changed = []
    for p in root.xpath("./w:body/w:p", namespaces=NS):
        value = text(p)
        if FIGURE_SENTENCE[0] in value:
            set_text(p, value.replace(*FIGURE_SENTENCE)); changed.append("figure3_reference")
        if value == CAPTION_CHANGE[0]:
            set_text(p, CAPTION_CHANGE[1]); changed.append("figure3_caption")
        for number, note in NOTES.items():
            if value.startswith(f"图{number} "):
                added = copy.deepcopy(p)
                set_text(added, note)
                # Notes remain editable body paragraphs; captions retain template styles.
                props = added.find("w:pPr", NS)
                if props is None:
                    props = etree.SubElement(added, f"{{{NS['w']}}}pPr")
                for name in ("jc", "keepNext", "pageBreakBefore"):
                    for child in props.findall(f"w:{name}", NS): props.remove(child)
                style = props.find("w:pStyle", NS)
                if style is None: style = etree.SubElement(props, f"{{{NS['w']}}}pStyle")
                style.set(f"{{{NS['w']}}}val", "BodyText")
                for rp in added.findall("w:r/w:rPr", NS):
                    for name in ("b", "bCs", "sz", "szCs"):
                        for node in rp.findall(f"w:{name}", NS): rp.remove(node)
                    for name in ("sz", "szCs"):
                        node = etree.SubElement(rp, f"{{{NS['w']}}}{name}")
                        node.set(f"{{{NS['w']}}}val", "18")
                p.addnext(added)
    assert changed == ["figure3_reference", "figure3_caption"]
    assert before_math == [etree.tostring(x) for x in root.xpath(".//m:oMath", namespaces=NS)]
    assert before_tables == [etree.tostring(x) for x in root.xpath(".//w:tbl", namespaces=NS)]
    blobs["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for name, blob in blobs.items(): archive.writestr(name, blob)
    for stem in ("模型假设_Q2", "问题分析_Q2"):
        shutil.copy2(PAPER / f"{stem}_v3.docx", PAPER / f"{stem}_v4.docx")
    audit = {"source": str(source.relative_to(ROOT)), "output": str(target.relative_to(ROOT)),
        "unchanged_math": len(before_math), "unchanged_tables_including_equations": len(before_tables),
        "changed_existing_text": changed, "added_figure_notes": 4,
        "images": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in figures}}
    out = ROOT / "03_结果/Q2_关键化合物/publication_audit/q2_v4_docx_changes.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
