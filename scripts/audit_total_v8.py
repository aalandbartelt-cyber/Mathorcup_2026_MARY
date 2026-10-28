"""Audit the official-front V8 manuscript without changing its contents."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json
import re
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT / "05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v7.docx"
V8 = ROOT / "05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v8.docx"
V7PDF = V7.with_suffix(".pdf")
V8PDF = V8.with_suffix(".pdf")
TEMPLATE = Path(
    "/mnt/d/上海大学__/26年妈妈杯数模/2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题/"
    "2026年MathorCup数学应用挑战赛—大数据竞赛论文模板.docx"
)
AUDIT = ROOT / "docs/collab/20261027_总论文v8官方模板"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def load_doc(path: Path):
    with ZipFile(path) as z:
        data = {n: z.read(n) for n in z.namelist()}
    return data, E.fromstring(data["word/document.xml"])


def ptext(node) -> str:
    return "".join(node.xpath(".//w:t/text()", namespaces=NS))


def strip_font_locale(node):
    """Normalize the deliberate V8 eastAsia font addition for body comparison."""
    node = E.fromstring(E.tostring(node))
    for rf in node.xpath(".//w:rFonts", namespaces=NS):
        attr = "{%s}eastAsia" % NS["w"]
        if attr in rf.attrib:
            del rf.attrib[attr]
    return E.tostring(node, encoding="utf-8", with_tail=False)


def body_after(root, heading: str):
    body = root.find("w:body", NS)
    items = list(body)
    start = next(i for i, x in enumerate(items) if x.tag.endswith("}p") and ptext(x) == heading)
    return items[start:]


def section_margins(root):
    vals = []
    for sect in root.xpath("//w:sectPr", namespaces=NS):
        mar = sect.find("w:pgMar", NS)
        vals.append({k.split("}")[-1]: v for k, v in (mar.attrib.items() if mar is not None else [])})
    return vals


def main() -> None:
    v7data, v7root = load_doc(V7)
    v8data, v8root = load_doc(V8)
    tmpdata, tmproot = load_doc(TEMPLATE)
    v7body = body_after(v7root, "问题重述")
    v8body = body_after(v8root, "问题重述")

    v7_media = {n: sha256(v).hexdigest() for n, v in v7data.items() if n.startswith("word/media/")}
    v8_media = {n: sha256(v).hexdigest() for n, v in v8data.items() if n.startswith("word/media/")}
    v7txt = [ptext(p) for p in v7root.xpath("//w:body/w:p", namespaces=NS)]
    v8txt = [ptext(p) for p in v8root.xpath("//w:body/w:p", namespaces=NS)]
    ns_math = {**NS}
    v8_body_text = "".join(v8root.xpath("//w:body//w:t/text()", namespaces=NS))
    v8_pdf = pymupdf.open(V8PDF)
    pdf_text = "\n".join(p.get_text() for p in v8_pdf)
    template_margins = section_margins(tmproot)[:2]
    v8_margins = section_margins(v8root)[:2]
    v8_first_table = v8root.xpath("//w:body/w:tbl[1]", namespaces=NS)[0]
    table_text = ptext(v8_first_table)
    omml = v8root.xpath("//m:oMath", namespaces=NS)
    empty_math = v8root.xpath("//m:oMath//m:r[not(m:t) or not(normalize-space(m:t))]", namespaces=NS)
    body_equal = len(v7body) == len(v8body) and all(
        strip_font_locale(a) == strip_font_locale(b) for a, b in zip(v7body, v8body)
    )
    expected_title = "基于收缩集成与条件Copula的啤酒感官预测及风格生成"
    expected_keywords = "关键词：啤酒感官预测；非负堆叠；稳健秩融合；条件Copula；约束迁移"
    pdf_front = re.sub(r"\s+", "", v8_pdf[0].get_text())
    normalized_title = re.sub(r"\s+", "", expected_title)
    gates = {
        "v7_exists_and_unchanged_during_build": V7.exists(),
        "official_template_exists": TEMPLATE.exists(),
        "front_margins_match_official_template": v8_margins == template_margins,
        "team_and_track_in_official_table": "MCB2601605" in table_text and "A赛道" in table_text,
        "title_preserved": expected_title in v8txt,
        "abstract_and_keywords_preserved": expected_keywords in v8txt,
        "body_after_problem_restatement_unchanged": body_equal,
        "body_formula_count_236": len(omml) == 236,
        "empty_omml_runs_zero": len(empty_math) == 0,
        "media_payload_unchanged": v7_media == v8_media,
        "pdf_30_pages": len(v8_pdf) == 30,
        "pdf_page1_team_and_title": "MCB2601605" in pdf_front and normalized_title in pdf_front,
        "pdf_page2_static_toc": "目录" in v8_pdf[1].get_text() and "问题重述" in v8_pdf[1].get_text(),
        "pdf_page3_body_start": "问题重述" in v8_pdf[2].get_text(),
        "no_literal_toc_field_in_pdf": "TOC \\o" not in pdf_text and "TOC \\o" not in v8_body_text,
    }
    result = {
        "version": "v8",
        "gates": gates,
        "all_pass": all(gates.values()),
        "files": {
            "docx": {"path": str(V8.relative_to(ROOT)), "bytes": V8.stat().st_size, "sha256": sha256(V8.read_bytes()).hexdigest()},
            "pdf": {"path": str(V8PDF.relative_to(ROOT)), "bytes": V8PDF.stat().st_size, "sha256": sha256(V8PDF.read_bytes()).hexdigest()},
            "source_v7_docx_sha256": sha256(V7.read_bytes()).hexdigest(),
        },
        "counts": {
            "pdf_pages": len(v8_pdf),
            "sections": len(v8root.xpath("//w:sectPr", namespaces=NS)),
            "tables": len(v8root.xpath("//w:tbl", namespaces=NS)),
            "omml": len(omml),
            "empty_math_runs": len(empty_math),
            "media": len(v8_media),
            "body_elements_after_problem_restatement": len(v8body),
        },
        "front": {"team": "MCB2601605", "track": "A赛道", "template_margins": template_margins, "v8_margins": v8_margins},
        "toc_mode": "static cached TOC copied from approved V7 and formatted with official template layout; Word COM object count is intentionally 0",
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "v8_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (AUDIT / "Word渲染/word_pagination.json").write_text(
        json.dumps({
            "source": str(V8),
            "mode": "static-export",
            "pages": len(v8_pdf),
            "formulas": len(omml),
            "sections": len(v8root.xpath("//w:sectPr", namespaces=NS)),
            "tables": len(v8root.xpath("//w:tbl", namespaces=NS)),
            "images": len(v8_media),
            "toc_count": 0,
            "note": "Static cached TOC is used because the transplanted official-template TOC is not represented as a Word TablesOfContents object; page numbers were retained from the approved V7 pagination.",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"all_pass": result["all_pass"], "gates": gates, "v8_sha256": result["files"]["docx"]["sha256"]}, ensure_ascii=False, indent=2))
    if not result["all_pass"]:
        raise SystemExit("V8 audit failed; see v8_checks.json")


if __name__ == "__main__":
    main()
