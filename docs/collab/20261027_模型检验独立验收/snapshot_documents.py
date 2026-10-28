"""Read-only inventory of pre-existing manuscripts and packages; no generation."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
from lxml import etree
import pymupdf

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}


def inspect(path):
    data = path.read_bytes()
    result = {"path": str(path.relative_to(ROOT)), "bytes": len(data),
              "sha256": hashlib.sha256(data).hexdigest(), "role": "existing_source_only"}
    if path.suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            result["pages"] = len(pdf)
            result["captions"] = [
                {"page": i + 1, "text": line.strip()}
                for i, page in enumerate(pdf) for line in page.get_text().splitlines()
                if line.strip().startswith(("表", "图", "附图"))
            ]
    elif path.suffix == ".docx":
        with ZipFile(path) as z:
            xml = etree.fromstring(z.read("word/document.xml"))
            tables = xml.xpath(".//w:tbl", namespaces=NS)
            result.update(
                omml=len(xml.xpath(".//m:oMath", namespaces=NS)),
                empty_math_runs=len(xml.xpath(".//m:oMath//m:r[not(m:t) or not(normalize-space(m:t))]", namespaces=NS)),
                drawings=len(xml.xpath(".//w:drawing", namespaces=NS)),
                tables=len(tables),
                content_tables=sum(not bool(t.xpath(".//m:oMath", namespaces=NS)) for t in tables),
            )
    elif path.suffix == ".zip":
        with ZipFile(path) as z:
            result["files"] = sum(not n.endswith("/") for n in z.namelist())
            result["zip_error"] = z.testzip()
    return result


def main():
    paths = set()
    for base in [ROOT, ROOT / ".worktrees/q2-evidence-opt", ROOT / ".worktrees/q3-foundation"]:
        paper = base / "05_论文"
        if not paper.exists():
            continue
        for q in ["Q1_感官评分", "Q2_关键化合物", "Q3_风格生成"]:
            folder = paper / q
            if folder.exists():
                paths.update(p for p in folder.glob("*") if p.suffix in {".docx", ".pdf"})
        handoff = paper / "交付给论文手"
        if handoff.exists():
            paths.update(handoff.glob("*.zip"))
    records = [inspect(p) for p in sorted(paths)]
    output = OUT / "existing_document_snapshot.json"
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded {len(records)} existing documents/packages: {output}")


if __name__ == "__main__":
    main()
