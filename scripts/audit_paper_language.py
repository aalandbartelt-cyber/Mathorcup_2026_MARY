"""Scan publication text, filenames and nested archives for internal terminology."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import argparse
import gzip
import json
import re
from lxml import etree

PATTERN = re.compile(
    r"(?<![A-Za-z0-9])E\d{3}(?![A-Za-z0-9])|参考组|对照组|晋级|门槛|promotion|\bpass\b|BZD|agent(?:2|3)?|学长版",
    re.I,
)
TEXT_EXT = {".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".py", ".r", ".tex", ".xml", ".svg", ".html", ".bib", ".toml", ".ini"}


def text_hits(text, location):
    return [
        {"location": location, "term": match.group(0), "context": text[max(0, match.start() - 35):match.end() + 45]}
        for match in PATTERN.finditer(text)
    ]


def scan_blob(name, data, records):
    records["files"].append(name)
    records["hits"] += text_hits(name, name + " [filename]")
    ext = Path(name).suffix.lower()
    if ext in {".docx", ".zip"}:
        with ZipFile(BytesIO(data)) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                blob = archive.read(member)
                location = name + "!" + member
                if ext == ".docx" and (member.endswith(".xml") or member.endswith(".rels")):
                    tree = etree.fromstring(blob)
                    records["files"].append(location)
                    records["hits"] += text_hits(member, location + " [filename]")
                    records["hits"] += text_hits("".join(tree.itertext()), location)
                    for node in tree.iter():
                        for value in node.attrib.values():
                            records["hits"] += text_hits(value, location + " [attribute]")
                elif ext == ".zip":
                    scan_blob(location, blob, records)
                elif member.startswith("word/media/"):
                    records["images"].append(location)
    elif ext == ".gz":
        scan_blob(name[:-3], gzip.decompress(data), records)
    elif ext == ".pdf":
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            for number, page in enumerate(pdf, start=1):
                records["hits"] += text_hits(page.get_text(), f"{name} [page {number}]")
            records["hits"] += text_hits(json.dumps(pdf.metadata, ensure_ascii=False), name + " [metadata]")
    elif ext in TEXT_EXT or Path(name).name in {"SHA256SUMS", "LICENSE", ".gitkeep"}:
        records["hits"] += text_hits(data.decode("utf-8-sig"), name)
    elif ext == ".png":
        from PIL import Image

        image = Image.open(BytesIO(data))
        records["hits"] += text_hits(json.dumps(image.info, default=str, ensure_ascii=False), name + " [metadata]")
        records["images"].append(name)
    elif ext in {".npz", ".npy"}:
        records["numeric_archives"].append(name)
    else:
        records["unhandled"].append(name)


def audit(paths):
    records = {"files": [], "hits": [], "images": [], "numeric_archives": [], "unhandled": []}
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
        for item in files:
            scan_blob(str(item), item.read_bytes(), records)
    records["scope"] = [str(path) for path in paths]
    records["hit_count"] = len(records["hits"])
    records["image_text_method"] = "PNG metadata scan; visible labels checked from plotting source/CSV and visual rendering, no OCR claim"
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"hit_count": result["hit_count"], "entries": len(result["files"]), "unhandled": result["unhandled"], "report": str(args.output)}, ensure_ascii=False))
    raise SystemExit(1 if result["hits"] or result["unhandled"] else 0)


if __name__ == "__main__":
    main()
