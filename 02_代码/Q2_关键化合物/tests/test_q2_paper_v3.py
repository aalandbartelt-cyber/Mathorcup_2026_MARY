from __future__ import annotations

import hashlib
import importlib
import re
import sys
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "05_论文/Q2_关键化合物"
FIGURE = ROOT / "04_图表/Q2_关键化合物/v3"
PACKAGE = ROOT / "05_论文/交付给论文手/Q2_关键化合物_交付包_v3.zip"
MAIN_DOCX = PAPER / "第二问_论文正文_v3.docx"
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"m": MATH_NS, "w": WORD_NS}
SRC = ROOT / "02_代码/Q2_关键化合物/src"
sys.path.insert(0, str(SRC))

FIGURES = (
    "图1_共表达图阈值敏感性_v3.png",
    "图2_模型性能与四路一致性_v3.png",
    "图3_候选稳定性与证据等级_v3.png",
    "图4_A级关系四路排名与OAV_v3.png",
)

BANNED = re.compile(
    r"(?<![A-Za-z0-9])E\d{3}(?![A-Za-z0-9])|参考组|对照组|晋级|门槛|promotion|\bpass\b|BZD|agent[23]?|学长版",
    flags=re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_xml(path: Path) -> etree._Element:
    assert path.exists(), f"missing v3 DOCX: {path}"
    with ZipFile(path) as archive:
        return etree.fromstring(archive.read("word/document.xml"))


def _math_text(equation: etree._Element) -> str:
    return "".join(equation.xpath(".//m:t/text()", namespaces=NS))


def test_v2_frozen_artifacts_are_unchanged() -> None:
    expected = {
        "05_论文/Q2_关键化合物/第二问_论文正文_v2.docx": "fa763808cb3286183fb0b420090c9f1dc2a3b2e574c90746a6aa91d017530c84",
        "05_论文/Q2_关键化合物/第二问_论文正文_v2.pdf": "005d8b3cc2c3948bb62bae61375e7fde635210ec0e9a9a2b3e2e76ee6c53f827",
        "05_论文/交付给论文手/Q2_关键化合物_交付包_v2.zip": "1f0ed3287364a432c080355b280ce1e81b0296e03debed0f72104a31507f200d",
    }
    assert {relative: _sha256(ROOT / relative) for relative in expected} == expected


def test_bootstrap_formula_builder_has_no_empty_run_and_complete_limits() -> None:
    module = importlib.import_module("q2_omml")
    equation = etree.Element(f"{{{MATH_NS}}}oMath", nsmap={"m": MATH_NS})
    for node in module.selection_frequency():
        equation.append(node)
    assert _math_text(equation).replace(" ", "") == "πkj=1Bb=1BIk∈Tbj"
    assert equation.xpath("count(.//m:f)", namespaces=NS) == 1
    assert equation.xpath("count(.//m:nary)", namespaces=NS) == 1
    assert equation.xpath("count(.//m:sSubSup)", namespaces=NS) == 1
    assert equation.xpath("count(.//m:d)", namespaces=NS) == 1
    for run in equation.xpath(".//m:r", namespaces=NS):
        values = run.xpath("./m:t/text()", namespaces=NS)
        assert values and "".join(values).strip()
        assert run.xpath("./w:rPr/w:rFonts[@w:ascii='Cambria Math']", namespaces=NS)


def test_v3_docx_has_seven_complete_native_equations() -> None:
    root = _document_xml(MAIN_DOCX)
    equations = root.xpath(".//m:oMath", namespaces=NS)
    assert len(equations) == 7

    required_structures = (
        {"sSub"},
        {"f", "nary", "sSub", "sSup", "bar", "d"},
        {"f", "sSub", "sSup", "d"},
        {"sSub", "sSup", "d"},
        {"f", "nary", "sSub"},
        {"f", "nary", "sSubSup", "sSub"},
        {"f", "sSub"},
    )
    required_text = (
        ("y", "f", "ε"),
        ("ρ", "R", "n"),
        ("β", "λ", "L"),
        ("Δ", "MAE", "f"),
        ("S", "60", "r"),
        ("π", "B", "b", "I", "∈", "T", "k", "j"),
        ("OAV", "c", "τ"),
    )

    for index, equation in enumerate(equations):
        tags = {etree.QName(node).localname for node in equation.iter()}
        assert required_structures[index] <= tags, (index + 1, tags)
        text = _math_text(equation)
        assert all(token in text for token in required_text[index]), (index + 1, text)
        for run in equation.xpath(".//m:r", namespaces=NS):
            values = run.xpath("./m:t/text()", namespaces=NS)
            assert values and "".join(values).strip(), (index + 1, etree.tostring(run))
            fonts = run.xpath("./w:rPr/w:rFonts", namespaces=NS)
            assert fonts, (index + 1, etree.tostring(run))
            assert fonts[0].get(f"{{{WORD_NS}}}ascii") == "Cambria Math"

    formula6 = equations[5]
    assert formula6.xpath("count(.//m:f)", namespaces=NS) == 1
    assert formula6.xpath("count(.//m:nary)", namespaces=NS) == 1
    assert formula6.xpath("count(.//m:sSubSup)", namespaces=NS) >= 1


def test_v3_figures_are_3_by_2_300_dpi_and_have_source_contract() -> None:
    assert (FIGURE / "figure_contract.md").exists()
    assert (FIGURE / "source_data_manifest.json").exists()
    assert (FIGURE / "plot_q2_paper_v3.R").exists()
    for filename in FIGURES:
        path = FIGURE / filename
        assert path.exists(), path
        with Image.open(path) as image:
            width, height = image.size
            assert abs(width / height - 1.5) < 0.01
            dpi = image.info.get("dpi", (0, 0))
            assert min(dpi) >= 295


def test_v3_figure_source_and_filenames_have_no_banned_terms() -> None:
    assert FIGURE.exists()
    for path in FIGURE.rglob("*"):
        assert not BANNED.search(path.name), path
        if path.is_file() and path.suffix.lower() in {".r", ".csv", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not BANNED.search(text), path


def test_v3_package_has_no_banned_terms_in_names_or_text() -> None:
    assert PACKAGE.exists(), PACKAGE
    with ZipFile(PACKAGE) as archive:
        for name in archive.namelist():
            assert not BANNED.search(name), name
            if Path(name).suffix.lower() in {".md", ".csv", ".json", ".txt", ".py", ".r"}:
                text = archive.read(name).decode("utf-8", errors="ignore")
                assert not BANNED.search(text), name
