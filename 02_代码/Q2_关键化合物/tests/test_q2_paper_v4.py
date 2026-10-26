"""Regressions for a visual-only revision of frozen scientific results."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "05_论文/Q2_关键化合物"
FIG = ROOT / "04_图表/Q2_关键化合物/v4"
AUDIT = ROOT / "03_结果/Q2_关键化合物/publication_audit"
NS = {"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m":"http://schemas.openxmlformats.org/officeDocument/2006/math"}
sys.path.insert(0,str(ROOT/"02_代码/Q2_关键化合物/src"))
sys.path.insert(0,str(ROOT/"scripts"))


def xml(version):
    with ZipFile(PAPER/f"第二问_论文正文_v{version}.docx") as archive:
        return etree.fromstring(archive.read("word/document.xml"))


def test_v4_preserves_frozen_results_and_all_v3_deliverables():
    expected=json.loads((AUDIT/"q2_v4_frozen_inputs.json").read_text())
    assert len(expected)>=50
    assert {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in expected}==expected


def test_v4_only_changes_approved_figure_text_and_preserves_tables_equations():
    from build_q2_v4_handoff import FIGURE_SENTENCE, CAPTION_CHANGE, NOTES
    old,new=xml(3),xml(4)
    for path in (".//m:oMath",".//w:tbl"):
        assert [etree.tostring(x) for x in old.xpath(path,namespaces=NS)]==[etree.tostring(x) for x in new.xpath(path,namespaces=NS)]
    def paragraphs(tree):
        return ["".join(p.xpath(".//w:t/text()",namespaces=NS)) for p in tree.xpath("./w:body/w:p",namespaces=NS)]
    expected=[p.replace(*FIGURE_SENTENCE).replace(*CAPTION_CHANGE) for p in paragraphs(old)]
    assert [p for p in paragraphs(new) if p not in NOTES.values()]==expected
    equations=new.xpath(".//m:oMath",namespaces=NS)
    assert len(equations)==7
    assert all("".join(r.xpath("m:t/text()",namespaces=NS)).strip() for e in equations for r in e.xpath(".//m:r",namespaces=NS))


def test_v4_binned_distributions_preserve_all_candidate_counts():
    source=pd.read_csv(FIG/"source_data/图3_候选稳定性.csv")
    bins=pd.read_csv(FIG/"source_data/图3_稳定性分箱汇总.csv")
    assert len(source)==150 and bins.groupby("指标").n.sum().eq(150).all()
    for metric,group in bins.groupby("指标"):
        assert group.groupby("化学证据等级").n.sum().to_dict()==source["化学证据等级"].value_counts().to_dict()
        for _,row in source.iterrows():
            assert 0<=row[metric]<=1
        assert (group.groupby("化学证据等级")["组内占比"].sum()-1).abs().max()<1e-12


def test_v4_row_labels_rank_and_oav_do_not_shift():
    source=pd.read_csv(FIG/"source_data/图4_A级关系名次与OAV.csv")
    rendered=pd.read_csv(FIG/"source_data/图4_逐行排版核验.csv",keep_default_na=False)
    assert rendered["关系"].tolist()==source["关系"].tolist()
    assert rendered["行"].tolist()==list(range(8,0,-1))
    for col in ("总关联","分层自助","风格校正","逐风格删除"):
        assert rendered[col].tolist()==source[col].tolist()
    assert rendered["OAV显示"].tolist()==["1.20 [0.85, 1.64]","1.20 [0.85, 1.64]","0 [0, 0.36]",
        "0.35 [0.27, 0.48]","4.14 [3.26, 4.90]","4.14 [3.26, 4.90]","N/A","0.35 [0.27, 0.48]"]


def test_v4_source_tables_preserve_v3_numeric_columns():
    for source in (FIG.parent/"v3/source_data").glob("*.csv"):
        old=pd.read_csv(source);new=pd.read_csv(FIG/"source_data"/source.name)
        pd.testing.assert_frame_equal(old,new[old.columns])


def test_v4_geometry_and_q1_font_consistency():
    checks=json.loads((AUDIT/"q2_v4_visual_checks.json").read_text())
    assert checks["families"]==["SimSun"]
    assert min(checks["font_sizes_pt"])>=10.5
    assert checks["text_boxes"]>=200
    assert checks["text_overlap_count"]==checks["text_outside_count"]==0
    boxes=pd.read_csv(AUDIT/"q2_v4_text_boxes.csv")
    labels=set(boxes.loc[boxes.figure==4,"label"])
    assert {"名次","中位数","四分位区间","OAV＝1"}<=labels
    expected=[(2220,1215),(2220,1380),(2220,1215),(2460,1380)]
    for i,size in enumerate(expected,1):
        path=next(FIG.glob(f"图{i}_*_v4.png"))
        with Image.open(path) as image:
            assert image.size==size
            assert min(image.info["dpi"])>=299
        record=next(r for r in checks["figures"] if r["name"]==path.name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record["sha256"]


def test_v4_delivery_and_figure_sources_are_clean():
    from audit_paper_language import audit
    package=ROOT/"05_论文/交付给论文手/Q2_关键化合物_交付包_v4.zip"
    result=audit([package,FIG])
    assert not result["hits"] and not result["unhandled"]
    with ZipFile(package) as z:
        names=z.namelist()
        assert len([n for n in names if n.endswith('.png')])==4
        assert all('_v3' not in n for n in names)
        assert any(n.endswith('Q2_论文审计合订本_v4.pdf') for n in names)
        for path in PAPER.glob('*_v4.pdf'):
            assert z.read('Q2_关键化合物/'+path.name)==path.read_bytes()
        for path in FIG.glob('图*_v4.png'):
            assert z.read('Q2_关键化合物/图片/'+path.name)==path.read_bytes()
