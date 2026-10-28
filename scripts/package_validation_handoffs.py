"""Build dated review handoffs from approved source assets and new manuscripts.

Keeps old archives untouched. Source-ID mapping is saved only in the internal audit.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import gzip
import hashlib
import json
import re
import shutil
import sys
import pandas as pd
from pypdf import PdfReader, PdfWriter
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/collab/20261027_模型检验独立验收"
DATE_DIR = "待审_20261027"
OUT = ROOT / "05_论文/交付给论文手" / DATE_DIR
ID_MAP = {
    "E000":"data_contract","E001":"ridge","E002":"pls2","E003":"extra_trees",
    "E004":"ordinal_boosting","E005":"continuous_ensemble","E006":"continuous_ordinal",
    "E007":"adaptive_shrinkage","E008":"tabular_candidate","E009":"final_nnls",
    "E101":"marginal_spearman","E102":"pls_vip","E103":"elastic_net","E104":"graph_en",
    "E105":"tree_attribution","E106":"association_consensus","E107":"bootstrap",
    "E108":"style_sensitivity","E109":"chemical_reference","E110":"candidate_audit",
    "E111":"chemical_check","E112":"robust_final_ranking","E113":"partial_graph",
    "E114":"conditional_effect","E115":"final_attribution","E116":"conditional_primitive",
    "E117":"half_sample_control","E118":"fusion_ablation","E119":"chemical_extension",
    "E200":"chemical_contract","E201":"empirical_baselines","E202":"distribution_baselines",
    "E203":"structural_ablation","E204":"initial_migration","E205":"sensory_profiles",
    "E206":"proxy_search","E207":"locked_candidates","E208":"fixed_spectrum_sensitivity",
    "E209":"joint_generation","E210":"correlation_sensitivity","E211":"local_mixtures",
    "E212":"external_directions","E213":"residual_generation","E214":"floor_margins",
    "E215":"alternative_mixture","E216":"endpoint_sensitivity","E217":"migration_sensitivity",
}
TEXT={".py",".r",".md",".txt",".csv",".json",".yaml",".yml",".svg",".tex"}


def neutral(text):
    # Keep all numeric payloads and rankings. Only source identifiers are renamed.
    for key in sorted(ID_MAP,key=len,reverse=True):
        text=re.sub(re.escape(key),ID_MAP[key],text,flags=re.I)
    return text


def clean_data(name,data):
    if name.endswith(".csv.gz"):
        return gzip.compress(neutral(gzip.decompress(data).decode("utf-8-sig")).encode(),mtime=0)
    if Path(name).suffix.lower() in TEXT:
        return neutral(data.decode("utf-8-sig")).encode("utf-8")
    return data


def base_assets(path):
    assets={}
    with ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("/"): continue
            short=name.split("/",1)[1]
            if Path(short).suffix in {".docx",".pdf"} and "/" not in short: continue
            if short in {"SHA256SUMS.txt","交付说明.md","当前版本说明.md","当前版本说明_v2.md"}: continue
            if "图清单" in short: continue
            # The incorrectly named old file is republished with its actual scope.
            if short=="数据与结果/最终感官模型Top3删除敏感性.csv":
                short="数据与结果/最终感官模型_折内条件字段遮蔽敏感性.csv"
            assets[short]=z.read(name)
    return assets


def merge_pdf(folder,q,version):
    order=[next(folder.glob("第*正文*.pdf")),next(folder.glob("模型假设*.pdf")),next(folder.glob("问题分析*.pdf"))]
    writer=PdfWriter()
    for path in order:
        writer.append(str(path))
    writer.add_metadata({"/Title":q+"论文审计合订本","/Author":""})
    path=folder/f"{q[:2]}_论文审计合订本_{version}_待审.pdf"
    with path.open("wb") as stream: writer.write(stream)
    return path


def add_folder(assets,source,prefix,filter_fn=lambda p: True):
    if source.exists():
        for path in source.rglob("*"):
            if path.is_file() and filter_fn(path):
                assets[prefix+str(path.relative_to(source))]=path.read_bytes()


def q1_data(assets):
    data=ROOT/"05_论文/Q1_感官评分/数据与结果"
    add_folder(assets,data/"validation_v1","数据与结果/validation_v1/",lambda p: p.suffix==".csv")
    add_folder(assets,data/"validation_ablation_v1","数据与结果/validation_ablation_v1/",lambda p: p.suffix==".csv")
    note={"calibration":"其余重复使用同一250个真值，仅为样本内校准检查。",
          "target_metrics":"逐目标可靠性使用mean-OOF；重复稳定性使用逐重复宏均值。",
          "small_retraining":"消融和留一风格只核对源CSV、日志及实现；未保存原始预测，不能从预测独立复算。",
          "label_permutation":"全局一次目标整行置换后执行固定CV，非训练折内独立置换。"}
    assets["数据与结果/validation_ablation_v1/证据边界.json"]=json.dumps(note,ensure_ascii=False,indent=2).encode()


def q2_data(assets):
    original=ROOT/"05_论文/Q2_关键化合物/数据与结果/validation_v1"
    # Source completeness, controls, anchors, and sensitivity are valid; the old mixed-scope summary is excluded.
    for name in ["q2_validation_completeness.csv","q2_validation_negative_controls.csv","q2_validation_anchor.csv","q2_validation_deletion.csv"]:
        assets["数据与结果/validation_v1/"+name]=(original/name).read_bytes()
    for path in (AUDIT/"Q2").glob("*.csv"):
        assets["数据与结果/validation_v1/"+path.name]=path.read_bytes()
    note={"stability":"bootstrap选出集与四路融合正式答案分别报告；总关联参考集单列。",
          "masking":"固定最终预测器遮蔽折内条件效应前三字段，不是正式答案前三字段的删除检验。",
          "shadow":"2000条记录共6000个Top-3槽，影子1524、真实4476；非零误选。",
          "chemical":"化学证据分级只沿用既有冻结核验；不生成新的化学分级。"}
    assets["数据与结果/validation_v1/检验口径.json"]=json.dumps(note,ensure_ascii=False,indent=2).encode()


def figure_assets(assets,q,folder):
    # New figure assets and their source CSV/R scripts are reviewable and reproducible.
    for path in folder.glob("*"):
        if path.suffix.lower() in {".png",".pdf",".svg"} and path.name.startswith(("图","附图")):
            assets["图片/"+path.name]=path.read_bytes()
    add_folder(assets,folder/"source_data","图源/数据/")
    for path in folder.glob("*.csv"):
        if "图清单" not in path.name:
            assets["图源/数据/"+path.name]=path.read_bytes()
    for path in folder.glob("*.R"):
        assets["图源/"+path.name]=path.read_bytes()
    if (folder/"figure_contract.md").exists():
        assets["图源/figure_contract.md"]=(folder/"figure_contract.md").read_bytes()
    assets["图源/plot_style.R"]=(ROOT/"scripts/plot_style.R").read_bytes()
    # QA panels stay outside the delivery package; they are not paper figures.
    images=sorted(n for n in assets if n.startswith("图片/") and n.endswith(".png"))
    doc=next((ROOT/"05_论文"/q/DATE_DIR).glob("第*正文*.docx"))
    with ZipFile(doc) as z:
        xml=etree.fromstring(z.read("word/document.xml"))
    wns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    titles={}
    for p in xml.xpath("./w:body/w:p",namespaces=wns):
        text="".join(p.xpath(".//w:t/text()",namespaces=wns))
        m=re.match(r"(附图S\d|图\d)\s+(.+)",text)
        if m: titles[m.group(1)]=m.group(2)
    rows=[]
    for name in images:
        stem=Path(name).stem
        figure_id=re.match(r"(?:附图S\d|图\d)",stem).group()
        rows.append({"图号":figure_id,"文件":name,
                     "图题":titles[figure_id],
                     "状态":"待用户审计","口径":"正文图注及图源CSV"})
    assets[f"图片/{q[:2]}_图清单.csv"]=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def build(q,version,base):
    folder=ROOT/"05_论文"/q/DATE_DIR
    assets=base_assets(base)
    merge_pdf(folder,q,version)
    for path in folder.glob("*"):
        if path.suffix in {".docx",".pdf"}: assets[path.name]=path.read_bytes()
    if q.startswith("Q1"): q1_data(assets)
    elif q.startswith("Q2"): q2_data(assets)
    else: add_folder(assets,folder/"数据与结果/validation_v1","数据与结果/validation_v1/")
    figure_assets(assets,q,ROOT/"04_图表"/q/DATE_DIR)
    assets["程序/核验交付.py"]=(ROOT/"scripts/verify_publication_handoff.py").read_bytes()
    changes={
        "Q1":"基于原正式正文追加校准、区间、逐目标可靠性表7、图5及附图S2/S3；MAE改点图，目标中文映射；样本内校准和单次全局置换按实际说明。",
        "Q2":"基于已审正文和图1—图4追加表5/图5；bootstrap候选集LOSO改为独立复算0.6582，正式答案为0.6076；遮蔽准确标注折内条件效应前三字段。",
        "Q3":"基于当前正文追加检验汇总与结构消融、固定谱逐字段置换负对照和已锁候选裕量敏感性；维持Copula主结果。"
    }
    bounds={
        "Q1":"重复残差校准不是独立验证；17/16/17为相对三分位；新增消融无原始预测，只有源CSV/实现/日志核对。",
        "Q2":"影子字段占6000槽的25.4%，不能称零误选；CO₂的bootstrap名次第4而最终融合名次第3；遮蔽不是正式答案充分性。",
        "Q3":"合成画像和统计候选不等于实际可酿造、安全、唯一或全局最优；小风格仅描述，未见风格外推未验证；两候选来自同一源且对裕量敏感。"
    }
    readme=f"# {q} 论文手交付说明\n\n状态：{version} 待审稿，尚未经用户确认替换旧正式版。\n\n## 版本变化\n\n{changes[q[:2]]}\n\n## 使用边界\n\n{bounds[q[:2]]}\n\n## 组成与使用\n\n正文DOCX、Word同源PDF、审计合订本、模型假设、问题分析、图片/图清单、结果表和检验数据均已包含。正文优先核对表格与图题；附录核心代码交论文手编排。新图须经用户审图后方可认定为正式稿。主模型、旧预测、Top-3排序均沿用冻结源。\n\nSHA256SUMS.txt逐文件列出本包字节摘要；复核时在包根目录运行sha256sum -c SHA256SUMS.txt。来源文件的中性名称与哈希见来源校验.json；内部编号映射仅留仓库验收记录。\n"
    assets["交付说明.md"]=readme.encode()
    assets[f"版本变化_{version}.md"]=(changes[q[:2]]+"\n\n"+bounds[q[:2]]+"\n").encode()
    clean={neutral(n):clean_data(n,data) for n,data in assets.items()}
    provenance={n:hashlib.sha256(data).hexdigest() for n,data in clean.items() if n.startswith("数据与结果/")}
    clean["来源校验.json"]=json.dumps(provenance,ensure_ascii=False,indent=2).encode()
    lines=[hashlib.sha256(data).hexdigest()+"  "+n for n,data in sorted(clean.items())]
    clean["SHA256SUMS.txt"]=("\n".join(lines)+"\n").encode()
    target=OUT/q
    target.mkdir(parents=True,exist_ok=True)
    for name,data in clean.items():
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    zpath=OUT/f"{q}_交付包_{version}_待审.zip"
    with ZipFile(zpath,"w",ZIP_DEFLATED) as z:
        for name,data in sorted(clean.items()): z.writestr(q+"/"+name,data)
    return {"question":q,"version":version,"path":str(zpath.relative_to(ROOT)),"files":len(clean),
            "bytes":zpath.stat().st_size,"sha256":hashlib.sha256(zpath.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    profiles=[
        ("Q1_感官评分","v6",ROOT/"05_论文/交付给论文手/Q1_感官评分_交付包.zip"),
        ("Q2_关键化合物","v5",ROOT/".worktrees/q2-evidence-opt/05_论文/交付给论文手/Q2_关键化合物_交付包_v4.zip"),
        ("Q3_风格生成","v3",ROOT/".worktrees/q3-foundation/05_论文/交付给论文手/Q3_风格生成_交付包_v2.zip")]
    results=[build(*profile) for profile in profiles]
    (AUDIT/"package_summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    (AUDIT/"internal_source_id_mapping.json").write_text(json.dumps(ID_MAP,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
