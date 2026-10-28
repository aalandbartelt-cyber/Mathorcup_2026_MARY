"""Clone approved manuscripts and add independently verified validation sections.

Only writes the dated review directories. Frozen source documents are read-only.
Run after the evidence/figure preparation has completed.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import re
import shutil

from lxml import etree as E
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/collab/20261027_模型检验独立验收"
DATE_DIR = "待审_20261027"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"}


def tag(n):
    prefix, local = n.split(":")
    return "{" + NS[prefix] + "}" + local


def txt(element):
    return "".join(element.xpath(".//w:t/text()", namespaces=NS))


def prop(element, name, value=None):
    node = element.find(tag(name))
    if node is None:
        node = E.SubElement(element, tag(name))
    if value is not None:
        node.set(tag("w:val"), str(value))
    return node


def ppr(p):
    x = p.find(tag("w:pPr"))
    if x is None:
        x = E.Element(tag("w:pPr")); p.insert(0, x)
    return x


class Editor:
    def __init__(self, source):
        self.source = source
        with ZipFile(source) as z:
            self.files = {n: z.read(n) for n in z.namelist()}
        self.root = E.fromstring(self.files["word/document.xml"])
        self.body = self.root.find(tag("w:body"))
        self.rels = E.fromstring(self.files["word/_rels/document.xml.rels"])
        self.content_types = E.fromstring(self.files["[Content_Types].xml"])
        self.templates = {}
        for p in self.body.findall(tag("w:p")):
            s = p.find("w:pPr/w:pStyle", NS)
            if s is not None and s.get(tag("w:val")) not in self.templates:
                self.templates[s.get(tag("w:val"))] = deepcopy(p)
        self.caption = next(deepcopy(p) for p in self.body.findall(tag("w:p"))
                            if re.match(r"图\d\s", txt(p)))
        self.picture = next(deepcopy(p) for p in self.body.findall(tag("w:p"))
                            if p.find(".//w:drawing", NS) is not None)
        self.data_table = next(deepcopy(t) for t in self.body.findall(tag("w:tbl"))
                               if not t.xpath(".//m:oMath", namespaces=NS))

    def paragraph(self, text, style="BodyText", caption=False):
        p = deepcopy(self.caption if caption else self.templates[style])
        props = p.find(tag("w:pPr"))
        first = p.find("w:r/w:rPr", NS)
        first = deepcopy(first) if first is not None else None
        for child in list(p):
            if child is not props:
                p.remove(child)
        run = E.SubElement(p, tag("w:r"))
        if first is not None:
            run.append(first)
        t = E.SubElement(run, tag("w:t")); t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        for name in ["w:pageBreakBefore", "w:keepNext"]:
            n = ppr(p).find(tag(name))
            if n is not None:
                ppr(p).remove(n)
        if style.startswith("Heading") or (caption and text.startswith("表")):
            prop(ppr(p), "w:keepNext")
        prop(ppr(p), "w:keepLines")
        return p

    def find(self, value):
        return next(p for p in self.body.findall(tag("w:p")) if txt(p).strip() == value)

    def insert(self, anchor, nodes):
        for n in nodes:
            anchor.addprevious(n)

    def replace_text(self, old, new):
        p = next(p for p in self.body.xpath(".//w:p", namespaces=NS) if txt(p).strip()==old)
        replacement=deepcopy(p)
        properties=replacement.find(tag("w:pPr"))
        run_properties=replacement.find("w:r/w:rPr", NS)
        run_properties=deepcopy(run_properties) if run_properties is not None else None
        for child in list(replacement):
            if child is not properties: replacement.remove(child)
        run=E.SubElement(replacement,tag("w:r"))
        if run_properties is not None: run.append(run_properties)
        t=E.SubElement(run,tag("w:t")); t.text=new
        t.set("{http://www.w3.org/XML/1998/namespace}space","preserve")
        p.getparent().replace(p,replacement)

    def table(self, rows, widths=None):
        table = deepcopy(self.data_table)
        oldrows = table.findall(tag("w:tr"))
        htemplate = deepcopy(oldrows[0]); btemplate = deepcopy(oldrows[min(1, len(oldrows)-1)])
        for tr in oldrows:
            table.remove(tr)
        cols = len(rows[0]); widths = widths or [2300] + [3100] * (cols - 2) + [3240]
        total = sum(widths)
        grid = table.find(tag("w:tblGrid"))
        for x in list(grid):
            grid.remove(x)
        for width in widths:
            x = E.SubElement(grid, tag("w:gridCol")); x.set(tag("w:w"), str(width))
        tp = table.find(tag("w:tblPr")); tw = tp.find(tag("w:tblW"))
        if tw is not None:
            tw.set(tag("w:w"), str(total)); tw.set(tag("w:type"), "dxa")
        for ri, row in enumerate(rows):
            tr = deepcopy(htemplate if ri == 0 else btemplate)
            ctemplate = deepcopy(tr.findall(tag("w:tc"))[0])
            for c in tr.findall(tag("w:tc")):
                tr.remove(c)
            for ci, val in enumerate(row):
                c = deepcopy(ctemplate); cp = c.find(tag("w:tcPr"))
                for x in list(c):
                    if x is not cp:
                        c.remove(x)
                cw = cp.find(tag("w:tcW"))
                if cw is not None:
                    cw.set(tag("w:w"), str(widths[ci])); cw.set(tag("w:type"), "dxa")
                p = self.paragraph(str(val)); pp = ppr(p)
                prop(pp, "w:jc", "left")
                for name in ["w:spacing", "w:ind"]:
                    x = pp.find(tag(name))
                    if x is not None:
                        pp.remove(x)
                space = E.SubElement(pp, tag("w:spacing"))
                space.set(tag("w:before"), "0"); space.set(tag("w:after"), "0")
                space.set(tag("w:line"), "220"); space.set(tag("w:lineRule"), "auto")
                ind = E.SubElement(pp, tag("w:ind")); ind.set(tag("w:firstLine"), "0")
                run = p.find(tag("w:r")); rp = run.find(tag("w:rPr"))
                if rp is None:
                    rp = E.Element(tag("w:rPr")); run.insert(0, rp)
                prop(rp, "w:sz", "18"); prop(rp, "w:szCs", "18")
                if ri == 0:
                    prop(rp, "w:b")
                c.append(p); tr.append(c)
            table.append(tr)
        return table

    def image(self, path, caption, width_cm=15.5):
        p = deepcopy(self.picture)
        # The image paragraph keeps the mother document's paragraph formatting.
        draw = p.find(".//w:drawing", NS)
        relid = f"rIdReview{len(self.rels)+1}"
        name = f"review_{len(self.rels)+1}.png"
        self.files["word/media/"+name] = path.read_bytes()
        relns = "http://schemas.openxmlformats.org/package/2006/relationships"
        E.SubElement(self.rels, "{"+relns+"}Relationship", Id=relid,
                     Type=NS["r"]+"/image", Target="media/"+name)
        draw.find(".//a:blip", NS).set(tag("r:embed"), relid)
        width, height = Image.open(path).size
        cx = round(width_cm / 2.54 * 914400); cy = round(cx * height / width)
        ext = draw.find(".//wp:extent", NS); ext.set("cx", str(cx)); ext.set("cy", str(cy))
        for x in draw.findall(".//a:xfrm/a:ext", NS):
            x.set("cx", str(cx)); x.set("cy", str(cy))
        for x in draw.findall(".//wp:docPr", NS):
            x.set("id", str(1000+len(self.rels))); x.set("name", caption)
            x.set("descr", caption)
        prop(ppr(p), "w:keepNext"); prop(ppr(p), "w:keepLines")
        cap = self.paragraph(caption, caption=True)
        return [p, cap]

    def replace_figure(self, index, path):
        draw = self.root.xpath(".//w:drawing", namespaces=NS)[index]
        rid = draw.find(".//a:blip", NS).get(tag("r:embed"))
        member = next(x.get("Target") for x in self.rels if x.get("Id")==rid)
        self.files["word/"+member] = path.read_bytes()

    def keep_groups(self):
        # Join the rows of each content table into one pagination group.
        for table in self.body.findall(tag("w:tbl")):
            rows = table.findall(tag("w:tr"))
            for i, tr in enumerate(rows):
                trp = tr.find(tag("w:trPr"))
                if trp is None:
                    trp=E.Element(tag("w:trPr")); tr.insert(0,trp)
                prop(trp,"w:cantSplit")
                for p in tr.xpath(".//w:p", namespaces=NS):
                    pp=ppr(p); old=pp.find(tag("w:keepNext"))
                    if old is not None: pp.remove(old)
                    if i<len(rows)-1: prop(pp,"w:keepNext")
                    prop(pp,"w:keepLines")
            previous=table.getprevious()
            if previous is not None and previous.tag==tag("w:p") and re.match(r"表\d", txt(previous)):
                prop(ppr(previous),"w:keepNext")
        for p in self.body.findall(tag("w:p")):
            if p.find(".//w:drawing",NS) is not None:
                prop(ppr(p),"w:keepNext"); prop(ppr(p),"w:keepLines")

    def save(self,path):
        self.keep_groups()
        property_orders={
            "pPr":"pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange",
            "trPr":"cnfStyle divId gridBefore gridAfter wBefore wAfter cantSplit trHeight tblHeader tblCellSpacing jc hidden ins del trPrChange",
        }
        for parent_name, names in property_orders.items():
            order={name:i for i,name in enumerate(names.split())}
            for parent in self.root.xpath(".//w:"+parent_name,namespaces=NS):
                children=list(parent)
                parent[:]=sorted(children,key=lambda n:order.get(E.QName(n).localname,999))
        self.files["word/document.xml"]=E.tostring(self.root,encoding="UTF-8",xml_declaration=True,standalone=True)
        self.files["word/_rels/document.xml.rels"]=E.tostring(self.rels,encoding="UTF-8",xml_declaration=True,standalone=True)
        if "docProps/core.xml" in self.files:
            core=E.fromstring(self.files["docProps/core.xml"])
            for child in core:
                if child.tag.split("}")[-1] in {"creator","lastModifiedBy"}: child.text=""
                if child.tag.split("}")[-1]=="title":
                    child.text=txt(self.body.find(tag("w:p")))
            self.files["docProps/core.xml"]=E.tostring(core,encoding="UTF-8",xml_declaration=True,standalone=True)
        path.parent.mkdir(parents=True,exist_ok=True)
        with ZipFile(path,"w",ZIP_DEFLATED) as z:
            for n,data in self.files.items(): z.writestr(n,data)
        return path


def copy_companions(q, source_dir, suffix, version):
    out=ROOT/"05_论文"/q/DATE_DIR
    for prefix in ["模型假设", "问题分析"]:
        src=source_dir/f"{prefix}_{q[:2]}_{suffix}.docx"
        dest=out/f"{prefix}_{q[:2]}_{version}_待审.docx"
        with ZipFile(src) as archive:
            files={n:archive.read(n) for n in archive.namelist()}
        core=E.fromstring(files['docProps/core.xml'])
        question_name={'Q1':'问题一','Q2':'问题二','Q3':'问题三'}[q[:2]]
        title=f'模型假设（{question_name}）' if prefix=='模型假设' else question_name+'分析'
        for node in core:
            local=E.QName(node).localname
            if local=='title':node.text=title
            if local in {'creator','lastModifiedBy'}:node.text=''
        files['docProps/core.xml']=E.tostring(core,encoding='UTF-8',xml_declaration=True,standalone=True)
        with ZipFile(dest,'w',ZIP_DEFLATED) as archive:
            for n,data in files.items():archive.writestr(n,data)


def load_section(path):
    return json.loads(path.read_text())


def add_section(editor,anchor,section):
    nodes=[editor.paragraph(section["heading"],"Heading3")]
    nodes += [editor.paragraph(t) for t in section["paragraphs"]]
    for tab in section.get("tables",[]):
        nodes += [editor.paragraph(tab["caption"],caption=True),editor.table(tab["rows"],tab.get("widths"))]
        if tab.get("note"): nodes.append(editor.paragraph(tab["note"]))
    for fig in section.get("figures",[]):
        nodes += editor.image(ROOT/fig["path"],fig["caption"],fig.get("width_cm",15.5))
        if fig.get("note"): nodes.append(editor.paragraph(fig["note"]))
    editor.insert(anchor,nodes)


def build_q1():
    q="Q1_感官评分"; source=ROOT/"05_论文"/q/"第一问_论文正文_v5.docx"
    doc=Editor(source); section=load_section(AUDIT/"Q1/paper_addition.json")
    add_section(doc,doc.find("结果解释与模型评价"),section)
    sect=doc.body.find(tag("w:sectPr"))
    for fig in section.get("appendix_figures",[]):
        p=doc.paragraph(fig.get("lead",""))
        prop(ppr(p),"w:pageBreakBefore")
        doc.insert(sect,[p]+doc.image(ROOT/fig["path"],fig["caption"],15.8)+[doc.paragraph(fig["note"])])
    out=ROOT/"05_论文"/q/DATE_DIR/"第一问_论文正文_v6_待审.docx"
    doc.save(out); copy_companions(q,source.parent,"v3","v6")
    return out


def build_q2():
    q="Q2_关键化合物"; source=ROOT/".worktrees/q2-evidence-opt/05_论文"/q/"第二问_论文正文_v4.docx"
    doc=Editor(source); section=load_section(AUDIT/"Q2/paper_addition.json")
    for change in section.get("replacements",[]):
        doc.replace_text(change["old"],change["new"])
    # Figures 1–4 come from the accepted source document; only the approved label fix is refreshed.
    f2=ROOT/"04_图表"/q/DATE_DIR/"图2_模型性能与四路一致性_v4.png"
    if f2.exists(): doc.replace_figure(1,f2)
    add_section(doc,doc.find("合理性与局限"),section)
    out=ROOT/"05_论文"/q/DATE_DIR/"第二问_论文正文_v5_待审.docx"
    doc.save(out); copy_companions(q,source.parent,"v4","v5")
    return out


def build_q3():
    q="Q3_风格生成"; source=ROOT/".worktrees/q3-foundation/05_论文"/q/"第三问_论文正文_v2.docx"
    doc=Editor(source); section=load_section(AUDIT/"Q3/paper_addition.json")
    add_section(doc,doc.find("合理性与局限"),section)
    # Fix the existing path notation to x(λ); preserve its native equation container.
    omaths=doc.root.xpath(".//m:oMath",namespaces=NS)
    om=omaths[4]
    import importlib.util
    helper=ROOT/".worktrees/q3-foundation/scripts/q3_omml.py"
    spec=importlib.util.spec_from_file_location("q3_math",helper); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for child in list(om): om.remove(child)
    items=[module.var("x"),module.delimiter([module.var("λ")]), module.roman(" = (1 − "),module.var("λ"),module.roman(")"),module.subscript([module.var("x")],[module.roman("0")]),module.roman(" + "),module.var("λ"),module.var("a"),module.roman(",  0 ≤ "),module.var("λ"),module.roman(" ≤ 1")]
    for child in items: om.append(child)
    out=ROOT/"05_论文"/q/DATE_DIR/"第三问_论文正文_v3_待审.docx"
    doc.save(out); copy_companions(q,source.parent,"v2","v3")
    return out


def main():
    generated=[build_q1(),build_q2(),build_q3()]
    jobs=[]
    for path in generated:
        for doc in sorted(path.parent.glob("*.docx")):
            jobs.append({"source":str(doc),"name":doc.name})
    (AUDIT/"word_export_jobs.json").write_text(json.dumps(jobs,ensure_ascii=False,indent=2)+"\n")
    print("\n".join(str(p) for p in generated))


if __name__=="__main__": main()
