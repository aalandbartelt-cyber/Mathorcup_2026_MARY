from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / '05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v7.docx'
TEMPLATE = Path('/mnt/d/上海大学__/26年妈妈杯数模/2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题/2026年MathorCup数学应用挑战赛—大数据竞赛论文模板.docx')
OUT = ROOT / '05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v8.docx'
N = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W = '{%s}' % N['w']
R = '{%s}' % N['r']


def text(el):
    return ''.join(el.xpath('.//w:t/text()', namespaces=N))


def set_text_single(p, s, template_run=None):
    # Keep pPr and create one ordinary run using template run properties.
    for child in list(p):
        if child.tag != W+'pPr':
            p.remove(child)
    r = E.Element(W+'r')
    if template_run is not None:
        trp = template_run.find(W+'rPr')
        if trp is not None:
            r.append(deepcopy(trp))
    t = E.SubElement(r, W+'t')
    t.text = s
    p.append(r)


def ensure_rpr(r, size=24, bold=None):
    rpr = r.find(W+'rPr')
    if rpr is None:
        rpr = E.Element(W+'rPr'); r.insert(0, rpr)
    # Remove style-only/old font sizing that can override the official front.
    for q in list(rpr):
        if q.tag in {W+'sz', W+'szCs', W+'rFonts'}:
            rpr.remove(q)
    rf = E.Element(W+'rFonts'); rf.set(W+'ascii','Times New Roman'); rf.set(W+'hAnsi','Times New Roman'); rf.set(W+'eastAsia','宋体'); rpr.insert(0, rf)
    sz = E.Element(W+'sz'); sz.set(W+'val',str(size)); rpr.append(sz)
    szc = E.Element(W+'szCs'); szc.set(W+'val',str(size)); rpr.append(szc)
    if bold is not None:
        for q in list(rpr):
            if q.tag in {W+'b', W+'bCs'}: rpr.remove(q)
        if bold:
            rpr.append(E.Element(W+'b')); rpr.append(E.Element(W+'bCs'))
    return rpr


def format_front_body(p, template_ppr, size=24):
    # Use official placeholder paragraph metrics, preserve v7 bold spans.
    old_runs = [deepcopy(x) for x in p if x.tag == W+'r']
    for child in list(p):
        if child.tag != W+'pPr': p.remove(child)
    ppr = deepcopy(template_ppr)
    # Keep no placeholder text and no paragraph style from the source.
    ps = ppr.find(W+'pStyle')
    if ps is not None: ppr.remove(ps)
    p.insert(0, ppr)
    for r in old_runs:
        ensure_rpr(r, size=size)
        p.append(r)
    if not old_runs:
        r=E.SubElement(p,W+'r'); ensure_rpr(r,size=size); E.SubElement(r,W+'t').text=''


def format_toc_p(p, level, template_p):
    # Copy the official level's paragraph layout, without its conflicting style id.
    tppr = template_p.find(W+'pPr')
    ppr = deepcopy(tppr) if tppr is not None else E.Element(W+'pPr')
    ps = ppr.find(W+'pStyle')
    if ps is not None: ppr.remove(ps)
    # official template tabs and explicit level indentation
    for x in list(ppr):
        if x.tag in {W+'ind', W+'tabs', W+'rPr'}: ppr.remove(x)
    tabs=E.SubElement(ppr,W+'tabs'); tab=E.SubElement(tabs,W+'tab'); tab.set(W+'val','right'); tab.set(W+'leader','dot'); tab.set(W+'pos','9070')
    ind=E.SubElement(ppr,W+'ind')
    if level==2: ind.set(W+'left','210')
    elif level==3: ind.set(W+'left','420')
    else: ind.set(W+'left','0')
    rpr=E.SubElement(ppr,W+'rPr'); rf=E.SubElement(rpr,W+'rFonts'); rf.set(W+'ascii','Times New Roman'); rf.set(W+'hAnsi','Times New Roman'); rf.set(W+'eastAsia','宋体')
    sz=E.SubElement(rpr,W+'sz'); sz.set(W+'val','24' if level==1 else '22'); szc=E.SubElement(rpr,W+'szCs'); szc.set(W+'val','24' if level==1 else '22')
    if level in (1,2): E.SubElement(rpr,W+'b'); E.SubElement(rpr,W+'bCs')
    # Normalize cached runs using the same safe run-property structure as the
    # official template; retaining field markers and hyperlink contents.
    for r in p.xpath('.//w:r', namespaces=N):
        ensure_rpr(r, size=(24 if level==1 else 22), bold=None)
    oldppr=p.find(W+'pPr')
    if oldppr is not None: p.remove(oldppr)
    p.insert(0,ppr)


def remap_sect_refs(sect, mapping):
    for ref in sect.xpath('.//*[@r:id]', namespaces=N):
        rid=ref.get(R+'id')
        if rid in mapping: ref.set(R+'id',mapping[rid])


def main():
    if not SRC.exists() or not TEMPLATE.exists(): raise SystemExit('source/template missing')
    with ZipFile(SRC) as z: src={n:z.read(n) for n in z.namelist()}
    with ZipFile(TEMPLATE) as z: tmp={n:z.read(n) for n in z.namelist()}
    sr=E.fromstring(src['word/document.xml']); sb=sr.find('w:body',N)
    tr=E.fromstring(tmp['word/document.xml']); tb=tr.find('w:body',N)
    se=list(sb); te=list(tb)
    # v7 body starts at the first paragraph whose exact text is 问题重述.
    body_i=next(i for i,x in enumerate(se) if x.tag==W+'p' and text(x)=='问题重述')
    old_body = [deepcopy(x) for x in se[body_i:]]
    # official front skeleton elements
    template_table = next(x for x in te if x.tag==W+'tbl')
    front_title = next(x for x in te if x.tag==W+'p' and text(x)=='论文题目')
    front_abs_head = next(x for x in te if x.tag==W+'p' and text(x)=='摘 要')
    front_abs_body = next(x for x in te if x.tag==W+'p' and text(x)=='（摘要正文）')
    front_keywords = next(x for x in te if x.tag==W+'p' and x.find('w:pPr/w:sectPr',N) is not None)
    toc_head = next(x for x in te if x.tag==W+'p' and text(x)=='目录')
    toc_templates = {1: next(x for x in te if x.tag==W+'p' and text(x).startswith('键入章级(第 1 级)')),
                     2: next(x for x in te if x.tag==W+'p' and text(x).startswith('键入章级(第 2 级)')),
                     3: next(x for x in te if x.tag==W+'p' and text(x).startswith('键入章标题(第 3 级)'))}
    toc_sect = next(x for x in te if x.tag==W+'p' and x.find('w:pPr/w:sectPr',N) is not None and text(x)=='')
    # There are two blank section paragraphs in the template; first is front (keywords) and second is TOC.
    template_sects=[x.find('w:pPr/w:sectPr',N) for x in te if x.tag==W+'p' and x.find('w:pPr/w:sectPr',N) is not None]
    front_sect=template_sects[0]; toc_sect=template_sects[1]
    # Set table values while retaining the official table geometry/borders.
    tbl=deepcopy(template_table)
    cells=tbl.xpath('.//w:tc',namespaces=N)
    # first row right cell / second row right cell
    for idx,val in [(1,'MCB2601605'),(3,'A赛道')]:
        cell=cells[idx]
        for t in cell.xpath('.//w:t',namespaces=N): t.getparent().remove(t)
        # preserve first paragraph and rPr from existing first run
        p=cell.find('w:p',N)
        for child in list(p):
            if child.tag != W+'pPr': p.remove(child)
        r=E.SubElement(p,W+'r'); rpr=E.SubElement(r,W+'rPr'); rf=E.SubElement(rpr,W+'rFonts'); rf.set(W+'hint','eastAsia'); E.SubElement(r,W+'t').text=val
    # Approved v7 front text.
    title = next(x for x in se if x.tag==W+'p' and text(x)=='基于收缩集成与条件Copula的啤酒感官预测及风格生成')
    abstract_ps = [next(x for x in se if x.tag==W+'p' and text(x)==text(se[j])) for j in range(2,7)]
    keywords = next(x for x in se if x.tag==W+'p' and text(x).startswith('关键词：'))
    p_title=deepcopy(front_title); set_text_single(p_title,text(title),p_title.find('w:r',N))
    p_abshead=deepcopy(front_abs_head)
    # Official abstract heading is exactly as the template; preserve its space.
    p_absbody=[]
    for ap in abstract_ps:
        q=deepcopy(ap); format_front_body(q, front_abs_body.find('w:pPr',N), size=24); p_absbody.append(q)
    p_kw=deepcopy(front_keywords)
    # Replace placeholder runs with v7 keyword runs and retain official section properties.
    pkw_ppr=p_kw.find('w:pPr',N)
    # The copied template section uses rId3 for its footer; in v7 rId3 is footnotes.
    # Remap it to the existing empty footer2 relationship (rId9).
    if pkw_ppr is not None and pkw_ppr.find('w:sectPr',N) is not None:
        remap_sect_refs(pkw_ppr.find('w:sectPr',N), {'rId3':'rId9'})
    old_kw_runs=[deepcopy(r) for r in keywords if r.tag==W+'r']
    for child in list(p_kw):
        if child.tag != W+'pPr': p_kw.remove(child)
    for r in old_kw_runs:
        ensure_rpr(r,size=24)
        p_kw.append(r)
    # Front section relation IDs: official rId3 footer1 -> existing empty v7 footer2 rId9.
    front_sect_cp=deepcopy(front_sect); remap_sect_refs(front_sect_cp, {'rId3':'rId9'})
    # TOC head and cache from v7, with official paragraph layout. Use all 27 cached lines.
    p_toc=deepcopy(toc_head)
    toc_cache=[deepcopy(x) for x in se[next(i for i,x in enumerate(se) if x.tag==W+'p' and text(x)=='目 录')+1:body_i] if x.tag==W+'p' and text(x)]
    # The source has an empty section paragraph immediately before body; exclude it.
    p_toc_cache=[]
    for p in toc_cache:
        s=text(p)
        level=1 if (re.match(r'^[一二三四五六七八九十]+、',s) or s.startswith('参考文献')) else 2
        format_toc_p(p,level,toc_templates[level])
        p_toc_cache.append(p)
    # The official template has a static directory on page 2.  Preserve the
    # cached hyperlinks/page numbers from v7, but remove the top-level TOC
    # field instruction so a static Word export never prints the field code.
    if p_toc_cache:
        first = p_toc_cache[0]
        hyperlink = first.find('w:hyperlink', N)
        if hyperlink is not None:
            for child in list(first):
                if child.tag == W+'pPr':
                    continue
                if child is hyperlink:
                    break
                first.remove(child)
    toc_sect_cp=deepcopy(toc_sect); remap_sect_refs(toc_sect_cp, {'rId4':'rId6'})
    # Use template title/front/TOC sections, then unchanged v7 body including final section properties.
    new_body=E.Element(W+'body')
    new_body.append(tbl)
    new_body.append(p_title)
    new_body.append(p_abshead)
    new_body.extend(p_absbody)
    p_kw.get_or_add if False else None
    new_body.append(p_kw)
    new_body.append(p_toc)
    new_body.extend(p_toc_cache)
    # explicit TOC section break paragraph using template's second section settings
    secp=E.Element(W+'p'); secppr=E.SubElement(secp,W+'pPr'); secppr.append(toc_sect_cp)
    new_body.append(secp)
    # unchanged body starts with 问题重述 and ends with v7 final sectPr
    new_body.extend(old_body)
    sr.remove(sb); sr.append(new_body)
    # Keep the official CJK font explicit on all paragraph runs after the
    # section-layout transplant so Word resolves the mixed front/body styles.
    for rf in sr.xpath('.//w:body//w:r/w:rPr/w:rFonts', namespaces=N):
        rf.set(W+'eastAsia', '宋体')
    src['word/document.xml']=E.tostring(sr,encoding='UTF-8',xml_declaration=True,standalone=True)
    # Keep v7 styles/media/numbering; all template front layout is direct formatting.
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
        for name,data in src.items(): z.writestr(name,data)
    print(OUT)

if __name__=='__main__': main()
