"""Use installed Word to update a real TOC, transplant only its cache, and export.

Word never saves or replaces the mother stylesheet and numbering parts.
"""
from copy import deepcopy as cp
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import argparse
import json
import shutil
import subprocess
from lxml import etree as E

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'docs/collab/20261027_总论文整合'
FILE=ROOT/'05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v1.docx'
WORK=Path('/mnt/c/Users/Public/mathorcup_total_20261027')
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
   'pkg':'http://schemas.microsoft.com/office/2006/xmlPackage'}
W='{'+N['w']+'}'


def main():
    global FILE,AUDIT,WORK
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=FILE)
    parser.add_argument('--audit',type=Path,default=AUDIT)
    parser.add_argument('--work',type=Path,default=WORK)
    args=parser.parse_args();FILE=args.input.resolve();AUDIT=args.audit.resolve();WORK=args.work.resolve()
    AUDIT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    script=WORK/'export.ps1';script.write_bytes(b'\xef\xbb\xbf'+(ROOT/'scripts/export_total_paper_word.ps1').read_bytes())
    local=WORK/FILE.name;shutil.copy2(FILE,local)
    prefix=subprocess.check_output(['wslpath','-w',str(WORK)],text=True).strip()
    cmd=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',prefix+'\\export.ps1',
         '-Source',prefix+'\\'+FILE.name,'-OutputDir',prefix]
    subprocess.run(cmd+['-UpdateToc'],check=True)
    with ZipFile(FILE) as z:files={n:z.read(n) for n in z.namelist()}
    root=E.fromstring(files['word/document.xml']);body=root.find('w:body',N)
    frag=E.parse(str(WORK/'toc_fragment.xml'))
    fragment=frag.find('.//w:body',N)
    paras=[cp(p) for p in fragment if p.tag==W+'p']
    # Word's Flat OPC fragment may use renamed style ids. Use only the mother's Normal.
    for p in paras:
        pp=p.find(W+'pPr')
        if pp is None:pp=E.Element(W+'pPr');p.insert(0,pp)
        style=pp.find(W+'pStyle')
        if style is None:style=E.Element(W+'pStyle');pp.insert(0,style)
        style.set(W+'val','1')
    old=next(p for p in body.findall(W+'p') if p.xpath('.//w:instrText[contains(text(),"TOC ")]',namespaces=N))
    index=body.index(old)
    # A previously updated TOC spans several paragraphs; replace its whole cache.
    stop=next(i for i in range(index+1,len(body)) if body[i].find('w:pPr/w:sectPr',N) is not None)
    for node in list(body)[index:stop]:body.remove(node)
    for p in reversed(paras):body.insert(index,p)
    # Keep the generated TOC hyperlinks usable without importing any source styles.
    record=json.loads((WORK/'word_pagination.json').read_text(encoding='utf-8-sig'))
    headings=[p for p in body.findall(W+'p') if p.find('w:pPr/w:pStyle',N) is not None and p.find('w:pPr/w:pStyle',N).get(W+'val') in ['2','3','4']]
    old_ids={x.get(W+'id') for x in root.findall('.//'+W+'bookmarkStart') if x.get(W+'name','').startswith('_Toc')}
    for x in list(root.findall('.//'+W+'bookmarkStart'))+list(root.findall('.//'+W+'bookmarkEnd')):
        if x.get(W+'id') in old_ids:x.getparent().remove(x)
    for p,h in zip(headings,record['headings']):
        hx=E.fromstring(h['xml'].encode());names=hx.xpath('.//w:bookmarkStart[starts-with(@w:name,"_Toc")]/@w:name',namespaces=N)
        for name in names:
            ident=str(11000+headings.index(p));start=E.Element(W+'bookmarkStart');start.set(W+'id',ident);start.set(W+'name',name)
            end=E.Element(W+'bookmarkEnd');end.set(W+'id',ident);p.insert(1,start);p.append(end)
    files['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
    with ZipFile(FILE,'w',ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    shutil.copy2(WORK/'toc_fragment.xml',AUDIT/'toc_fragment.xml')
    shutil.copy2(WORK/'word_pagination.json',AUDIT/'word_pagination_updated.json')
    # Re-open the actual delivered DOCX without changing it; export and audit that file.
    shutil.copy2(FILE,local);subprocess.run(cmd,check=True)
    shutil.copy2(local.with_suffix('.pdf'),FILE.with_suffix('.pdf'))
    shutil.copy2(WORK/'word_pagination.json',AUDIT/'word_pagination.json')
    print(FILE.with_suffix('.pdf'))


if __name__=='__main__':main()
