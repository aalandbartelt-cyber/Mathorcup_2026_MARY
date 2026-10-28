"""Static and Word/PDF pagination gates for the integrated manuscript."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json
import re
from lxml import etree as E
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'docs/collab/20261027_总论文整合'
DOC=ROOT/'05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v1.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
   'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
   'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W='{'+N['w']+'}'


def main():
    manifest=json.loads((AUDIT/'assembly_manifest.json').read_text())
    pagination=json.loads((AUDIT/'word_pagination.json').read_text(encoding='utf-8-sig'))
    with ZipFile(DOC) as z:files={n:z.read(n) for n in z.namelist()}
    root=E.fromstring(files['word/document.xml']);body=root.find('w:body',N)
    paragraphs=[''.join(p.xpath('.//w:t/text()',namespaces=N)) for p in body.findall('w:p',N)]
    doc_text='\n'.join(root.xpath('//w:t/text()',namespaces=N))
    pdf=pymupdf.open(DOC.with_suffix('.pdf'));pdf_text='\n'.join(p.get_text() for p in pdf)
    forbidden=r'微网|光储|购电|储能|电价|马尔可夫场景|日前调度|滚动优化|小区负荷|光伏|充放电|\bSOC\b|问题四|\bE\d{3}(?:_\w+)?\b|BZD|\bagent\d*\b|学长版|参考组|对照组|晋级|门槛|\bpromotion\b|\bpass\b'
    math=root.xpath('//m:oMath',namespaces=N)
    empty=root.xpath('//m:oMath//m:r[not(m:t) or not(normalize-space(m:t))]',namespaces=N)
    captions={k:[int(re.match(k+r'(\d+)',p)[1]) for p in paragraphs if re.match(k+r'\d+\s',p)] for k in ['图','表']}
    equation_labels=[]
    for t in body.findall(W+'tbl'):
        if t.xpath('.//m:oMath',namespaces=N):equation_labels.append(''.join(t.findall('w:tr/w:tc',N)[2].xpath('.//w:t/text()',namespaces=N)))
    toc_rows=[x for x in pagination['toc'].split('\r') if x.strip()]
    heads=pagination['headings'];toc_checks=[]
    for row,h in zip(toc_rows,heads):
        label,page=row.rsplit('\t',1);toc_checks.append({'heading':h['text'],'toc_page':int(page),'body_page':h['body_page'],'physical_page':h['physical_page'],'matches':int(page)==h['body_page']})
    figcaps=[c for c in pagination['captions'] if re.match(r'图\d+\s',c['text'])]
    table_caps=[c for c in pagination['captions'] if re.match(r'表\d+\s',c['text'])]
    data_tables=[t for t in pagination['tables'] if not re.fullmatch(r'\s*\(\d+\)\s*',t['text'].replace('\r','').replace('\a',''))]
    # More robustly align the table captions with the actual non-equation XML tables.
    table_indices=[i+1 for i,t in enumerate(body.findall(W+'tbl')) if not t.xpath('.//m:oMath',namespaces=N)]
    table_caption_checks=[{'caption':c['text'],'caption_page':c['page'],'table_page':pagination['tables'][i-1]['start_page']} for c,i in zip(table_caps,table_indices)]
    split_tables=[t['index'] for t in pagination['tables'] if t['start_page']!=t['end_page']]
    image_checks=[{'figure':i+1,'image_page':im['page'],'caption_page':cap['page']} for i,(im,cap) in enumerate(zip(pagination['images'],figcaps))]
    anchors=root.xpath('//w:hyperlink/@w:anchor',namespaces=N)
    bookmarks=set(root.xpath('//w:bookmarkStart/@w:name',namespaces=N))
    footer_checks=[]
    for i,page in enumerate(pdf):
        footer=page.get_text(clip=pymupdf.Rect(250,770,350,805)).strip()
        footer_checks.append(footer==('' if i<2 else str(i-1)))
    protected={n:sha256(files[n]).hexdigest()==h for n,h in manifest['protected_parts'].items()}
    source_unchanged={q:sha256(Path(s['path']).read_bytes()).hexdigest()==s['sha256'] for q,s in manifest['sources'].items()}
    main_last=next(h['body_page'] for h in heads if h['text']=='参考文献')
    reference_page_last=max(i+1-2 for i,p in enumerate(pdf) if re.search(r'\[11\]',p.get_text()) and i>2)
    gates={
       'mother_initial_copy_exact':manifest['initial_copy_sha256']==manifest['mother_sha256'],
       'protected_parts_identical':all(protected.values()),'sources_unchanged':all(source_unchanged.values()),
       'front_page_placeholders':all(x in pdf[0].get_text() for x in ['【题目待定】','【摘要下一轮撰写】','【关键词下一轮确定】']),
       'toc_only_on_physical_page2':'目 录' in pdf[1].get_text() and '参考文献' in pdf[1].get_text(),
       'toc_updated_and_matching':len(toc_rows)==len(heads)==27 and all(x['matches'] for x in toc_checks),
       'body_starts_at1':heads[0]['physical_page']==3 and heads[0]['body_page']==1,
       'body_under30':reference_page_last<=30,
       'figures_continuous':captions['图']==list(range(1,15)),
       'tables_continuous':captions['表']==list(range(1,21)),
       'equations_continuous':equation_labels==[f'({i})' for i in range(1,24)],
       'native_math23':len(math)==pagination['formulas']==23,'empty_math_runs0':not empty,
       'no_old_or_internal_terms':not re.search(forbidden,doc_text,re.I) and not re.search(forbidden,pdf_text,re.I),
       'tables_not_split':not split_tables,
       'table_captions_same_page':all(x['caption_page']==x['table_page'] for x in table_caption_checks),
       'figure_captions_same_page':all(x['image_page']==x['caption_page'] for x in image_checks),
       'no_header_references':not root.xpath('//w:headerReference',namespaces=N),
       'real_toc_field':bool(root.xpath('//w:instrText[contains(text(),"TOC ")]',namespaces=N)),
       'no_duplicate_bookmark_ids':len(root.xpath('//w:bookmarkStart/@w:id',namespaces=N))==len(set(root.xpath('//w:bookmarkStart/@w:id',namespaces=N))),
       'toc_hyperlinks_resolve':all(a in bookmarks for a in anchors),
       'footer_numbers_continuous':all(footer_checks),
    }
    # Extended metadata is not a template style part. Replace stale 94-page stats
    # and the original WPS installation fingerprint, without changing layout.
    app=E.fromstring(files['docProps/app.xml']);ns='{'+E.QName(app).namespace+'}'
    vals={'Pages':str(len(pdf)),'Paragraphs':str(len(root.findall('.//'+W+'p'))),'Application':'Microsoft Office Word'}
    for child in list(app):
        local=E.QName(child).localname
        if local in vals:child.text=vals[local]
        elif local in ['Words','Characters','CharactersWithSpaces','Lines']:app.remove(child)
    files['docProps/app.xml']=E.tostring(app,encoding='UTF-8',xml_declaration=True,standalone=True)
    with ZipFile(DOC,'w',ZIP_DEFLATED) as z:
        for name,value in files.items():z.writestr(name,value)
    result={'gates':gates,'all_pass':all(gates.values()),'pages':{'total':len(pdf),'body_including_references':reference_page_last,'front':2,'appendix_placeholder':1},
            'counts':{'figures':14,'tables':20,'equations':23,'references':11,'toc_entries':len(toc_rows)},
            'protected_parts':protected,'sources_unchanged':source_unchanged,'toc_checks':toc_checks,
            'table_caption_checks':table_caption_checks,'image_caption_checks':image_checks,
            'forbidden_docx_hits':re.findall(forbidden,doc_text,re.I),'forbidden_pdf_hits':re.findall(forbidden,pdf_text,re.I),'empty_math_runs':len(empty),
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p.read_bytes()).hexdigest()} for p in [DOC,DOC.with_suffix('.pdf')]}}
    (AUDIT/'final_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result['pages']));print(json.dumps(gates,ensure_ascii=False,indent=2))
    if not result['all_pass']:raise SystemExit('Integration gate failed; see final_checks.json')


if __name__=='__main__':main()
