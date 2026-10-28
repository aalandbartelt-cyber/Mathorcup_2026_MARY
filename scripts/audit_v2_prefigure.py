"""Audit the pre-figure working document without marking it a final submission."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import re
from lxml import etree as E
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'docs/collab/20261027_总论文v2准备'
DOC=ROOT/'05_论文/总论文/v2_准备稿/总论文_v2_图前工作稿.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W='{'+NS['w']+'}'


def read(path):
    with ZipFile(path) as z:return {n:z.read(n) for n in z.namelist()}


def visible(node):
    return ''.join(node.xpath('.//w:t/text() | .//m:t/text()',namespaces=NS))


def rows(table):return [[visible(c) for c in row.findall(W+'tc')] for row in table.findall(W+'tr')]


def main():
    current=read(DOC);root=E.fromstring(current['word/document.xml']);body=root.find(W+'body')
    baseline=ROOT/'05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v1.docx';old=read(baseline);oldroot=E.fromstring(old['word/document.xml'])
    pdf=pymupdf.open(DOC.with_suffix('.pdf'))
    pag=json.loads((AUDIT/'Word渲染/word_pagination.json').read_text(encoding='utf-8-sig'))
    snapshot=json.loads((AUDIT/'v1_保护快照.json').read_text())
    unchanged={name:sha256((ROOT/'05_论文/总论文'/name).read_bytes()).hexdigest()==record['sha256'] for name,record in snapshot.items()}
    ts=body.findall(W+'tbl');display=[t for t in ts if re.fullmatch(r'\(\d+\)',visible(t.findall('w:tr/w:tc',NS)[-1]))]
    content=[t for t in ts if t not in display];symbols=content[0]
    empties=root.xpath('//m:r[not(m:t) or not(normalize-space(m:t))]',namespaces=NS)
    native_rows=[len(row.findall('w:tc',NS)[0].xpath('.//m:oMath',namespaces=NS)) for row in symbols.findall(W+'tr')[1:]]
    notes=[]
    for t in display:
        after=t.getnext();notes.append(after is not None and after.tag==W+'p' and visible(after).startswith('式中，'))
    def digits(value):return re.findall(r'[-−+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',value.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉²¹','012345678921')))
    oldcontent=[t for t in oldroot.findall('w:body/w:tbl',NS) if not t.xpath('.//m:oMath',namespaces=NS)]
    numbers=[]
    for i,(before,after) in enumerate(zip(oldcontent[1:],content[1:]),2):
        # CO₂ and R² are superscripts in both representations, but spectral
        # tolerances become full signed exponent tokens in native mathematics.
        before_numbers=digits(visible(before).replace('10⁻⁶','10−6').replace('10⁻¹²','10−12').replace('10⁻¹⁵','10−15'))
        after_numbers=digits(visible(after))
        # Changing label 0/0.01/0.02z to explicit units adds no numerical fact.
        numbers.append({'table':i,'matches':before_numbers==after_numbers,'before':before_numbers,'after':after_numbers})
    text='\n'.join(root.xpath('//w:t/text()',namespaces=NS))
    disallowed=r'微网|光储|购电|储能|电价|马尔可夫场景|日前调度|滚动优化|小区负荷|光伏|充放电|\bSOC\b|问题四|\bE\d{3}\b|BZD|\bagent\d*\b|学长版|参考组|对照组|晋级|门槛'
    residual=[x for x in re.findall(r'[\u0370-\u03ff\u2070-\u209f]|\^|\[\[[^]]+\]\]',text)]
    toc=[s for s in pag['toc'].split('\r') if s.strip()]
    main_heads=[h for h in pag['headings'] if h['physical_page']<=next(i+1 for i,p in enumerate(pdf) if 'AI 工具使用声明' in p.get_text())]
    # Heading count stops at the reference heading; appendix is excluded by the
    # real TOC bookmark irrespective of its level.
    main_heads=main_heads[:next(i+1 for i,h in enumerate(main_heads) if h['text']=='参考文献')]
    toc_ok=len(toc)==len(main_heads)==27 and all(int(row.rsplit('\t',1)[1])==h['body_page'] for row,h in zip(toc,main_heads))
    refpages={label:[i+1 for i,p in enumerate(pdf) if label in p.get_text()] for label in ['[12]','[13]','AI 工具使用声明']}
    allmath=root.xpath('//m:oMath',namespaces=NS)
    def image_bytes(f,r):
        rel={x.get('Id'):x.get('Target') for x in E.fromstring(f['word/_rels/document.xml.rels'])}
        return [f['word/'+rel[x.get('{'+NS['r']+'}embed')]] for x in r.xpath('//w:drawing//a:blip',namespaces=NS)]
    pictures=image_bytes(current,root);oldpictures=image_bytes(old,oldroot)
    protected=['word/styles.xml','word/numbering.xml','word/theme/theme1.xml','word/fontTable.xml','word/settings.xml']
    pages_before_appendix=next(h['body_page'] for h in pag['headings'] if h['text']=='附录')-1
    equation_page_errors=[]
    for index,t in enumerate(display):
        index_in_all=ts.index(t)
        if pag['tables'][index_in_all]['start_page']!=pag['tables'][index_in_all]['end_page']:equation_page_errors.append(index+1)
    figcaps=[c for c in pag['captions'] if c['text'].startswith('图')]
    main_image_checks=[im['page']==cap['page'] for im,cap in zip(pag['images'][:14],figcaps)]
    supplementary_checks=[]
    for index in range(1,4):
        caption_pages=[i+1 for i,page in enumerate(pdf) if any(line.strip().startswith(f'附图S{index}') for line in page.get_text().splitlines())]
        supplementary_checks.append(caption_pages==[pag['images'][13+index]['page']])
    plain_font=set()
    math_font=set()
    for p in pdf:
        for b in p.get_text('dict')['blocks']:
            for line in b.get('lines',[]):
                for span in line['spans']:
                    if 'CambriaMath' in span['font'] or 'Cambria Math' in span['font']:math_font.add(span['font'])
    gates={
      'v1_files_unchanged':all(unchanged.values()),'template_parts_unchanged':all(current[n]==old[n] for n in protected),
      'display_equations23':len(display)==23,'equation_numbers_continuous':[visible(t.findall('w:tr/w:tc',NS)[-1]) for t in display]==[f'({i})' for i in range(1,24)],
      'equation_tables_1x3':all(len(t.findall(W+'tr'))==1 and len(t.findall('w:tr/w:tc',NS))==3 for t in display),
      'empty_math_runs0':not empties,'symbol_cells_native':len(native_rows)==10 and all(n==1 for n in native_rows),
      'each_display_followed_by_explanation':all(notes),'no_plain_unicode_math_or_markers':not residual,
      'no_forbidden_terms':not re.search(disallowed,text,re.I),'results_tables_numbers_unchanged':all(t['matches'] for t in numbers),
      'old_figures_bytes_unchanged':pictures[:14]==oldpictures,'no_unapproved_data_plots':len(pictures)==17,
      'references12_13_literal':all(value in text for value in ['[12] ChatGPT，GPT-5.6 Sol，OpenAI，2026-10-27','[13] DeepSeek，DeepSeek-V4.1-Flash，深度求索（DeepSeek），2026-10-27']),
      'disclosure_honesty_pending_marked':'待按真实使用记录核定声明' in text,
      'toc_updated_and_matches':toc_ok,'table_splits0':all(t['start_page']==t['end_page'] for t in pag['tables']),
      'equations_not_split':not equation_page_errors,'main_image_captions_same_page':all(main_image_checks),
      'body_under30_before_data_insert':pages_before_appendix<=30,'pdf_cambria_math_visible':bool(math_font),
      'supplement_captions_same_page':all(supplementary_checks),
      'cover_unchanged':pdf[0].get_text().strip()==pymupdf.open(baseline.with_suffix('.pdf'))[0].get_text().strip(),
    }
    result={'gates':gates,'ready_for_user_figure_review':all(gates.values()),'not_final_v2':True,
            'body_pages':pages_before_appendix,'physical_pages':len(pdf),'omml_count':len(allmath),'empty_math_runs':len(empties),
            'symbol_rows_native':native_rows,'first_explanation_paragraphs':sum(notes),'math_pdf_fonts':sorted(math_font),
            'ai_candidate_not_adopted':True,'ai_reference_physical_pages':refpages,'v1_unchanged':unchanged,
            'result_tables':numbers,'residual_math':residual,'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p.read_bytes()).hexdigest()} for p in [DOC,DOC.with_suffix('.pdf')]}}
    (AUDIT/'图前工作稿验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ['physical_pages','body_pages','omml_count','empty_math_runs','math_pdf_fonts','ai_reference_physical_pages']},ensure_ascii=False,indent=2))
    print(json.dumps(gates,ensure_ascii=False,indent=2))
    if not result['ready_for_user_figure_review']:raise SystemExit('See failed gates in 图前工作稿验收.json')


if __name__=='__main__':main()
