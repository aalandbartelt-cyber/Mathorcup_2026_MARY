"""Release checks for the figure-approved integrated v2; no model computation."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import re
from lxml import etree as E
import pandas as pd
import numpy as np
import pymupdf

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'05_论文/总论文';AUDIT=ROOT/'docs/collab/20261027_总论文v2定稿'
DOC=BASE/'2026_MathorCup_大数据竞赛_总论文_v2.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
W='{'+N['w']+'}'


def read(p):
    with ZipFile(p) as z:return {n:z.read(n) for n in z.namelist()}


def visible(x):return ''.join(x.xpath('.//w:t/text() | .//m:t/text()',namespaces=N))


def images(files,root):
    rels={r.get('Id'):r.get('Target') for r in E.fromstring(files['word/_rels/document.xml.rels'])}
    return [files['word/'+rels[x.get('{'+N['r']+'}embed')]] for x in root.xpath('//w:drawing//a:blip',namespaces=N)]


def numbers(t):
    value=visible(t).translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁻','01234567890123456789−'))
    return re.findall(r'[-−+]?\d+(?:\.\d+)?',value)


def main():
    current=read(DOC);root=E.fromstring(current['word/document.xml']);body=root.find(W+'body')
    baseline=read(BASE/'2026_MathorCup_大数据竞赛_总论文_v1.docx');oldroot=E.fromstring(baseline['word/document.xml'])
    manifest=json.loads((AUDIT/'assembly_manifest.json').read_text())
    p=json.loads((AUDIT/'Word渲染/word_pagination.json').read_text(encoding='utf-8-sig'))
    pdf=pymupdf.open(DOC.with_suffix('.pdf'));text='\n'.join(root.xpath('//w:t/text()',namespaces=N))
    snapshot=json.loads((ROOT/'docs/collab/20261027_总论文v2准备/v1_保护快照.json').read_text())
    v1ok={name:sha256((BASE/name).read_bytes()).hexdigest()==r['sha256'] for name,r in snapshot.items()}
    tables=body.findall(W+'tbl');eq=[];content=[]
    for t in tables:
        cells=t.findall('w:tr/w:tc',N);label=visible(cells[-1])
        if re.fullmatch(r'\(\d+\)',label):eq.append((t,label))
        else:content.append(t)
    oldtables=[t for t in oldroot.findall('w:body/w:tbl',N) if not t.xpath('.//m:oMath',namespaces=N)]
    table_checks=[numbers(a)==numbers(b) for a,b in zip(oldtables[1:],content[1:])]
    pics=images(current,root);oldpics=images(baseline,oldroot)
    oldmap={int(k):int(v) for k,v in manifest['figure_map'].items()}
    image_checks={k:oldpics[k-1]==pics[v-1] for k,v in oldmap.items()}
    datafigchecks={n:pics[n-1]==(BASE/f'图稿_待审计/数据概览/Q{i}_数据概览_待审.png').read_bytes() for i,n in enumerate([1,7,13],1)}
    all_p=body.findall(W+'p');caps=[visible(x) for x in all_p if re.match(r'^图\d+\s',visible(x))]
    refs=[visible(x) for x in all_p if re.match(r'^\[\d+\]\s',visible(x))]
    ref_ids=[int(re.match(r'^\[(\d+)\]',x)[1]) for x in refs]
    empty=root.xpath('//m:r[not(m:t) or not(normalize-space(m:t))]',namespaces=N)
    starts=root.xpath('//w:bookmarkStart/@w:name',namespaces=N)
    hyperlinks=root.xpath('//w:hyperlink/@w:anchor',namespaces=N)
    symbol_rows=content[0].findall(W+'tr')[1:]
    norm=lambda s:re.sub(r'\s+','',s)
    toc=[row for row in p['toc'].split('\r') if row.strip()]
    main_heads=p['headings'][:next(i+1 for i,h in enumerate(p['headings']) if h['text']=='参考文献')]
    toc_ok=len(toc)==len(main_heads)==27 and all(int(row.rsplit('\t',1)[1])==h['body_page'] for row,h in zip(toc,main_heads))
    maincaps=[c for c in p['captions'] if c['text'].startswith('图')]
    picture_pages=[im['page'] for im in p['images']]
    mathfonts={s['font'] for page in pdf for b in page.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans'] if 'CambriaMath' in s['font']}
    ai_title=next(i+1 for i,page in enumerate(pdf) if 'AI 工具使用声明' in page.get_text())
    bodypages=next(h['body_page'] for h in p['headings'] if h['text']=='附录')-1
    forbidden=r'微网|光储|购电|储能|电价|马尔可夫场景|日前调度|滚动优化|小区负荷|光伏|充放电|\bSOC\b|问题四|\bE\d{3}\b|BZD|\bagent\d*\b|学长版|参考组|对照组|晋级|门槛'
    symbols_plain=re.findall(r'[\u0370-\u03ff\u2070-\u209f]|\^|\[\[[^]]+\]\]',text)
    q1src=pd.read_csv(ROOT/'05_论文/Q1_感官评分/数据与结果/final_oof_with_relative_grades.csv.gz')
    q1new=pd.read_csv(BASE/'数据与结果_v2/问题一_250啤酒50维相对评分.csv.gz')
    q2src=pd.read_csv(ROOT/'05_论文/Q2_关键化合物/数据与结果/关键化合物Top3_当前版.csv')
    q2new=pd.read_csv(BASE/'数据与结果_v2/问题二_50维Top3与证据.csv')
    q1same=all(np.allclose(q1src[c],q1new[c],atol=2e-15,rtol=0) if pd.api.types.is_numeric_dtype(q1src[c]) else q1src[c].equals(q1new[c]) for c in q1src)
    q2same=all(q2src[c].fillna('').equals(q2new[c].fillna('')) if not pd.api.types.is_numeric_dtype(q2src[c]) else np.allclose(q2src[c],q2new[c],equal_nan=True,atol=2e-15,rtol=0) for c in q2new)
    q3spectra=pd.read_csv(BASE/'数据与结果_v2/问题三_900条生成化学谱.csv.gz')
    # Appendix figure captions must inherit the exact same mother caption
    # paragraph and font weight instead of forcing regular Latin glyphs.
    appendix_caps=[x for x in all_p if re.match(r'^附图S[123]\s',visible(x))]
    appcapok=all(x.find('w:pPr/w:pStyle',N).get(W+'val')=='41' and not x.xpath('.//w:rPr/w:b[@w:val="0"]',namespaces=N) for x in appendix_caps)
    ref_head=next(x for x in all_p if visible(x)=='参考文献' and x.find('w:pPr/w:pStyle',N).get(W+'val')=='2')
    start=body.index(ref_head);end=next(i for i,x in enumerate(body) if visible(x)=='AI 工具使用声明')
    reference_scope=[visible(x) for x in list(body)[start+1:end] if re.match(r'^\[\d+\]\s',visible(x))]
    gates={
      'v1_and_notes_unchanged':all(v1ok.values()),
      'mother_style_parts_unchanged':all(current[n]==baseline[n] for n in ['word/styles.xml','word/numbering.xml','word/theme/theme1.xml','word/fontTable.xml','word/settings.xml']),
      'core_symbol_rows10':len(symbol_rows)==10 and all(len(x.findall('w:tc',N)[0].xpath('.//m:oMath',namespaces=N))==1 for x in symbol_rows),
      'equations1_to23':[label for _,label in eq]==[f'({i})' for i in range(1,24)],
      'equations_native_1x3':all(len(t.findall(W+'tr'))==1 and len(t.findall('w:tr/w:tc',N))==3 and len(t.xpath('.//m:oMath',namespaces=N))==1 for t,_ in eq),
      'omml_empty_run0':not empty,'plain_math_residual0':not symbols_plain,
      'figures1_to17':[int(re.match(r'图(\d+)',c)[1]) for c in caps]==list(range(1,18)),
      'original14_figures_unchanged':all(image_checks.values()),'approved_data_images_exact':all(datafigchecks.values()),
      'supplement_fonts_updated':len(pics)==20 and appcapok,
      'source19_tables_numbers_unchanged':len(table_checks)==19 and all(table_checks),
      'references_reordered1_to13':ref_ids==list(range(1,14)) and reference_scope==refs,
      'ai_references_first_two':refs[0]=='[1] ChatGPT，GPT-5.6 Sol，OpenAI，2026-10-27' and refs[1]=='[2] DeepSeek，DeepSeek-V4.1-Flash，深度求索（DeepSeek），2026-10-27',
      'requested_ai_paragraph_verbatim':manifest['ai_paragraph_exact'] in text,
      'actual_ai_assistance_disclosed':manifest['ai_scope_supplement'] in text,
      'propositions_and_proofs':all(f'命题{s}' in text for s in ['5-1','6-1','7-1']) and text.count('证明。')==3,
      'appendix_uses_figure_ids':'本图' not in text,
      'toc27_matches_pages':toc_ok and all('附录' not in row and 'AI 工具' not in row for row in toc),
      'toc_on_page2':'目 录' in pdf[1].get_text() and '参考文献' in pdf[1].get_text(),
      'no_header_refs':not root.xpath('//w:headerReference',namespaces=N),
      'toc_links_resolve':all(h in starts for h in hyperlinks),
      'no_table_split':all(t['start_page']==t['end_page'] for t in p['tables']),
      'main_images_same_page_as_caption':all(c['page']==im for c,im in zip(maincaps,picture_pages[:17])),
      'body_le30':bodypages<=30,'pdf_math_cambria':bool(mathfonts),
      'frozen_q1_values_unchanged':q1same,'q1_integer_display_valid':np.array_equal(q1new.relative_grade_integer,np.floor(q1src.grade_0_6+.5).astype(int)) and q1new.relative_grade_integer.between(0,6).all(),
      'q2_top3_and_evidence_unchanged':q2same and len(q2new)==150,
      'q3_generated_spectra_complete':q3spectra.shape==(900,233) and q3spectra.groupby('style').size().eq(300).all() and np.isfinite(q3spectra.drop(columns=['style','sample_id']).to_numpy()).all(),
      'forbidden_terms0':not re.search(forbidden,text,re.I),
      'cover_placeholders_kept':all(s in pdf[0].get_text() for s in ['【题目待定】','【摘要下一轮撰写】','【关键词下一轮确定】']),
    }
    gates={key:bool(value) for key,value in gates.items()}
    result={'gates':gates,'passed':all(gates.values()),'physical_pages':len(pdf),'body_pages':bodypages,
      'main_figures':17,'supplements':3,'content_tables':len(content),'display_formulas':len(eq),'omml_count':len(root.xpath('//m:oMath',namespaces=N)),
      'math_fonts':sorted(mathfonts),'empty_math_runs':len(empty),'propositions':3,'references':13,'ai_physical_page':ai_title,
      'v1_hash_checks':v1ok,'table_checks':table_checks,'old_figures':image_checks,'data_figures':datafigchecks,
      'files':{p.name:{'sha256':sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in [DOC,DOC.with_suffix('.pdf')]},
      'ai_usage_scope':'Requested paragraph preserved; evidenced substantive assistance disclosed separately. Version and human-review statements are supplied by team, not independently certified.'}
    (AUDIT/'final_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ['physical_pages','body_pages','main_figures','content_tables','display_formulas','omml_count','ai_physical_page']},ensure_ascii=False))
    print(json.dumps(gates,ensure_ascii=False,indent=2))
    if not result['passed']:raise SystemExit('v2 checks failed')


if __name__=='__main__':main()
