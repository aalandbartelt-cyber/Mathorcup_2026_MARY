"""Verify the scoped v3 polish against immutable v2, plus native Word PDF."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import re
from io import BytesIO

from lxml import etree as E
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文'
AUDIT = ROOT / 'docs/collab/20261027_总论文v3修订'
DOC = BASE / '2026_MathorCup_大数据竞赛_总论文_v3.docx'
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
     'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
     'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W = '{' + N['w'] + '}'


def read(path):
    with ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def txt(x):
    return ''.join(x.xpath('.//w:t/text() | .//m:t/text()', namespaces=N))


def main():
    files = read(DOC); oldfiles = read(BASE / '2026_MathorCup_大数据竞赛_总论文_v2.docx')
    root = E.fromstring(files['word/document.xml']); oldroot = E.fromstring(oldfiles['word/document.xml'])
    body = root.find(W + 'body'); oldbody = oldroot.find(W + 'body')
    def normalize(t):
        return re.sub('[—–]+', '-', txt(t)).replace('本研究', '本文')
    def equation(t):
        cells = t.findall('w:tr/w:tc', N)
        return bool(cells and re.fullmatch(r'\(\d+\)', txt(cells[-1])))
    tables = body.findall(W + 'tbl'); oldtables = oldbody.findall(W + 'tbl')
    eq = [t for t in tables if equation(t)]; oldeq = [t for t in oldtables if equation(t)]
    content = [t for t in tables if not equation(t)]; oldcontent = [t for t in oldtables if not equation(t)]
    table12 = next(x.getnext() for x in body if txt(x).startswith('表12 '))
    oldtable12 = next(x.getnext() for x in oldbody if txt(x).startswith('表12 '))
    table_checks = [normalize(a) == normalize(b) for a, b in zip(oldcontent, content) if b is not table12]
    signature = lambda x: E.tostring(x, method='c14n', exclusive=True)
    eq_checks = [signature(a.find('.//m:oMath', N)) == signature(b.find('.//m:oMath', N)) for a, b in zip(oldeq, eq)]
    ps = body.findall(W + 'p')
    word = '\n'.join(root.xpath('//w:t/text()', namespaces=N))
    alltext = txt(root)
    p = json.loads((AUDIT / 'Word渲染/word_pagination.json').read_text(encoding='utf-8-sig'))
    pdf = pymupdf.open(DOC.with_suffix('.pdf'))
    pdftext = '\n'.join(x.get_text() for x in pdf)
    refs = [txt(x) for x in ps if re.match(r'^\[\d+\]\s', txt(x))]
    cited = {int(n) for x in ps if not re.match(r'^\[\d+\]\s', txt(x)) for n in re.findall(r'\[(\d+)\]', txt(x))}
    notes = [x for x in ps if txt(x).startswith('式中')]
    proofs = [x for x in ps if txt(x).startswith('证明。')]
    actual_heading = lambda x: x.find('w:pPr/w:pStyle', N) is not None and x.find('w:pPr/w:pStyle', N).get(W + 'val') in ['2', '3', '4']
    headings = [x for x in ps if actual_heading(x)]
    analysis = [txt(x) for x in headings if re.match(r'^2\.[123] ', txt(x))]
    counts = {}
    for label in ['8.1 ', '8.2 ']:
        h = next(x for x in headings if txt(x).startswith(label)); n = 0
        x = h.getnext()
        while x is not None and x.tag == W + 'p' and not actual_heading(x):
            if re.match(r'^（\d+）', txt(x)):
                n += 1
            x = x.getnext()
        counts[label] = n
    toc = [x for x in p['toc'].split('\r') if x.strip()]
    main_h = p['headings'][:next(i + 1 for i, h in enumerate(p['headings']) if h['text'] == '参考文献')]
    toc_ok = len(toc) == len(main_h) == 27 and all(int(x.rsplit('\t', 1)[1]) == h['body_page'] for x, h in zip(toc, main_h))
    snapshot = json.loads((AUDIT / '旧版本保护快照.json').read_text())
    protected = {n: sha256((ROOT / n).read_bytes()).hexdigest() == v['sha256'] for n, v in snapshot.items()}
    pdf_fonts = {span['font'] for page in pdf for block in page.get_text('dict')['blocks'] for line in block.get('lines', []) for span in line['spans']}
    math_fonts = sorted(f for f in pdf_fonts if 'CambriaMath' in f)
    body_pages = next(h['body_page'] for h in p['headings'] if h['text'] == '附录') - 1
    captions = [c for c in p['captions'] if c['text'].startswith('图')]
    empty = root.xpath('//m:r[not(m:t) or not(normalize-space(m:t))]', namespaces=N)
    forbidden = r'微网|光储|购电|储能|电价|\bE\d{3}\b|BZD|\bagent\d*\b|学长版|参考组|对照组|晋级|门槛'
    cell_fonts = table12.xpath('.//w:rPr/w:rFonts', namespaces=N)
    exact_ai = json.loads((ROOT / 'docs/collab/20261027_总论文v2定稿/assembly_manifest.json').read_text())
    edits = json.loads((AUDIT / '修改清单.json').read_text())['figure_character_edits']
    media_edits = {x['media'] for x in edits.values()}
    changed = {n for n in files if files[n] != oldfiles[n]}
    plot_checks = (BASE / '图表_v3_字符统一/数值图层核验.csv').read_text()
    gates = {
        'old_versions_hash_unchanged': all(protected.values()),
        'only_authorized_xml_and5_label_images_changed': set(files) == set(oldfiles) and changed == {'word/document.xml'} | media_edits,
        'other15_images_unchanged': all(files[n] == oldfiles[n] for n in files if n.startswith('word/media/') and n not in media_edits) and len(root.xpath('//w:drawing', namespaces=N)) == 20,
        'all_image_canvas_dimensions_unchanged': all(Image.open(BytesIO(files[n])).size == Image.open(BytesIO(oldfiles[n])).size for n in files if n.startswith('word/media/') and n.endswith('.png')),
        'five_plot_numeric_layers_unchanged': plot_checks.count('TRUE') == 5 and 'FALSE' not in plot_checks,
        'other19_content_tables_unchanged': len(table_checks) == 19 and all(table_checks),
        'display23_formulas_unchanged': len(eq) == len(oldeq) == 23 and all(eq_checks),
        'native_display_1row3cols': all(len(t.findall(W + 'tr')) == 1 and len(t.findall('w:tr/w:tc', N)) == 3 for t in eq),
        'formula_numbers1_to23': [txt(t.findall('w:tr/w:tc', N)[-1]) for t in eq] == [f'({i})' for i in range(1, 24)],
        'omml_empty_run0': not empty,
        'core_notation_native10': len(content[0].findall(W + 'tr')) == 11 and all(len(row.findall('w:tc', N)[0].xpath('.//m:oMath', namespaces=N)) == 1 for row in content[0].findall(W + 'tr')[1:]),
        'text_emdash_endash0': not re.search('[—–]', alltext),
        'pdf_text_emdash_endash0': not re.search('[—–]', pdftext),
        'voice_unified': '本研究' not in alltext and '本研究' not in pdftext,
        'analysis_headings_short': analysis == ['2.1 问题一分析', '2.2 问题二分析', '2.3 问题三分析'],
        '23_explanations_flush_left': len(notes) == 23 and all(x.find('w:pPr/w:ind', N).get(W + 'firstLine') == '0' and x.find('w:pPr/w:ind', N).get(W + 'firstLineChars') == '0' for x in notes),
        'proofs3_native_QED': len(proofs) == 3 and all(x.xpath('./m:oMath[last()]//m:t/text()', namespaces=N)[-1] == '□' for x in proofs) and pdftext.count('□') == 3,
        'solver_scipy_active_set': 'SciPy主动集NNLS' in word and 'NIPALS迭代' in word,
        'solver_coordinate_descent': '循环坐标下降' in word and 'GraphicalLasso的坐标下降模式' in word,
        'solver_finite_search': '七个路径比例作有限枚举' in word and '最优性仅限给定候选集' in word,
        'table12_compact_6rows3cols': len(table12.findall(W + 'tr')) == 6 and all(len(tr.findall(W + 'tc')) == 3 for tr in table12.findall(W + 'tr')),
        'table12_uniform_fonts10pt': len(cell_fonts) == 18 and all(f.get(W + 'ascii') == 'Times New Roman' and f.get(W + 'eastAsia') == '宋体' for f in cell_fonts) and all(x.get(W + 'val') == '20' for x in table12.xpath('.//w:rPr/w:sz', namespaces=N)),
        'table12_all5_method_rows_cited': all(re.search(r'\[\d+\]', txt(tr)) for tr in table12.findall(W + 'tr')[1:]),
        'RRF_source_scope_explicit': '四路稳健排序（倒数秩项[14]）' in txt(table12),
        'references1_to14': [int(re.match(r'^\[(\d+)\]', x)[1]) for x in refs] == list(range(1, 15)) and '10.1145/1571941.1572114' in refs[-1],
        'all_intext_references_resolve': cited <= set(range(1, 15)) and 14 in cited,
        'advantages3_limitations4_points': counts == {'8.1 ': 3, '8.2 ': 4},
        'ai_disclosure_unchanged': exact_ai['ai_paragraph_exact'] in word and exact_ai['ai_scope_supplement'] in word and refs[:2] == ['[1] ChatGPT，GPT-5.6 Sol，OpenAI，2026-10-27', '[2] DeepSeek，DeepSeek-V4.1-Flash，深度求索（DeepSeek），2026-10-27'],
        'technical_route_not_embedded': '技术路线图' not in alltext and len(root.xpath('//w:drawing', namespaces=N)) == 20,
        'toc27_pages_correct': toc_ok,
        'toc_on_second_page': '目 录' in pdf[1].get_text() and '参考文献' in pdf[1].get_text(),
        'toc_no_appendix_AI': all('附录' not in x and 'AI 工具' not in x for x in toc),
        'tables_not_split': all(t['start_page'] == t['end_page'] for t in p['tables']),
        'figure_captions_same_page': all(c['page'] == im['page'] for c, im in zip(captions, p['images'][:17])),
        'body_le30_pages': body_pages <= 30,
        'pdf_CambriaMath_present': bool(math_fonts),
        'no_internal_terms': not re.search(forbidden, alltext, re.I),
        'cover_placeholders_unchanged': all(x in pdf[0].get_text() for x in ['【题目待定】', '【摘要下一轮撰写】', '【关键词下一轮确定】']),
    }
    out = {'passed': all(gates.values()), 'checks': {k: bool(v) for k, v in gates.items()},
           'physical_pages': len(pdf), 'body_pages': body_pages, 'main_figures': 17,
           'supplement_figures': 3, 'content_tables': len(content), 'display_formulas': len(eq),
           'native_math_objects': len(root.xpath('//m:oMath', namespaces=N)),
           'empty_math_runs': len(empty), 'pdf_math_fonts': math_fonts, 'references': len(refs),
           'protected_files': protected, 'table12_physical_page': next(x['page'] for x in p['captions'] if x['text'].startswith('表12 ')),
           'QED_pdf_count': pdftext.count('□'), 'files': {str(x.relative_to(ROOT)): {'sha256': sha256(x.read_bytes()).hexdigest(), 'bytes': x.stat().st_size} for x in [DOC, DOC.with_suffix('.pdf')]}}
    (AUDIT / 'final_checks.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not out['passed']:
        raise SystemExit('v3 checks failed')


if __name__ == '__main__':
    main()
