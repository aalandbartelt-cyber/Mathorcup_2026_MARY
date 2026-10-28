"""Audit the approved roadmap insertion against the complete frozen v3 copy."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import re

from lxml import etree as E
import pymupdf

from insert_total_v4_route import INTRO, CAPTION, FIGURE, N, W

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文'
AUDIT = ROOT / 'docs/collab/20261027_总论文v4嵌图'
DOC = BASE / '2026_MathorCup_大数据竞赛_总论文_v4.docx'


def read(path):
    with ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def text(x):
    return ''.join(x.xpath('.//w:t/text() | .//m:t/text()', namespaces=N))


def remap(value):
    return re.sub(r'(?<!附)图\s*(\d+)(?!\d)', lambda m: '图' + str(int(m[1]) + 1), value)


def main():
    files = read(DOC); source = read(BASE / '2026_MathorCup_大数据竞赛_总论文_v3.docx')
    root = E.fromstring(files['word/document.xml']); old = E.fromstring(source['word/document.xml'])
    body = root.find(W + 'body'); oldbody = old.find(W + 'body')
    relmap = {x.get('Id'): x.get('Target') for x in E.fromstring(files['word/_rels/document.xml.rels'])}
    captions = [x for x in body if re.match(r'^图\d+\s', text(x))]
    supplements = [x for x in body if re.match(r'^附图S\d+\s', text(x))]
    newcap = next(x for x in captions if text(x) == CAPTION)
    newimage = newcap.getprevious(); intro = newimage.getprevious()
    assert text(intro) == INTRO
    start = next(i for i, x in enumerate(body) if text(x) == '问题重述')
    oldstart = next(i for i, x in enumerate(oldbody) if text(x) == '问题重述')
    newsequence = [(x.tag, text(x)) for x in list(body)[start:] if x not in [newcap, newimage, intro]]
    expected = [(x.tag, remap(text(x))) for x in list(oldbody)[oldstart:]]
    oldmath = old.xpath('//m:oMath', namespaces=N); newmath = root.xpath('//m:oMath', namespaces=N)
    sig = lambda x: E.tostring(x, method='c14n', exclusive=True)
    allmathsame = len(oldmath) == len(newmath) == 237 and all(sig(a) == sig(b) for a, b in zip(oldmath, newmath))
    newtables = body.findall(W + 'tbl'); oldtables = oldbody.findall(W + 'tbl')
    alltablesame = len(newtables) == len(oldtables) == 43 and all(sig(a) == sig(b) for a, b in zip(oldtables, newtables))
    def equation(t):
        return re.fullmatch(r'\(\d+\)', text(t.findall('w:tr/w:tc', N)[-1]))
    equations = [t for t in newtables if equation(t)]
    refs = [text(x) for x in body if re.match(r'^\[\d+\]\s', text(x))]
    oldrefs = [text(x) for x in oldbody if re.match(r'^\[\d+\]\s', text(x))]
    word = '\n'.join(root.xpath('//w:t/text()', namespaces=N))
    oldword = '\n'.join(old.xpath('//w:t/text()', namespaces=N))
    oldcount = Counter(map(int, re.findall(r'(?<!附)图\s*(\d+)', oldword)))
    newcount = Counter(map(int, re.findall(r'(?<!附)图\s*(\d+)', word)))
    expected_count = Counter({n + 1: count for n, count in oldcount.items()}); expected_count[1] = 2
    notes = [x for x in body if text(x).startswith('式中')]
    proofs = [x for x in body if text(x).startswith('证明。')]
    snapshot = json.loads((AUDIT / '旧版本保护快照.json').read_text())
    protected = {p: sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in snapshot.items()}
    p = json.loads((AUDIT / 'Word渲染/word_pagination.json').read_text(encoding='utf-8-sig'))
    pdf = pymupdf.open(DOC.with_suffix('.pdf')); pdftext = '\n'.join(page.get_text() for page in pdf)
    toc = [x for x in p['toc'].split('\r') if x.strip()]
    mainheads = p['headings'][:next(i + 1 for i, h in enumerate(p['headings']) if h['text'] == '参考文献')]
    tocok = len(toc) == len(mainheads) == 27 and all(int(row.rsplit('\t', 1)[1]) == h['body_page'] for row, h in zip(toc, mainheads))
    allcaptions = [text(x) for x in body if re.match(r'^(?:图\d+|附图S\d+)\s', text(x))]
    # Word's standard caption enumeration omits 附图; locate the latter in the
    # actual PDF instead of assuming all figures are listed there.
    maincaps = [x for x in p['captions'] if x['text'].startswith('图')]
    imgpages = [x['page'] for x in p['images']]
    # Supplemental caption text may contain font-spacing differences in PDF.
    supplement_pages = {text(x).split(' ', 1)[0]: next(i + 1 for i, page in enumerate(pdf)
                         if re.sub(r'\s+', '', text(x)) in re.sub(r'\s+', '', page.get_text())) for x in supplements}
    mathtypefaces = {span['font'] for page in pdf for block in page.get_text('dict')['blocks']
                    for line in block.get('lines', []) for span in line['spans'] if 'CambriaMath' in span['font']}
    before_parts = set(source); after_parts = set(files)
    oldmedia = [n for n in source if n.startswith('word/media/')]
    newmedia = [n for n in files if n.startswith('word/media/') and n not in source]
    newrid = newimage.find('.//a:blip', N).get('{' + N['r'] + '}embed')
    routepart = 'word/' + relmap[newrid]
    changed = {n for n in before_parts if source[n] != files[n]}
    routepage = next(x['page'] for x in maincaps if x['text'] == CAPTION)
    appendix_page = next(x['physical_page'] for x in p['headings'] if x['text'] == '附录')
    body_pages = next(x['body_page'] for x in p['headings'] if x['text'] == '附录') - 1
    forbidden = r'微网|光储|购电|储能|电价|\bE\d{3}\b|BZD|\bagent\d*\b|学长版|参考组|对照组|晋级|门槛'
    graph_names = all(d.get('name') == text(cap).split(' ', 1)[0]
                      and d.get('descr') == text(cap)
                      for cap in captions + supplements for d in cap.getprevious().xpath('.//wp:docPr | .//pic:cNvPr', namespaces=N))
    gates = {
        'v1_v2_v3_and_notes_unchanged': all(protected.values()),
        'only_body_relationships_and_new_image_changed': changed == {'word/document.xml', 'word/_rels/document.xml.rels'} and after_parts - before_parts == {routepart},
        'all20_existing_media_bytes_unchanged': len(oldmedia) == 20 and all(files[n] == source[n] for n in oldmedia),
        'approved_png_embedded_exactly': newmedia == [routepart] and files[routepart] == FIGURE.read_bytes(),
        'only_allowed_prose_and_numbering_changes': newsequence == expected,
        'route_after_question_analysis': text(intro.getprevious()).startswith('风格生成需要') and text(newcap.getnext()) == '模型假设',
        'prose_explicitly_references_route': '三问技术路线如图1所示' in text(intro),
        'crossquestion_boundaries_preserved': 'NNLS归因仅作独立解释' in text(intro) and '该排序不作为第三问优化输入' in text(intro) and '0-6七档仅表示相对强度' in text(intro),
        'main_figures1_to18': [int(re.match(r'图(\d+)', text(x))[1]) for x in captions] == list(range(1, 19)),
        'all_main_figure_reference_counts_mapped': newcount == expected_count,
        'supplements_S1_S2_S3_unchanged': [text(x) for x in supplements] == [text(x) for x in oldbody if re.match(r'^附图S\d+\s', text(x))],
        'drawing_names_and_alt_synced': graph_names,
        'all43_tables_byte_equivalent': alltablesame,
        'all237_math_objects_byte_equivalent': allmathsame,
        'equation_numbering1_to23_unchanged': [text(t.findall('w:tr/w:tc', N)[-1]) for t in equations] == [f'({i})' for i in range(1, 24)],
        'empty_math_run0': not root.xpath('//m:r[not(m:t) or not(normalize-space(m:t))]', namespaces=N),
        '23_formula_explanations_still_flush': len(notes) == 23 and all(x.find('w:pPr/w:ind', N).get(W + 'firstLine') == '0' and x.find('w:pPr/w:ind', N).get(W + 'firstLineChars') == '0' for x in notes),
        '3_QED_retained': len(proofs) == 3 and pdftext.count('□') == 3,
        'references14_unchanged': refs == oldrefs and len(refs) == 14,
        'toc27_pages_correct': tocok,
        'analysis_heading_numbers_not_duplicated': all(h['number'] == '' for h in p['headings'] if re.match(r'^2\.[123] ', h['text'])),
        'toc_second_page_no_appendix_or_AI': '目 录' in pdf[1].get_text() and all('附录' not in x and 'AI 工具' not in x for x in toc),
        'all43_tables_not_split': all(x['start_page'] == x['end_page'] for x in p['tables']),
        'all18_main_figures_with_captions': len(maincaps) == 18 and len(imgpages) == 21 and all(c['page'] == im for c, im in zip(maincaps, imgpages[:18])),
        'supplemental_figures_with_captions': list(supplement_pages.values()) == imgpages[18:],
        'math_font_CambriaMath': bool(mathtypefaces),
        'body_le30_pages': body_pages <= 30,
        'no_unwanted_dash_or_voice': not re.search(r'[—–]|本研究', word + pdftext),
        'no_internal_or_old_template_words': not re.search(forbidden, word, re.I),
        'cover_placeholders_retained': all(x in pdf[0].get_text() for x in ['【题目待定】', '【摘要下一轮撰写】', '【关键词下一轮确定】']),
    }
    out = {'passed': all(gates.values()), 'checks': {k: bool(v) for k, v in gates.items()},
           'physical_pages': len(pdf), 'body_pages': body_pages, 'appendix_physical_page': appendix_page,
           'route_physical_page': routepage, 'route_body_page': routepage - 2,
           'main_figures': 18, 'supplemental_figures': 3, 'content_tables': 20,
           'display_formulas': len(equations), 'native_math_objects': len(newmath), 'references': len(refs),
           'old_figure_reference_counts': dict(oldcount), 'new_figure_reference_counts': dict(newcount),
           'protected_files': protected, 'math_fonts': sorted(mathtypefaces),
           'files': {str(x.relative_to(ROOT)): {'sha256': sha256(x.read_bytes()).hexdigest(), 'bytes': x.stat().st_size}
                     for x in [DOC, DOC.with_suffix('.pdf')]}}
    (AUDIT / 'final_checks.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not out['passed']:
        raise SystemExit('v4 audit failed')


if __name__ == '__main__':
    main()
