"""Verify appendix deletion and the separation of unapproved front matter."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import json
import re

from lxml import etree as E
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文'
AUDIT = ROOT / 'docs/collab/20261027_总论文v5摘要待审'
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
     'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
W = '{' + N['w'] + '}'


def read(path):
    with ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def txt(x):
    return ''.join(x.xpath('.//w:t/text() | .//m:t/text()', namespaces=N))


def main():
    path = BASE / '2026_MathorCup_大数据竞赛_总论文_v5.docx'
    source = read(BASE / '2026_MathorCup_大数据竞赛_总论文_v4.docx'); files = read(path)
    old = E.fromstring(source['word/document.xml']); root = E.fromstring(files['word/document.xml'])
    ob = old.find(W + 'body'); b = root.find(W + 'body')
    record = json.loads((AUDIT / '删附录清单.json').read_text())
    deleted = {x['text'] for x in record['deleted_paragraphs']}
    start = next(i for i, x in enumerate(ob) if txt(x) == '问题重述')
    newstart = next(i for i, x in enumerate(b) if txt(x) == '问题重述')
    expected = [(x.tag, txt(x)) for x in list(ob)[start:] if txt(x) not in deleted]
    actual = [(x.tag, txt(x)) for x in list(b)[newstart:]]
    signature = lambda x: E.tostring(x, method='c14n', exclusive=True)
    om = [x for p in ob if txt(p) not in deleted for x in p.xpath('.//m:oMath', namespaces=N)]
    nm = root.xpath('//m:oMath', namespaces=N)
    pdf = pymupdf.open(path.with_suffix('.pdf')); fulltext = '\n'.join(x.get_text() for x in pdf)
    p = json.loads((AUDIT / 'Word渲染/word_pagination.json').read_text(encoding='utf-8-sig'))
    headings = p['headings'][:next(i + 1 for i, h in enumerate(p['headings']) if h['text'] == '参考文献')]
    toc = [x for x in p['toc'].split('\r') if x.strip()]
    snapshot = json.loads((AUDIT / '旧版本保护快照.json').read_text())
    protected = {name: sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in snapshot.items()}
    gates = {
        'old_versions_protected': all(protected.values()),
        'only_document_xml_changed': set(files) == set(source) and [n for n in files if files[n] != source[n]] == ['word/document.xml'],
        'only_four_authorized_appendix_paragraphs_removed': expected == actual and len(deleted) == 4,
        'appendix_B2_B3_absent': all(x not in txt(root) and x not in fulltext for x in ['B.2 问题二补充图表', 'B.3 问题三补充图表']),
        'all43_tables_unchanged': len(b.findall(W + 'tbl')) == 43 and all(signature(x) == signature(y) for x, y in zip(ob.findall(W + 'tbl'), b.findall(W + 'tbl'))),
        'remaining236_math_objects_unchanged': len(om) == len(nm) == 236 and all(signature(x) == signature(y) for x, y in zip(om, nm)),
        'only_one_redundant_CO2_math_removed': record['deleted_math_objects'] == 1,
        'empty_math_run0': not root.xpath('//m:r[not(m:t) or not(normalize-space(m:t))]', namespaces=N),
        'all21_media_unchanged': len([n for n in files if n.startswith('word/media/')]) == 21 and all(files[n] == source[n] for n in files if n.startswith('word/media/')),
        'all_18_main_figures_and3_supplements': len(root.xpath('//w:drawing', namespaces=N)) == 21,
        'cover_placeholders_retained': all(x in pdf[0].get_text() for x in ['【题目待定】', '【摘要下一轮撰写】', '【关键词下一轮确定】']),
        'draft_title_not_embedded': '基于收缩集成与条件Copula的啤酒感官预测及风格生成' not in txt(root),
        'toc27_pages_correct': len(toc) == len(headings) == 27 and all(int(row.rsplit('\t', 1)[1]) == h['body_page'] for row, h in zip(toc, headings)),
        'tables_not_split': all(x['start_page'] == x['end_page'] for x in p['tables']),
        '3_QED_present': fulltext.count('□') == 3,
        'no_forbidden_terms': not re.search(r'BZD|\bagent\d*\b|学长版|参考组|对照组|晋级|门槛|\bE\d{3}\b|微网|光储|购电|储能|电价', txt(root), re.I),
    }
    out = {'passed': all(gates.values()), 'checks': gates, 'physical_pages': len(pdf),
           'body_pages': next(h['body_page'] for h in p['headings'] if h['text'] == '附录') - 1,
           'native_math_objects': len(nm), 'old_versions_protected': protected,
           'cover_status': 'title/abstract/keywords separately pending user approval',
           'files': {str(q.relative_to(ROOT)): {'sha256': sha256(q.read_bytes()).hexdigest(), 'bytes': q.stat().st_size} for q in [path, path.with_suffix('.pdf')]}}
    (AUDIT / 'final_checks.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not out['passed']:
        raise SystemExit('v5 audit failed')


if __name__ == '__main__':
    main()
