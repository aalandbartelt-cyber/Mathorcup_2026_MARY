"""Audit exact approved front-matter adoption and unchanged frozen paper body."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import re

from lxml import etree as E
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文'
AUDIT = ROOT / 'docs/collab/20261027_总论文v6与核心代码'
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W = '{' + N['w'] + '}'


def read(p):
    with ZipFile(p) as z:
        return {n: z.read(n) for n in z.namelist()}


def txt(x):
    return ''.join(x.xpath('.//w:t/text() | .//m:t/text()', namespaces=N))


def main():
    path = BASE / '2026_MathorCup_大数据竞赛_总论文_v6.docx'
    files = read(path); oldfiles = read(BASE / '2026_MathorCup_大数据竞赛_总论文_v5.docx')
    proof = read(BASE / '首页_待审计/首页排版预览_待审_v1.docx')
    root = E.fromstring(files['word/document.xml']); old = E.fromstring(oldfiles['word/document.xml'])
    body = root.find(W + 'body'); ob = old.find(W + 'body')
    proot = E.fromstring(proof['word/document.xml']); pb = proot.find(W + 'body')
    sig = lambda x: E.tostring(x, method='c14n', exclusive=True)
    start = next(i for i, x in enumerate(body) if txt(x) == '问题重述')
    oldstart = next(i for i, x in enumerate(ob) if txt(x) == '问题重述')
    p = json.loads((AUDIT / 'Word渲染/word_pagination.json').read_text(encoding='utf-8-sig'))
    pdf = pymupdf.open(path.with_suffix('.pdf')); pdftext = '\n'.join(page.get_text() for page in pdf)
    preview = pymupdf.open(BASE / '首页_待审计/首页排版预览_待审_v1.pdf')
    toc = [x for x in p['toc'].split('\r') if x.strip()]
    heads = p['headings'][:next(i + 1 for i, h in enumerate(p['headings']) if h['text'] == '参考文献')]
    om = old.xpath('//m:oMath', namespaces=N); nm = root.xpath('//m:oMath', namespaces=N)
    maincaps = [x for x in p['captions'] if x['text'].startswith('图')]
    snapshot = json.loads((AUDIT / '旧版本保护快照.json').read_text())
    protected = {n: sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in snapshot.items()}
    fulltext = txt(root)
    exact_ai = json.loads((ROOT / 'docs/collab/20261027_总论文v2定稿/assembly_manifest.json').read_text())
    checks = {
        'all_old_papers_and_approved_proof_unchanged': all(protected.values()),
        'only_document_xml_changed': set(files) == set(oldfiles) and [n for n in files if files[n] != oldfiles[n]] == ['word/document.xml'],
        'approved8_front_paragraphs_exact': all(sig(body[i]) == sig(pb[i]) for i in range(8)),
        'body_text_after_front_unchanged': [(x.tag, txt(x)) for x in list(body)[start:]] == [(x.tag, txt(x)) for x in list(ob)[oldstart:]],
        'all43_tables_unchanged': len(body.findall(W + 'tbl')) == 43 and all(sig(a) == sig(b) for a, b in zip(ob.findall(W + 'tbl'), body.findall(W + 'tbl'))),
        'all236_math_objects_unchanged': len(om) == len(nm) == 236 and all(sig(a) == sig(b) for a, b in zip(om, nm)),
        'all21_media_unchanged': len([n for n in files if n.startswith('word/media/')]) == 21 and all(files[n] == oldfiles[n] for n in files if n.startswith('word/media/')),
        'all_template_styles_unchanged': all(files[n] == oldfiles[n] for n in ['word/styles.xml', 'word/numbering.xml', 'word/settings.xml', 'word/theme/theme1.xml', 'word/fontTable.xml']),
        'cover_pdf_exact_text': re.sub(r'\s+', '', pdf[0].get_text()) == re.sub(r'\s+', '', preview[0].get_text()),
        'no_cover_placeholders_or_markdown_marks': not re.search(r'【题目待定】|【摘要下一轮撰写】|【关键词下一轮确定】|\*\*', fulltext),
        'front_title_abstract_keywords_page1': all(x in re.sub(r'\s+', '', pdf[0].get_text()) for x in ['基于收缩集成与条件Copula的啤酒感官预测及风格生成', '摘 要'.replace(' ', ''), '关键词：', '约束迁移']),
        'toc_on_second_page': '目 录' in pdf[1].get_text() and p['headings'][0]['physical_page'] == 3,
        'toc27_pages_correct': len(toc) == len(heads) == 27 and all(int(row.rsplit('\t', 1)[1]) == h['body_page'] for row, h in zip(toc, heads)),
        'no_appendix_or_AI_in_toc': all('附录' not in row and 'AI 工具' not in row for row in toc),
        'no_table_split': all(t['start_page'] == t['end_page'] for t in p['tables']),
        'all_main_figures_with_captions': len(maincaps) == 18 and all(c['page'] == im['page'] for c, im in zip(maincaps, p['images'][:18])),
        'empty_math_runs0': not root.xpath('//m:r[not(m:t) or not(normalize-space(m:t))]', namespaces=N),
        'proofs_have3_QED': pdftext.count('□') == 3,
        'no_old_B2_B3_subsections': 'B.2 问题二补充图表' not in fulltext and 'B.3 问题三补充图表' not in fulltext,
        'AI_disclosure_unchanged': exact_ai['ai_paragraph_exact'] in fulltext and exact_ai['ai_scope_supplement'] in fulltext,
        'no_disallowed_terms_or_dashes': not re.search(r'[—–]|本研究|\bE\d{3}\b|BZD|\bagent\d*\b|学长版|参考组|对照组|晋级|门槛|微网|光储|购电|储能|电价', fulltext, re.I),
    }
    bodypages = next(h['body_page'] for h in p['headings'] if h['text'] == '附录') - 1
    checks['body_le30_pages'] = bodypages <= 30
    out = {'passed': all(checks.values()), 'checks': checks, 'physical_pages': len(pdf),
           'body_pages': bodypages, 'main_figures': 18, 'supplemental_figures': 3,
           'content_tables': 20, 'numbered_formulas': 23, 'native_math_objects': len(nm),
           'old_version_protection': protected,
           'files': {str(p.relative_to(ROOT)): {'sha256': sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                     for p in [path, path.with_suffix('.pdf')]}}
    (AUDIT / 'paper_checks.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not out['passed']:
        raise SystemExit('v6 paper audit failed')


if __name__ == '__main__':
    main()
