"""Remove empty appendix subsections; keep new front matter separate for review."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from hashlib import sha256
import json

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文'
AUDIT = ROOT / 'docs/collab/20261027_总论文v5摘要待审'
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W = '{' + N['w'] + '}'


def text(x):
    return ''.join(x.xpath('.//w:t/text() | .//m:t/text()', namespaces=N))


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    source = BASE / '2026_MathorCup_大数据竞赛_总论文_v4.docx'
    target = BASE / '2026_MathorCup_大数据竞赛_总论文_v5.docx'
    snapshot = AUDIT / '旧版本保护快照.json'
    if not snapshot.exists():
        snapshot.write_text(json.dumps({str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                                       for p in BASE.iterdir() if p.is_file() and
                                       (p.suffix in ['.docx', '.pdf'] or p.name.startswith('SHA256SUMS'))},
                                      ensure_ascii=False, indent=2))
    with ZipFile(source) as z:
        files = {n: z.read(n) for n in z.namelist()}
    root = E.fromstring(files['word/document.xml']); body = root.find(W + 'body')
    deleted = []
    for prefix in ['B.2 问题二补充图表', 'B.3 问题三补充图表']:
        h = next(x for x in body if text(x) == prefix)
        p = h.getnext()
        assert p.tag == W + 'p' and len(text(p)) > 40
        for x in [h, p]:
            deleted.append({'text': text(x), 'omml_count': len(x.xpath('.//m:oMath', namespaces=N))})
            body.remove(x)
    # All front-page placeholders remain until each item is user-approved.
    assert all(x in text(root) for x in ['【题目待定】', '【摘要下一轮撰写】', '【关键词下一轮确定】'])
    files['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    with ZipFile(target, 'w', ZIP_DEFLATED) as z:
        for n, data in files.items():
            z.writestr(n, data)
    record = {'source': str(source.relative_to(ROOT)), 'output': str(target.relative_to(ROOT)),
              'deleted_paragraphs': deleted, 'deleted_math_objects': sum(x['omml_count'] for x in deleted),
              'cover_replaced': False, 'title_review': 'pending', 'abstract_review': 'pending',
              'keywords_review': 'pending', 'no_training': True, 'no_result_changes': True,
              'no_git_commit': True}
    (AUDIT / '删附录清单.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
