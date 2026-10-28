"""Adopt the exact user-approved front-page proof into a preserved v5 copy."""
from copy import deepcopy as cp
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文'
AUDIT = ROOT / 'docs/collab/20261027_总论文v6与核心代码'
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + N['w'] + '}'


def read(path):
    with ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def text(x):
    return ''.join(x.xpath('.//w:t/text()', namespaces=N))


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    source = BASE / '2026_MathorCup_大数据竞赛_总论文_v5.docx'
    front = BASE / '首页_待审计/首页排版预览_待审_v1.docx'
    target = BASE / '2026_MathorCup_大数据竞赛_总论文_v6.docx'
    snapshot = AUDIT / '旧版本保护快照.json'
    if not snapshot.exists():
        protected = [p for p in BASE.iterdir() if p.is_file() and
                     (p.suffix in ['.docx', '.pdf'] or p.name.startswith('SHA256SUMS'))]
        protected += [front, front.with_suffix('.pdf'), BASE / '首页_待审计/题目摘要关键词_候选_v1.json']
        snapshot.write_text(json.dumps({str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                                       for p in protected}, ensure_ascii=False, indent=2))
    files = read(source); reviewed = read(front)
    assert all(files[n] == reviewed[n] for n in ['word/styles.xml', 'word/numbering.xml',
               'word/settings.xml', 'word/theme/theme1.xml', 'word/fontTable.xml'])
    root = E.fromstring(files['word/document.xml']); body = root.find(W + 'body')
    proofroot = E.fromstring(reviewed['word/document.xml']); proofbody = proofroot.find(W + 'body')
    newparagraphs = [cp(p) for p in proofbody if p.tag == W + 'p']
    assert len(newparagraphs) == 8
    assert text(body[4]) == '目 录'
    assert text(body[0]) == '【题目待定】'
    assert text(body[2]) == '【摘要下一轮撰写】'
    assert text(body[3]) == '【关键词下一轮确定】'
    for p in list(body)[:4]:
        body.remove(p)
    for p in reversed(newparagraphs):
        body.insert(0, p)
    files['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    with ZipFile(target, 'w', ZIP_DEFLATED) as z:
        for n, raw in files.items():
            z.writestr(n, raw)
    record = {'approval': '没问题出论文吧。然后把三个问 附录需要的核心代码打包给我 我发给队友',
              'source': str(source.relative_to(ROOT)), 'approved_front': str(front.relative_to(ROOT)),
              'output': str(target.relative_to(ROOT)), 'front_paragraphs_adopted_exactly': 8,
              'title': text(newparagraphs[0]), 'title_status': 'approved',
              'abstract_status': 'approved', 'keywords_status': 'approved',
              'body_and_results_unchanged': True, 'appendix_code_to_teammate_separately': True,
              'no_training': True, 'no_git_commit': True}
    (AUDIT / '首页采用记录.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(target)


if __name__ == '__main__':
    main()
