"""Add only the user-requested per-question premises to approved abstract prose."""
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
CONDITIONS = {
    '针对问题一，': '在化学测量存在缺失、感官目标为标准化连续值的条件下，',
    '针对问题二，': '在化合物相关数据与实测特征均已给定、关键因素限定为严格单体的条件下，',
    '针对问题三，': '在风格标签已知、目标风格样本较少且迁移须满足缺陷不恶化约束的条件下，',
}


def main():
    source = BASE / '2026_MathorCup_大数据竞赛_总论文_v6.docx'
    target = BASE / '2026_MathorCup_大数据竞赛_总论文_v7.docx'
    with ZipFile(source) as z:
        files = {n: z.read(n) for n in z.namelist()}
    root = E.fromstring(files['word/document.xml']); body = root.find(W + 'body')
    changes = []
    for p in list(body)[:8]:
        nodes = p.xpath('.//w:t', namespaces=N)
        value = ''.join(t.text or '' for t in nodes)
        for prefix, premise in CONDITIONS.items():
            if value.startswith(prefix):
                assert nodes[0].text == prefix
                assert nodes[1].text and not nodes[1].text.startswith('在')
                nodes[1].text = premise + nodes[1].text
                changes.append({'prefix': prefix, 'added_condition': premise})
    assert len(changes) == 3
    files['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    with ZipFile(target, 'w', ZIP_DEFLATED) as z:
        for n, raw in files.items():
            z.writestr(n, raw)
    record = {'source': str(source.relative_to(ROOT)), 'output': str(target.relative_to(ROOT)),
              'user_request': '就是要说一下条件', 'changes': changes,
              'source_docx_sha256': sha256(source.read_bytes()).hexdigest(),
              'source_pdf_sha256': sha256(source.with_suffix('.pdf').read_bytes()).hexdigest(),
              'no_result_or_model_change': True, 'original_bold_preserved': True}
    (AUDIT / '摘要补条件记录.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(target)


if __name__ == '__main__':
    main()
