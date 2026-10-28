"""Audit the Appendix A edit against V8 and the actual Windows package folder."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
import csv
import json
import re
import subprocess

from lxml import etree as E
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v8.docx'
DOCX = BASE.with_name(BASE.name.replace('_v8', '_v9'))
PDF = DOCX.with_suffix('.pdf')
AUDIT = ROOT / 'docs/collab/20261028_总论文v9附录A'
ATTACHMENT = Path('/mnt/d/上海大学__/26年妈妈杯数模/终稿/A2601605附件')
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=N))


def canonical(node):
    return E.tostring(node, method='c14n', exclusive=True)


def windows(path):
    return subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()


def main():
    with ZipFile(BASE) as base, ZipFile(DOCX) as edited:
        assert base.testzip() is None and edited.testzip() is None
        changed = [n for n in base.namelist() if base.read(n) != edited.read(n)]
        assert changed == ['word/document.xml'], changed
        before = E.fromstring(base.read('word/document.xml'))
        after = E.fromstring(edited.read('word/document.xml'))
    before_blocks = list(before.find('w:body', N))
    after_blocks = list(after.find('w:body', N))
    start = next(i for i, p in enumerate(before_blocks) if text(p) == '待支撑材料整理后补充')
    end = next(i for i, p in enumerate(after_blocks) if text(p) == '附录B 补充表格与图片')
    assert all(canonical(a) == canonical(b) for a, b in zip(before_blocks[:start], after_blocks[:start]))
    assert len(before_blocks[start+1:]) == len(after_blocks[end:])
    assert all(canonical(a) == canonical(b) for a, b in zip(before_blocks[start+1:], after_blocks[end:]))
    added = after_blocks[start:end]
    assert [E.QName(p).localname for p in added] == ['p', 'p', 'tbl', 'p', 'tbl']
    spec = json.loads((AUDIT / 'appendix_a_spec.json').read_text())
    assert text(added[0]) == spec['intro']
    assert 'A2601605附件.zip' in text(added[0])
    for table, rows in [(added[2], spec['overview']), (added[4], spec['programs'])]:
        actual = [[text(c) for c in r.findall('w:tc', N)] for r in table.findall('w:tr', N)]
        assert actual == rows
        assert table.find('w:tblPr/w:tblStyle', N).get('{%s}val' % N['w']) == '45'
        assert all(r.find('w:trPr/w:cantSplit', N) is not None for r in table.findall('w:tr', N)[1:])
        assert table.findall('w:tr', N)[0].find('w:trPr/w:tblHeader', N) is not None
    listed = [row[1] for row in spec['programs'][1:]]
    for relative in listed:
        assert (ATTACHMENT / relative.replace('\\', '/')).is_file(), relative
    actual_sources = {p.relative_to(ATTACHMENT).as_posix().replace('/', '\\') for p in (ATTACHMENT / 'A2601605源程序').iterdir() if p.is_file()}
    assert actual_sources == set(listed[:-1])
    frozen_inventory = json.loads((AUDIT / '实际附件目录盘点.json').read_text())
    for row in frozen_inventory['files']:
        assert digest(ATTACHMENT / row['relative_path']) == row['sha256']
    assert len([p for p in ATTACHMENT.rglob('*') if p.is_file()]) == 181

    assert len(after.xpath('//m:oMath', namespaces=N)) == 236
    assert not after.xpath('//m:oMath//m:r[not(m:t) or not(normalize-space(m:t))]', namespaces=N)
    assert '待支撑材料整理后补充' not in text(after)
    new_text = ''.join(text(p) for p in added)
    banned = re.findall(r'\bE\d{3}\b|BZD|agent\d*|学长版|微网|光储|储能|电价', new_text, re.I)
    assert not banned, banned

    pagination = json.loads((AUDIT / 'word_pagination.json').read_text(encoding='utf-8-sig'))
    a_tables = pagination['tables'][-2:]
    assert all(t['start_page'] == t['end_page'] for t in a_tables), a_tables
    assert [t['start_page'] for t in a_tables] == [28, 29]
    pdf = pymupdf.open(PDF)
    base_pdf = pymupdf.open(BASE.with_suffix('.pdf'))
    assert len(pdf) == 32
    normalize = lambda value: re.sub(r'\s+', '', value)
    for i in range(27):
        assert normalize(pdf[i].get_text()) == normalize(base_pdf[i].get_text()), ('body page changed', i+1)
    assert '参考文献' in pdf[26].get_text() and '附录' in pdf[27].get_text()
    assert '附录' not in pdf[1].get_text()
    for page_number, caption in [(28, '表A.1'), (29, '表A.2')]:
        page = pdf[page_number-1]
        assert caption in normalize(page.get_text())
        page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4), alpha=False).save(AUDIT / f'附录A_第{page_number}页.png')
    table2_lines = pdf[28].get_text().splitlines()
    assert any(line.strip().replace(' ', '') == '序号' for line in table2_lines), '序号表头不应分成两行'
    for page in pdf:
        for word in page.get_text('words'):
            assert word[0] >= -1 and word[1] >= -1 and word[2] <= page.rect.width+1 and word[3] <= page.rect.height+1

    cli = Path('/home/mary/.codex/skills/software-copyright-materials/vendor/docx-toolkit/scripts/dotnet/DocxToolkit.Cli/bin/Debug/net8.0/DocxToolkit.Cli.dll')
    sdk = {}
    for label, flags in [('structural', []), ('business', ['--business']), ('template_gate', ['--gate-check', windows(Path('/mnt/d/上海大学__/26年数学建模/26数模国赛/0043优化成果/定稿.docx'))])]:
        p = subprocess.run(['/mnt/c/Program Files/dotnet/dotnet.exe', windows(cli), 'validate', '--input', windows(DOCX), *flags, '--json'], capture_output=True, check=True)
        sdk[label] = json.loads(p.stdout.decode('utf-8-sig'))
        assert sdk[label]['isValid'], (label, sdk[label])
    (AUDIT / 'toolkit_validation.json').write_text(json.dumps(sdk, ensure_ascii=False, indent=2), encoding='utf-8')
    raw_sdk = json.loads((AUDIT / 'sdk_validation.json').read_text())
    assert raw_sdk['newErrorCount'] == 0
    checks = {
        'only_appendix_a_changed': True,
        'other_docx_parts_byte_identical': True,
        'first_27_pdf_pages_text_unchanged': True,
        'template_table_style_retained': True,
        'actual_paths_all_exist': True,
        'all_30_programs_listed': True,
        'windows_181_files_unchanged': True,
        'appendix_tables_do_not_split_pages': True,
        'appendix_caption_and_table_on_same_page': True,
        'sequence_header_single_line': True,
        'math_236_empty_runs_zero': True,
        'toolkit_structure_business_gate_pass': True,
        'new_sdk_findings_zero': True,
        'v8_preserved': digest(BASE) == 'd7f0519a6ab3ec4cabeb59acf530dfb8699df9597e0f73490dc8c60964125fd5',
    }
    assert all(checks.values()), checks
    result = {'checks': checks, 'all_pass': True,
              'counts': {'pdf_pages': len(pdf), 'body_pages_including_references': 25,
                         'appendix_a_pdf_pages': [28, 29], 'overview_rows': 5,
                         'program_rows': 31, 'source_programs': 30,
                         'attached_files': 181, 'formulas': 236,
                         'empty_math_runs': 0, 'paper_figures': 21},
              'files': {p.suffix[1:]: {'path': str(p.relative_to(ROOT)), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in [DOCX, PDF]},
              'archive_name': 'A2601605附件.zip',
              'limits': frozen_inventory['observations'],
              'sdk_note': f'严格SDK扫描沿用源稿的{raw_sdk["originalErrorCount"]}项原有兼容性记录，新增0；Toolkit结构、业务与模板门控检查均通过。'}
    (AUDIT / 'v9_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
