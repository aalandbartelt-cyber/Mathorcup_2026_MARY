"""Remove handoff metadata and apply documented Light+ syntax colors."""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, String

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / '05_论文/总论文'
BASE = PAPER / '2026_MathorCup_大数据竞赛_总论文_v10.docx'
OUT = PAPER / BASE.name.replace('_v10', '_v11')
OLD_SPEC = ROOT / 'docs/collab/20261028_总论文v10核心代码/appendix_core_spec.json'
AUDIT = ROOT / 'docs/collab/20261028_总论文v11附录清洗'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
PALETTE = {'text': '000000', 'keyword': '0000FF', 'control': 'AF00DB',
           'string': 'A31515', 'comment': '008000', 'number': '098658',
           'function': '795E26', 'type': '267F99'}
CONTROL = {'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally',
           'with', 'return', 'raise', 'break', 'continue', 'yield', 'assert', 'pass'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WithoutDocstrings(ast.NodeTransformer):
    def _drop(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node
    visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _drop


def computation(source):
    return ast.dump(WithoutDocstrings().visit(ast.parse(source)), include_attributes=False)


def clean_source(path, lines):
    removed = [line for line in lines if re.match(r'\s*#\s*Source logical name:', line)]
    text = '\n'.join(line for line in lines if line not in removed)
    if path.endswith('style_copula.py'):
        text = text.replace(
            '"""Style-conditioned empirical margins and shrinkage Gaussian Copula.\n\n'
            'Exact computational functions retained from the archived implementation.\n'
            'No input reading, model cache loading or top-level fitting occurs.\n"""',
            '"""Style-conditioned empirical margins and shrinkage Gaussian Copula."""')
        text = text.replace('# Raw covariance is retained only to reproduce the immutable pilot.',
                            '# Unshrunk covariance provides the covariance comparison.')
    if path.endswith('sensory_inference.py'):
        text = text.replace(
            '"""Pure inference with caller-supplied fitted learners and frozen parameters.\n\n'
            'The caller must validate model provenance before constructing the predictor.\n'
            'This module neither loads model caches nor fits learners or stacking weights.\n"""',
            '"""Sensory inference with fitted base learners and fixed stacking weights."""')
        text = text.replace('"""Match the archive\'s storage precision, rather than relax an error gate."""',
                            '"""Check prediction agreement at the stored numerical precision."""')
    if path.endswith('migration_search.py'):
        text = text.replace('# Stable input index breaks exact objective ties; the auditor is not an argument.',
                            '# Stable input index resolves exact objective ties.')
    assert computation('\n'.join(lines)) == computation(text), path
    return text, removed


def highlight(source):
    lines = source.splitlines()
    rows = [{'text': line, 'spans': []} for line in lines]
    tokens = list(PythonLexer(stripnl=False, ensurenl=False, tabsize=0).get_tokens_unprocessed(source))
    assert ''.join(value for _, _, value in tokens) == source
    row = 0
    for pos, kind, value in tokens:
        color, bold = PALETTE['text'], False
        if kind in Comment:
            color = PALETTE['comment']
        elif kind in String:
            color = PALETTE['string']
        elif kind in Number:
            color = PALETTE['number']
        elif kind in Keyword:
            color = PALETTE['control'] if value in CONTROL else PALETTE['keyword']
            bold = True
        elif kind in Name.Function or kind in Name.Decorator:
            color = PALETTE['function']
        elif kind in Name.Class or kind in Name.Namespace or value in {'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set', 'np', 'pd'}:
            color = PALETTE['type']
        elif kind in Name.Builtin or (kind in Name and re.match(r'\s*\(', source[pos + len(value):])):
            color = PALETTE['function']
        fragments = value.split('\n')
        for index, fragment in enumerate(fragments):
            if fragment and row < len(rows):
                rows[row]['spans'].append({'text': fragment, 'keyword': bold, 'color': color})
            if index < len(fragments) - 1:
                row += 1
    assert all(''.join(span['text'] for span in row['spans']) == row['text'] for row in rows)
    return rows


def prepare():
    AUDIT.mkdir(parents=True, exist_ok=True)
    old = json.loads(OLD_SPEC.read_text())
    spec = deepcopy(old)
    spec.update({'replace_existing': True, 'base_sha256': sha(BASE),
                 'base_pdf_sha256': sha(BASE.with_suffix('.pdf')), 'palette': PALETTE})
    intros = {
        'C': '本附录给出感官组收缩NNLS、非负权重求解与训练折内OOF预测的核心程序。',
        'D': '本附录给出四路稳健排序、感官模型独立归因与化学证据核验的核心程序。',
        'E': '本附录给出风格收缩Copula生成、感官画像推理和缺陷约束迁移的核心程序。',
    }
    software = {
        'C': 'Python 3.10.19、NumPy 2.2.6、SciPy 1.15.3、scikit-learn 1.7.2',
        'D': 'Python 3.10.19、NumPy 2.2.6、pandas 2.3.3、scikit-learn 1.7.2',
        'E': 'Python 3.10.19、NumPy 2.2.6、SciPy 1.15.3、pandas 2.3.3、scikit-learn 1.7.2',
    }
    deleted = []
    for section in spec['sections']:
        letter = section['letter']
        section['intro'] = intros[letter]
        for block in section['blocks']:
            source, removed = clean_source(block['path'], block['source_lines'])
            deleted += [{'path': block['path'], 'line': line} for line in removed]
            block['source_lines'] = source.splitlines()
            block['lines'] = highlight(source)
            if block['python_module']:
                block['software'] = f'软件：{software[letter]}。语法检查命令：python -m py_compile {block["path"]}。'
            else:
                block['software'] = f'软件：{software[letter]}。'
            if block['path'].endswith('inner_oof_excerpt.txt'):
                block['context'] = [
                    '输入为当前训练集、验证化学输入、基学习器参数、感官家族、收缩值与内层划分；输出为堆叠预测、非负权重、截距及折内基预测。',
                    'fit_outer_nnls在外层训练集内部构造OOF预测，再拟合非负堆叠权重。']
            elif block['path'].endswith('four_route_excerpt.txt'):
                block['context'] = [
                    'total、bootstrap、adjusted、deletion依次表示总关联、自助稳定性、风格校正和留一风格的完整候选名次表。',
                    '四表按target_code与feature_code对齐；reciprocal_rank_consensus综合支持数与倒数秩生成最终排序。']
            elif block['path'].endswith('核心流程摘录.txt'):
                block['context'] = [
                    '输入包括生成谱的风格与样本编号、50个感官目标、折模型预测、相对等级与支持域标记；输出为逐谱画像及逐风格、逐目标汇总表。',
                    '对模型轴取均值后按风格和感官目标分组，计算均值与分位数。model_fold_sd表示拟合扰动，不作为独立预测置信区间。']
    spec['removed_hash_headers'] = deleted
    spec['highlight_sources'] = [
        'https://github.com/microsoft/vscode/blob/main/extensions/theme-defaults/themes/light_vs.json',
        'https://github.com/microsoft/vscode/blob/main/extensions/theme-defaults/themes/light_plus.json',
        'https://code.visualstudio.com/api/language-extensions/syntax-highlight-guide']
    (AUDIT / 'appendix_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    print('Prepared: removed', len(deleted), 'metadata comments;',
          sum(len(b['lines']) for s in spec['sections'] for b in s['blocks']), 'code lines; executable AST unchanged')


def preserve_parts():
    temporary = OUT.with_suffix('.repack.docx')
    with ZipFile(OUT) as edited:
        document = edited.read('word/document.xml')
    with ZipFile(BASE) as old, ZipFile(temporary, 'w', ZIP_DEFLATED) as new:
        for info in old.infolist():
            new.writestr(info, document if info.filename == 'word/document.xml' else old.read(info.filename))
    temporary.replace(OUT)


def render():
    work = Path('/mnt/c/Users/Public/mathorcup_appendix_v11_20261028')
    work.mkdir(parents=True, exist_ok=True)
    (work / 'export.ps1').write_bytes(b'\xef\xbb\xbf' + (ROOT / 'scripts/export_total_paper_word.ps1').read_bytes())
    local = work / OUT.name
    shutil.copy2(OUT, local)
    windows = subprocess.check_output(['wslpath', '-w', str(work)], text=True).strip()
    subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile',
                    '-ExecutionPolicy', 'Bypass', '-File', windows + '\\export.ps1',
                    '-Source', windows + '\\' + OUT.name, '-OutputDir', windows], check=True)
    shutil.copy2(local.with_suffix('.pdf'), OUT.with_suffix('.pdf'))
    shutil.copy2(work / 'word_pagination.json', AUDIT / 'word_pagination.json')
    print(OUT.with_suffix('.pdf'))


def audit():
    from lxml import etree as E
    import pymupdf
    spec = json.loads((AUDIT / 'appendix_spec.json').read_text())
    old_spec = json.loads(OLD_SPEC.read_text())
    with ZipFile(BASE) as z:
        old_parts = {n: z.read(n) for n in z.namelist()}
    with ZipFile(OUT) as z:
        new_parts = {n: z.read(n) for n in z.namelist()}
    before = E.fromstring(old_parts['word/document.xml'])
    after = E.fromstring(new_parts['word/document.xml'])
    ob, nb = before.find('w:body', NS), after.find('w:body', NS)
    text = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
    canonical = lambda p: E.tostring(p, method='c14n')
    oc = next(i for i, p in enumerate(ob) if text(p) == '附录C 问题一模型核心程序')
    nc = next(i for i, p in enumerate(nb) if text(p) == '附录C 问题一模型核心程序')
    checks = {'v10_docx_preserved': sha(BASE) == spec['base_sha256'],
              'v10_pdf_preserved': sha(BASE.with_suffix('.pdf')) == spec['base_pdf_sha256'],
              'only_document_xml_changed': [n for n in old_parts if old_parts[n] != new_parts[n]] == ['word/document.xml'],
              'all_content_before_C_unchanged': oc == nc and all(canonical(x) == canonical(y) for x, y in zip(ob[:oc], nb[:nc])),
              'section_setup_unchanged': canonical(ob[-1]) == canonical(nb[-1])}
    core_package = ROOT / '05_论文/交付给论文手/三问附录核心代码_v2'
    checks['original_core_package_unchanged'] = all(sha(core_package / p) == h for p, h in spec['package_hashes'].items())
    paragraphs = nb.findall('w:p', NS)
    colors = Counter()
    for section, original_section in zip(spec['sections'], old_spec['sections']):
        for block, original_block in zip(section['blocks'], original_section['blocks']):
            h = next(p for p in paragraphs if text(p) == block['title'])
            p, actual = h.getnext(), []
            assert text(p).startswith('软件：')
            while p is not None:
                style = p.find('w:pPr/w:pStyle', NS)
                if style is not None and style.get('{%s}val' % NS['w']) == '49':
                    actual.append(text(p))
                    for color in p.xpath('./w:r/w:rPr/w:color/@w:val', namespaces=NS):
                        colors[color] += 1
                elif actual:
                    break
                p = p.getnext()
            checks['source_text_' + block['path']] = actual == block['source_lines']
            checks['computation_' + block['path']] = computation('\n'.join(actual)) == computation('\n'.join(original_block['source_lines']))
    full_text = '\n'.join(text(p) for p in nb)
    appendices = '\n'.join(text(p) for p in nb[nc:])
    forbidden = r'哈希|(?i:sha[-_ ]?256|\bhash\b)|Source logical name|verify_core\.py'
    checks['hash_and_source_metadata_zero'] = re.search(forbidden, full_text) is None
    checks['handoff_language_zero'] = not any(x in appendices for x in [
        'WSL', 'Ubuntu', 'conda', 'mathorcup', '核心代码包', '本次交付', '非独立入口',
        '不是独立', '不重新训练', '不重新生成结果', '以下内容来自已整理', '并非从原始附件',
        'archived implementation', 'No input reading', 'error gate', 'the auditor is not'])
    checks['syntax_color_variety'] = set(PALETTE.values()).issubset(colors)
    def luminance(color):
        rgb = [int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
        return sum(a * b for a, b in zip(linear, [.2126, .7152, .0722]))
    contrasts = {k: 1.05 / (luminance(v) + .05) for k, v in PALETTE.items()}
    checks['all_syntax_colors_white_contrast_4_5'] = min(contrasts.values()) >= 4.5
    checks['OMML_236_unchanged'] = len(after.xpath('//m:oMath', namespaces=NS)) == 236
    checks['OMML_empty_runs_zero'] = not after.xpath('//m:oMath//m:r[not(m:t) or not(string(m:t))]', namespaces=NS)
    checks['images_21_unchanged'] = len(after.xpath('//w:drawing', namespaces=NS)) == 21
    old_pdf, pdf = pymupdf.open(BASE.with_suffix('.pdf')), pymupdf.open(OUT.with_suffix('.pdf'))
    checks['PDF_first_31_pages_identical'] = all(old_pdf[i].get_text() == pdf[i].get_text() for i in range(31))
    starts = {}
    for section in spec['sections']:
        needle = section['heading'].replace(' ', '')
        starts[section['letter']] = next(i + 1 for i, page in enumerate(pdf)
            if needle in page.get_text().replace(' ', '').replace('\n', ''))
    overflow = []
    printed_colors = Counter()
    for i in range(32, len(pdf)):
        for block in pdf[i].get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if 'Consolas' in span['font']:
                        printed_colors[f'{span["color"]:06X}'] += 1
                    if min(x0, y0) < -0.5 or x1 > pdf[i].rect.width + 0.5 or y1 > pdf[i].rect.height + 0.5:
                        overflow.append({'page': i + 1, 'text': span['text'][:60]})
    checks['rendered_code_has_all_palette_colors'] = set(PALETTE.values()).issubset(printed_colors)
    checks['rendered_text_within_page'] = not overflow
    checks['PDF_no_hash_metadata'] = re.search(forbidden, '\n'.join(p.get_text() for p in pdf)) is None
    checks['no_appendices_in_TOC'] = not re.search('附录[ABCDE]', pdf[1].get_text())
    for letter, page in starts.items():
        pdf[page - 1].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(AUDIT / f'附录{letter}_第{page}页.png')
    pdf[-1].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(AUDIT / f'末页_第{len(pdf)}页.png')
    result = {'checks': checks, 'passed': all(checks.values()), 'pages': len(pdf),
              'body_pages_including_references': 25, 'appendix_starts': starts,
              'code_lines': sum(len(b['lines']) for s in spec['sections'] for b in s['blocks']),
              'removed_hash_comments': len(spec['removed_hash_headers']),
              'palette': PALETTE, 'white_contrast': contrasts,
              'XML_color_counts': dict(colors), 'PDF_color_counts': dict(printed_colors),
              'overflow': overflow, 'OMML': 236, 'empty_math_runs': 0,
              'sha256': {OUT.name: sha(OUT), OUT.with_suffix('.pdf').name: sha(OUT.with_suffix('.pdf'))}}
    (AUDIT / 'v11_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (PAPER / 'SHA256SUMS_v11').write_text(''.join(f'{h}  {n}\n' for n, h in result['sha256'].items()))
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'XML_color_counts', 'PDF_color_counts')}, ensure_ascii=False, indent=2))
    assert result['passed'], [k for k, v in checks.items() if not v]


if __name__ == '__main__':
    {'prepare': prepare, 'preserve-parts': preserve_parts, 'render': render, 'audit': audit}[sys.argv[1]]()
