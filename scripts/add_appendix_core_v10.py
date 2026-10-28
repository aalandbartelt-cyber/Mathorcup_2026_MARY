"""Prepare and audit the frozen core-code package for manuscript appendices C-E."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import keyword
import re
import sys
import tokenize
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / '05_论文/总论文'
BASE = PAPER / '2026_MathorCup_大数据竞赛_总论文_v9.docx'
OUT = PAPER / BASE.name.replace('_v9', '_v10')
PACKAGE = ROOT / '05_论文/交付给论文手/三问附录核心代码_v2'
AUDIT = ROOT / 'docs/collab/20261028_总论文v10核心代码'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def colored_lines(text):
    """Retain every source character; color only Python keywords, as in 0043."""
    lines = text.splitlines()
    marks = [[False] * len(line) for line in lines]
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.NAME and keyword.iskeyword(tok.string):
                r, c = tok.start
                if r <= len(marks):
                    marks[r - 1][c:tok.end[1]] = [True] * len(tok.string)
    except (tokenize.TokenError, IndentationError):
        pass
    result = []
    for line, mask in zip(lines, marks):
        spans = []
        start = 0
        for pos in range(1, len(line) + 1):
            if pos == len(line) or mask[pos] != mask[start]:
                spans.append({'text': line[start:pos], 'keyword': bool(mask[start])})
                start = pos
        result.append({'text': line, 'spans': spans})
    return result


def prepare():
    AUDIT.mkdir(parents=True, exist_ok=True)
    inventory = {str(p.relative_to(PACKAGE)): sha(p) for p in PACKAGE.rglob('*') if p.is_file()}
    sections = []
    layout = [
        ('C', '问题一', 'Q1_感官预测', '感官组收缩NNLS与折内求解', [
            ('nnls_core.py', '感官组收缩NNLS'),
            ('inner_oof_excerpt.txt', '训练折内OOF与堆叠求解流程摘录')]),
        ('D', '问题二', 'Q2_关键化合物', '稳健四路排序与独立解释', [
            ('rank_core.py', '完整名次适配与倒数秩融合'),
            ('four_route_excerpt.txt', '四路证据连接流程摘录'),
            ('frozen_nnls_core.py', '冻结感官预测与模块置换'),
            ('conditional_core.py', '条件残差置换与误差变化'),
            ('chemical_core.py', 'OAV与化学证据核验')]),
        ('E', '问题三', 'Q3_风格生成', '风格Copula生成与约束迁移', [
            ('style_copula.py', '风格收缩Copula与经验反变换'),
            ('sensory_inference.py', '冻结感官模型推理与画像尺度映射'),
            ('concentration_constraints.py', '浓度尺度转换与缺陷约束'),
            ('migration_search.py', '有限迁移候选与最小改动筛选'),
            ('evidence_status.py', '约束与代理证据状态'),
            ('distribution_diagnostics.py', '分布与相关结构诊断'),
            ('核心流程摘录.txt', '生成谱感官画像汇总流程摘录')]),
    ]
    dependencies = {
        'C': 'Python 3.10.19、NumPy 2.2.6、SciPy 1.15.3；完整折内流程另使用scikit-learn 1.7.2',
        'D': 'Python 3.10.19、NumPy 2.2.6、pandas 2.3.3、scikit-learn 1.7.2',
        'E': 'Python 3.10.19、NumPy 2.2.6、SciPy 1.15.3、pandas 2.3.3、scikit-learn 1.7.2',
    }
    for letter, question, folder, purpose, items in layout:
        section = {
            'letter': letter,
            'heading': f'附录{letter} {question}模型核心程序',
            'intro': f'本附录给出{question}{purpose}的核心程序，使用WSL2 Ubuntu中的conda环境mathorcup。'
                     '以下内容来自已整理的核心代码包，保留实际计算、排序和约束逻辑。'
                     '核心函数依赖调用方提供数据、冻结模型及完整工程接口，并非从原始附件一键复算的独立工程；'
                     '下列命令仅作语法、导入和哈希检查，不重新训练、不重新生成结果。',
            'blocks': [],
        }
        for index, (name, title) in enumerate(items, 1):
            path = PACKAGE / folder / name
            content = path.read_text(encoding='utf-8')
            if path.suffix == '.py':
                ast.parse(content)
                source = content
                context = []
                command = f'python -m py_compile {folder}/{name}'
                note = '该模块没有自动训练或生成入口；编译和统一核验不会调用模型求解函数。'
            elif name == 'inner_oof_excerpt.txt':
                context, source = content.split('# Source logical name:', 1)
                context = context.splitlines()[:5]
                source = '# Source logical name:' + source
                command = 'python verify_core.py'
                note = '此段为训练折内流程摘录，需完整工程提供基学习器与划分接口，不是独立脚本；不执行该函数。'
            elif name == 'four_route_excerpt.txt':
                context, source = content.split('# Source logical name:', 1)
                context = context.splitlines()[:4]
                source = '# Source logical name:' + source
                command = 'python verify_core.py'
                note = '此段为四路连接摘录，需预先提供各路完整候选名次表；不重新计算排名。'
            else:
                context, rest = content.split('```python\n', 1)
                source, _ = rest.split('\n```', 1)
                context = [x for x in context.splitlines()[2:] if x.strip()]
                command = 'python verify_core.py'
                note = '此段为已有生成谱和冻结预测结果的汇总摘录，不是独立生成脚本；不执行画像重算。'
            context = [x.lstrip('- ').replace('`', '') for x in context if x.strip()]
            section['blocks'].append({
                'title': f'{letter}.{index} {title}（{name}）',
                'path': f'{folder}/{name}', 'sha256': sha(path),
                'software': f'软件与命令：使用{dependencies[letter]}。在三问附录核心代码包根目录执行“{command}”，'
                            '并执行“python verify_core.py”统一核验。' + note
                            if command != 'python verify_core.py' else
                            f'软件与命令：使用{dependencies[letter]}。在三问附录核心代码包根目录执行“{command}”作统一静态核验。' + note,
                'context': context,
                'lines': colored_lines(source),
                'source_lines': source.splitlines(),
                'python_module': path.suffix == '.py',
            })
        sections.append(section)
    spec = {'base_sha256': sha(BASE), 'base_pdf_sha256': sha(BASE.with_suffix('.pdf')),
            'package_hashes': inventory, 'sections': sections,
            'source_modules': 11, 'process_excerpts': 3}
    (AUDIT / 'appendix_core_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Prepared 3 appendices, 11 Python modules, 3 process excerpts,',
          sum(len(b['lines']) for s in sections for b in s['blocks']), 'exact code lines')


def preserve_parts():
    temporary = OUT.with_suffix('.repack.docx')
    with ZipFile(OUT) as edited:
        document = edited.read('word/document.xml')
    with ZipFile(BASE) as original, ZipFile(temporary, 'w', ZIP_DEFLATED) as target:
        for info in original.infolist():
            target.writestr(info, document if info.filename == 'word/document.xml' else original.read(info.filename))
    temporary.replace(OUT)
    print('Retained every original ZIP part except word/document.xml')


def audit():
    from lxml import etree as E
    import fitz
    spec = json.loads((AUDIT / 'appendix_core_spec.json').read_text())
    with ZipFile(BASE) as z:
        old = E.fromstring(z.read('word/document.xml'))
        old_parts = {n: z.read(n) for n in z.namelist()}
    with ZipFile(OUT) as z:
        new = E.fromstring(z.read('word/document.xml'))
        changed = [n for n in z.namelist() if z.read(n) != old_parts[n]]
    ob, nb = old.find('w:body', NS), new.find('w:body', NS)
    text = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
    oc = next(i for i, p in enumerate(ob) if text(p) == spec['sections'][0]['heading'])
    nc = next(i for i, p in enumerate(nb) if text(p) == spec['sections'][0]['heading'])
    canonical = lambda x: E.tostring(x, method='c14n')
    checks = {}
    checks['v9_docx_preserved'] = sha(BASE) == spec['base_sha256']
    checks['v9_pdf_preserved'] = sha(BASE.with_suffix('.pdf')) == spec['base_pdf_sha256']
    checks['only_document_part_changed'] = changed == ['word/document.xml']
    checks['all_content_before_C_identical'] = oc == nc and all(canonical(a) == canonical(b) for a, b in zip(ob[:oc], nb[:nc]))
    checks['section_settings_identical'] = canonical(ob[-1]) == canonical(nb[-1])
    checks['core_package_unchanged'] = all(sha(PACKAGE / n) == h for n, h in spec['package_hashes'].items())
    paragraphs = nb.findall('w:p', NS)
    for section in spec['sections']:
        heading = next(p for p in paragraphs if text(p) == section['heading'])
        checks[f"{section['letter']}_new_page"] = heading.find('w:pPr/w:pageBreakBefore', NS) is not None
        for block in section['blocks']:
            h = next(p for p in paragraphs if text(p) == block['title'])
            p = h.getnext()
            assert text(p).startswith('软件与命令：'), block['path']
            actual = []
            while p is not None:
                style = p.find('w:pPr/w:pStyle', NS)
                if style is not None and style.get('{%s}val' % NS['w']) == '49':
                    actual.append(text(p))
                elif actual:
                    break
                p = p.getnext()
            checks['source_exact_' + block['path']] = actual == block['source_lines']
    math = new.xpath('//m:oMath', namespaces=NS)
    empty = new.xpath('//m:oMath//m:r[not(m:t) or not(string(m:t))]', namespaces=NS)
    checks['OMML_236_unchanged'] = len(math) == len(old.xpath('//m:oMath', namespaces=NS)) == 236
    checks['OMML_empty_runs_zero'] = not empty
    checks['images_21_unchanged'] = len(new.xpath('//w:drawing', namespaces=NS)) == 21
    added_text = '\n'.join(text(p) for p in nb[nc:])
    banned = re.findall(r'(?i)\bE\d{3}\b|BZD|agent[23]?|学长版|参考组|对照组|晋级|门槛|/home/|/mnt/[a-z]/|微网|光储|购电|电价', added_text)
    checks['new_appendix_banned_terms_zero'] = not banned
    old_pdf, pdf = fitz.open(BASE.with_suffix('.pdf')), fitz.open(OUT.with_suffix('.pdf'))
    checks['first_31_PDF_pages_text_identical'] = all(old_pdf[i].get_text() == pdf[i].get_text() for i in range(31))
    locations = {}
    for section in spec['sections']:
        needle = section['heading'].replace(' ', '')
        locations[section['letter']] = next(i + 1 for i, page in enumerate(pdf) if needle in page.get_text().replace(' ', '').replace('\n', ''))
    overflow = []
    for i in range(31, len(pdf)):
        for block in pdf[i].get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if x0 < -0.5 or x1 > pdf[i].rect.width + 0.5 or y0 < -0.5 or y1 > pdf[i].rect.height + 0.5:
                        overflow.append({'page': i + 1, 'text': span['text'][:80], 'bbox': span['bbox']})
    checks['appendix_text_within_page'] = not overflow
    fonts = sorted({f[3] for i in range(31, len(pdf)) for f in pdf[i].get_fonts()})
    checks['Consolas_in_rendered_PDF'] = any('Consolas' in f for f in fonts)
    for letter, page in locations.items():
        pdf[page - 1].get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(AUDIT / f'附录{letter}_第{page}页.png')
    # A long dictionary statement must wrap visibly without changing the source.
    last = len(pdf)
    pdf[last - 1].get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(AUDIT / f'末页_第{last}页.png')
    record = {'checks': checks, 'passed': all(checks.values()), 'pages': len(pdf),
              'body_pages_including_references': 25, 'appendix_starts': locations,
              'OMML': len(math), 'empty_math_runs': len(empty), 'images': 21,
              'python_modules': 11, 'process_excerpts': 3,
              'code_lines': sum(len(b['lines']) for s in spec['sections'] for b in s['blocks']),
              'banned_hits': banned, 'overflow': overflow, 'fonts': fonts,
              'sha256': {OUT.name: sha(OUT), OUT.with_suffix('.pdf').name: sha(OUT.with_suffix('.pdf'))}}
    (AUDIT / 'v10_checks.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    (PAPER / 'SHA256SUMS_v10').write_text(''.join(f'{h}  {n}\n' for n, h in record['sha256'].items()), encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k not in ('checks', 'fonts')}, ensure_ascii=False, indent=2))
    if not record['passed']:
        raise SystemExit('FAILED: ' + str([k for k, v in checks.items() if not v]))


if __name__ == '__main__':
    {'prepare': prepare, 'preserve-parts': preserve_parts, 'audit': audit}[sys.argv[1]]()
