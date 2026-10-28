"""Check manuscript-only changes, source excerpts, archive and live Word TOC."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from lxml import etree as E
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/collab/20261028_V13修订'
PAPER = ROOT / '05_论文/总论文'
BASE = PAPER / '2026_MathorCup_大数据竞赛_总论文_v12.docx'
DOCX = PAPER / '2026_MathorCup_大数据竞赛_总论文_v13.docx'
PDF = DOCX.with_suffix('.pdf')
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W = '{' + N['w'] + '}'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sig(node):
    return node.tag, tuple(sorted(node.attrib.items())), node.text or '', tuple(sig(c) for c in node)


def text(node):
    return ''.join(node.xpath('.//w:t/text()|.//m:t/text()', namespaces=N))


def windows(path):
    return subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()


def main():
    snapshot = json.loads((AUDIT / '修订前保护快照.json').read_text())
    spec = json.loads((AUDIT / '修订规格.json').read_text())
    code_map = json.loads((AUDIT / '附录14节_真实源范围.json').read_text())
    package = json.loads((AUDIT / '附件打包与14命令检查.json').read_text())
    with ZipFile(BASE) as z:
        original = {n: z.read(n) for n in z.namelist()}
    with ZipFile(DOCX) as z:
        edited = {n: z.read(n) for n in z.namelist()}
        assert z.testzip() is None
    old, new = E.fromstring(original['word/document.xml']), E.fromstring(edited['word/document.xml'])
    ob, nb = old.find('w:body', N), new.find('w:body', N)
    checks = {}
    for name, root in [('V12', PAPER), ('delivery', ROOT / '06_交付'),
                       ('windows_support', Path('/mnt/d/上海大学__/26年妈妈杯数模/终稿/A2601605附件'))]:
        checks[name + '_existing_files_unchanged'] = all(digest(root / p) == h for p, h in snapshot[name].items())
    checks['only_document_and_custom_styles_changed'] = sorted(n for n in original if original[n] != edited[n]) == ['word/document.xml', 'word/styles.xml']
    checks['all_images_bytes_unchanged'] = all(edited[n] == v for n, v in original.items() if n.startswith('word/media/'))
    checks['21_image_objects'] = len(new.xpath('//w:drawing', namespaces=N)) == 21
    checks['page_settings_unchanged'] = [sig(x) for x in old.xpath('//w:sectPr', namespaces=N)] == [sig(x) for x in new.xpath('//w:sectPr', namespaces=N)]
    def equations(root):
        return [e for t in root.find('w:body', N).findall('w:tbl', N)
                if len(t.findall('w:tr/w:tc', N)) == 3 for e in t.xpath('.//m:oMath', namespaces=N)]
    oe, ne = equations(old), equations(new)
    checks['23_displayed_equations_unchanged'] = len(oe) == len(ne) == 23 and all(sig(a) == sig(b) for a, b in zip(oe, ne))
    checks['native_math_empty_runs_zero'] = not new.xpath('//m:oMath//m:r[not(m:t) or not(string(m:t))]', namespaces=N)
    # Existing content table values remain intact; added T14 rows carry no result numbers.
    old_tables = [t for t in ob.findall('w:tbl', N) if len(t.findall('w:tr', N)) > 1]
    new_tables = [t for t in nb.findall('w:tbl', N) if len(t.findall('w:tr', N)) > 1]
    numbers = lambda node: re.findall(r'[−+\-]?\d+(?:\.\d+)?', re.sub(r'\[\d+\]', '', text(node)))
    checks['existing_content_table_numbers_unchanged'] = len(old_tables) == len(new_tables) and all(numbers(a) == numbers(b) for a, b in zip(old_tables, new_tables))
    old_styles = E.fromstring(original['word/styles.xml'])
    new_styles = E.fromstring(edited['word/styles.xml'])
    for s in old_styles.findall(W + 'style'):
        matching = new_styles.xpath('./w:style[@w:styleId=$s]', namespaces=N, s=s.get(W + 'styleId'))
        assert len(matching) == 1 and sig(s) == sig(matching[0])
    checks['original_styles_semantically_unchanged'] = True
    paragraphs = nb.findall('w:p', N)
    for section in spec['sections']:
        for block in section['blocks']:
            h = next(p for p in paragraphs if text(p) == block['title'])
            cursor, source = h.getnext(), []
            while cursor is not None:
                style = cursor.find('w:pPr/w:pStyle', N)
                if style is not None and style.get(W + 'val') == '49':
                    source.append(text(cursor))
                elif source:
                    break
                cursor = cursor.getnext()
            checks['exact_source_' + block['title'][:3]] = source == block['source_lines']
    all_text = text(nb)
    ref_heading = next(i for i, node in enumerate(nb) if text(node) == '参考文献')
    appendix_heading = next(i for i, node in enumerate(nb) if text(node) == '附录')
    main_text = '\n'.join(text(p) for p in list(nb)[:ref_heading])
    checks['T5_body_internal_words_zero'] = not re.search('路径池|Pareto|审计|协议|冻结', main_text)
    appendix_text = '\n'.join(text(p) for p in list(nb)[appendix_heading:])
    checks['no_old_core_subdirectories_or_modules'] = not re.search(r'Q1_感官预测/|Q2_关键化合物/|Q3_风格生成/|nnls_core\.py|rank_core\.py|style_copula\.py|inner_oof_excerpt\.txt|four_route_excerpt\.txt|核心流程摘录\.txt', appendix_text)
    checks['14_real_compile_commands'] = all_text.count('python -m py_compile') == package['command_count'] == 14 and all(c['exit_code'] == 0 for c in package['commands_run'])
    checks['all_annex_source_names_in_A2'] = all(b['file'] in text(next(t for t in nb.findall('w:tbl', N) if '支撑材料相对路径' in text(t))) for b in code_map['sections'])
    checks['archive_181_files'] = package['files'] == 181
    checks['archive_31_A2_paths_exist'] = package['appendix_A2_real_paths'] == 31
    checks['AI_missing_file_marked_pending'] = 'AI工具使用详情另行补充，未计入现有附件包' in all_text
    refs = [p for p in paragraphs if re.match(r'^\[\d+\]', text(p))]
    checks['14_reference_entries'] = [int(re.match(r'^\[(\d+)\]', text(p))[1]) for p in refs] == list(range(1, 15))
    before = '\n'.join(text(p) for p in list(nb)[:ref_heading])
    cited = [int(v) for v in re.findall(r'\[(\d+)\]', before)]
    order = list(dict.fromkeys(v for v in cited if v >= 3))
    checks['technical_references_in_first_appearance_order'] = order == list(range(3, 15))
    ai_p = next(p for p in paragraphs if text(p).startswith('本参赛队在竞赛过程中使用了AI工具'))
    checks['AI_1_2_cited_in_statement'] = '[1][2]' in text(ai_p)
    checks['all_references_cited'] = set(cited + [int(v) for v in re.findall(r'\[(\d+)\]', text(ai_p))]) == set(range(1, 15))
    checks['no_false_retrospective_preregistration'] = not any(x in all_text for x in ['该规则在看到结果前已固定', '宏R²第一的预登记'])
    toc_title = paragraphs.index(next(p for p in paragraphs if text(p) == '目录'))
    cache = []
    for p in paragraphs[toc_title + 1:]:
        if p.find('w:pPr/w:sectPr', N) is not None:
            break
        if text(p).strip():
            cache.append(p)
    checks['TOC_5_appendices_only'] = sum(bool(re.match(r'^附录[A-E] ', text(p))) for p in cache) == 5 and not any(re.match(r'^[CDE]\.\d', text(p)) for p in cache)
    checks['no_AI_statement_in_TOC'] = not any('AI工具' in text(p).replace(' ', '') for p in cache)
    names = set(new.xpath('//w:bookmarkStart/@w:name', namespaces=N))
    links = [name for p in cache for name in p.xpath('.//w:hyperlink/@w:anchor', namespaces=N)]
    checks['TOC_links_have_targets'] = bool(links) and all(name in names for name in links)
    pdf = pymupdf.open(PDF)
    record = json.loads((AUDIT / 'Word分页.json').read_text(encoding='utf-8-sig'))
    checks['PDF_same_Word_render_page_count'] = len(pdf) == record['pages']
    body_start = next(i + 1 for i, p in enumerate(pdf) if '1.1 问题研究背景' in p.get_text() and i > 1)
    appendix_start = next(i + 1 for i, p in enumerate(pdf) if '表A.1 支撑材料文件列表' in p.get_text())
    checks['body_le_30_pages'] = appendix_start - body_start <= 30
    checks['title_abstract_and_team_id_retained'] = 'MCB2601605' in pdf[0].get_text() and '关键词' in pdf[0].get_text()
    checks['T1_named_final_PDF_exists_and_matches'] = digest(Path('/mnt/d/上海大学__/26年妈妈杯数模/终稿/论文/A2601605.pdf')) == digest(PDF)
    overflow = []
    for i, page in enumerate(pdf):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if min(x0, y0) < -0.5 or x1 > page.rect.width + .5 or y1 > page.rect.height + .5:
                        overflow.append({'page': i + 1, 'text': span['text'][:60]})
    checks['PDF_text_within_page'] = not overflow
    split_tables = [t for t in record['tables'] if t['start_page'] != t['end_page']]
    checks['tables_do_not_split_pages'] = not split_tables
    # Verify cached TOC numbers against the exact headings Word paginated.
    heads = {h['text']: h for h in record['headings']}
    mismatches = []
    for p in cache:
        label = text(p)
        target = next((h for title, h in heads.items() if label.startswith(title) or label.startswith(h.get('number', '') + ' ' + title)), None)
        if target:
            number = int(re.search(r'(\d+)\s*$', label)[1])
            if number != target['body_page']:
                mismatches.append((label, target['body_page']))
    checks['TOC_numbers_match_Word_heading_pages'] = not mismatches

    cli = Path('/home/mary/.codex/skills/software-copyright-materials/vendor/docx-toolkit/scripts/dotnet/DocxToolkit.Cli/bin/Debug/net8.0/DocxToolkit.Cli.dll')
    reference = Path('/mnt/d/上海大学__/26年数学建模/26数模国赛/0043优化成果/定稿.docx')
    def validate(item):
        label, flags = item
        result = subprocess.run(['/mnt/c/Program Files/dotnet/dotnet.exe', windows(cli), 'validate', '--input', windows(DOCX), *flags, '--json'], capture_output=True, check=True)
        return label, json.loads(result.stdout.decode('utf-8-sig'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        gates = dict(pool.map(validate, [('structural', []), ('business', ['--business']), ('template_gate', ['--gate-check', windows(reference)])]))
    (AUDIT / 'Toolkit检查.json').write_text(json.dumps(gates, ensure_ascii=False, indent=2))
    checks['Toolkit_all_gates_pass'] = all(v['isValid'] for v in gates.values())
    checks['SDK_no_new_findings'] = json.loads((AUDIT / 'SDK检查.json').read_text())['newErrorCount'] == 0
    locations = {}
    needles = {'T7_Q1参数': 'RepeatedStratifiedKFold', 'T8_Q2重采样': 'SeedSequence',
               'T9_Q3端点': '同分位端点', 'T10_4VG解释': '表16 中4VG',
               'T5_口径解释': '候选池891', 'T4_参考文献': '[1] OpenAI',
               '附录A': '表A.1 支撑材料文件列表', '附录C': 'C.1 感官组',
               '附录D': 'D.1 完整名次', '附录E': 'E.1 风格收缩'}
    for label, needle in needles.items():
        found = [i + 1 for i, page in enumerate(pdf) if needle.replace(' ', '') in page.get_text().replace(' ', '').replace('\n', '')]
        locations[label] = found
        if found:
            i = found[-1] - 1 if label.startswith('附录') else found[0] - 1
            pdf[i].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(AUDIT / f'验收_{label}_{i+1}页.png')
    pdf[1].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(AUDIT / '验收_目录.png')
    result = {'passed': all(checks.values()), 'checks': checks, 'pdf_pages': len(pdf),
              'body_pages_including_references': appendix_start - body_start,
              'appendix_A_physical_page': appendix_start, 'TOC_entries': len(cache),
              'locations': locations, 'omml': len(new.xpath('//m:oMath', namespaces=N)),
              'display_equations': 23, 'figures': 21, 'code_sections': 14,
              'real_code_lines': sum(len(x['source_lines']) for x in code_map['sections']),
              'technical_citation_order': order, 'AI_citation_order_exception': 'AI1/2为用户固定工具条目，在文后声明引用；技术文献3-14按首次出现顺序排列。',
              'unresolved': ['AI工具使用详情说明.pdf尚未提供，未伪造，附录A已标待补。',
                             '包内TXT和部分绘图/默认数据路径原样保留，存在旧目录引用；本轮禁止修改30个扁平源文件，因此不宣称全工程一键复现。',
                             '是否达到90分需使用同一冻结框架独立复评，不能用任务加分预期冒充实评分。'],
              'overflow': overflow, 'split_tables': split_tables, 'TOC_page_mismatches': mismatches,
              'files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in [DOCX, PDF]}}
    (AUDIT / 'V13验收.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (PAPER / 'SHA256SUMS_v13').write_text(''.join(f'{v["sha256"]}  {k}\n' for k, v in result['files'].items()))
    print(json.dumps({k: v for k, v in result.items() if k not in ['checks', 'files', 'locations']}, ensure_ascii=False, indent=2))
    assert result['passed'], [k for k, v in checks.items() if not v]


if __name__ == '__main__':
    main()
