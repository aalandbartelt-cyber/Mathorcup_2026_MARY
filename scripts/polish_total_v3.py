"""Polish an exact v2 copy; frozen model outputs and graphics are read-only.

The separately reviewed technical roadmap is deliberately NOT embedded here.
"""
from copy import deepcopy as cp
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import re
from io import BytesIO

from lxml import etree as E
from PIL import Image
import native_math_total as nm

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文'
AUDIT = ROOT / 'docs/collab/20261027_总论文v3修订'
SOURCE = BASE / '2026_MathorCup_大数据竞赛_总论文_v2.docx'
TARGET = BASE / '2026_MathorCup_大数据竞赛_总论文_v3.docx'
N = {'w': nm.W, 'm': nm.M,
     'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
     'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W = '{' + nm.W + '}'


def text(x):
    return ''.join(x.xpath('.//w:t/text() | .//m:t/text()', namespaces=N))


def prop(x, tag, **attrs):
    y = x.find(W + tag)
    if y is None:
        y = E.SubElement(x, W + tag)
    for k, v in attrs.items():
        y.set(W + k, str(v))
    return y


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    snapshot = AUDIT / '旧版本保护快照.json'
    if not snapshot.exists():
        keep = [p for p in BASE.iterdir() if p.is_file() and
                ('_v1.' in p.name or '_v2.' in p.name or p.name == 'SHA256SUMS_v2')]
        snapshot.write_text(json.dumps({str(p.relative_to(ROOT)): {
            'bytes': p.stat().st_size, 'sha256': sha256(p.read_bytes()).hexdigest()
        } for p in keep}, ensure_ascii=False, indent=2))
    with ZipFile(SOURCE) as z:
        files = {n: z.read(n) for n in z.namelist()}
    root = E.fromstring(files['word/document.xml'])
    body = root.find(W + 'body')
    original = E.fromstring(files['word/document.xml'])
    source_body = original.find(W + 'body')
    paragraphs = body.findall(W + 'p')
    mother = cp(next(p for p in paragraphs
                     if p.find('w:pPr/w:pStyle', N) is not None
                     and p.find('w:pPr/w:pStyle', N).get(W + 'val') == '11'
                     and len(text(p)) > 80))

    def paragraph(value, template=None):
        p = cp(template if template is not None else mother)
        pr = p.find(W + 'pPr')
        if pr is None:
            pr = E.Element(W + 'pPr'); p.insert(0, pr)
        runprops = p.find('w:r/w:rPr', N)
        for child in list(p):
            if child is not pr:
                p.remove(child)
        for tag in ['sectPr', 'pageBreakBefore', 'numPr', 'keepNext']:
            old = pr.find(W + tag)
            if old is not None:
                pr.remove(old)
        r = E.SubElement(p, W + 'r')
        if runprops is not None:
            r.append(cp(runprops))
        t = E.SubElement(r, W + 't')
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        t.text = value
        return p

    def heading(prefix):
        return next(p for p in body.findall(W + 'p')
                    if text(p).startswith(prefix)
                    and p.find('w:pPr/w:pStyle', N) is not None
                    and p.find('w:pPr/w:pStyle', N).get(W + 'val') in ['2', '3'])

    before_dashes = sum((t.text or '').count('—') for t in root.xpath('//w:t | //m:t', namespaces=N))
    before_voice = sum((t.text or '').count('本研究') for t in root.xpath('//w:t', namespaces=N))
    for t in root.xpath('//w:t | //m:t', namespaces=N):
        if t.text:
            # Mathematical minus (U+2212) is intentionally preserved.
            t.text = re.sub('[—–]+', '-', t.text).replace('本研究', '本文')
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if re.search('[—–]', value):
                element.set(key, re.sub('[—–]+', '-', value))
    for n in [1, 2, 3]:
        p = heading(f'2.{n} 问题')
        body.replace(p, paragraph(f'2.{n} 问题{["一", "二", "三"][n-1]}分析', p))

    flush_count = 0
    for p in body.findall(W + 'p'):
        if text(p).startswith('式中'):
            pr = p.find(W + 'pPr')
            if pr is None:
                pr = E.Element(W + 'pPr'); p.insert(0, pr)
            ind = prop(pr, 'ind', left=0, firstLine=0, firstLineChars=0)
            for key in ['hanging', 'hangingChars', 'leftChars', 'start', 'startChars']:
                ind.attrib.pop(W + key, None)
            flush_count += 1
        if text(p).startswith('证明。'):
            p.append(nm.math(nm.r('□', roman=True)))

    solvers = {
        '5.3 ': '本文用scikit-learn实现基学习器：ExtraTrees采用随机切分建树，HGB采用直方图梯度提升，Ridge采用自动线性代数求解，PLS2采用NIPALS迭代。堆叠调用SciPy主动集NNLS，随后作0.5收缩并恢复自由截距；参数由内层3折选择，外层10×3划分用于评价。',
        '6.2 ': '本文将稀疏与图正则回归转为增广Lasso，用循环坐标下降求解；条件图消融采用GraphicalLasso的坐标下降模式并折内选参，模块用Louvain划分。四路按支持数与倒数秩排序；500次分层自助和留一风格用于检验，最终预测器仅作独立条件归因。',
        '7.2 ': '本文用经验秩变换及Ledoit-Wolf闭式收缩估计依赖结构，经相关正态采样与经验分位反变换生成谱。迁移对固定端点和七个路径比例作有限枚举，施加直接浓度约束后，按可行性与改动量排序；其最优性仅限给定候选集。'
    }
    for prefix, value in solvers.items():
        heading(prefix).addnext(paragraph(value))

    # Compact table12, keeping original numerical evidence in other tables/text.
    cap = next(p for p in body.findall(W + 'p') if text(p).startswith('表12 '))
    old_table = cap.getnext()
    assert old_table.tag == W + 'tbl'
    tbl = E.Element(W + 'tbl', nsmap=old_table.nsmap)
    tbl.append(cp(old_table.find(W + 'tblPr')))
    grid = E.SubElement(tbl, W + 'tblGrid')
    widths = [2450, 2250, 4030]
    for width in widths:
        E.SubElement(grid, W + 'gridCol').set(W + 'w', str(width))
    rows = [
        ['方法与来源', '本文用途', '采用方式与边界'],
        ['相关网络[5]', '描述成对共变', '全量图仅作描述；监督图在训练折重算，相关不等于因果。'],
        ['稀疏条件图[6]', '审计条件依赖', '采用折内GraphicalLasso检验结构，不替换正式四路排序。'],
        ['条件置换与SHAP[7][10]', '解释预测器依赖', '本文采用模块和条件置换；SHAP仅作文献比较。'],
        ['稳定选择[8][9]', '检验重采样稳定性', '频率用于稳定性说明，不援引未满足假设的理论误选保证。'],
        ['四路稳健排序\n（倒数秩项[14]）', '综合四路统计证据', '先比较支持数，再按倒数秩排序；预测归因独立报告。'],
    ]
    for ri, row in enumerate(rows):
        tr = E.SubElement(tbl, W + 'tr')
        trpr = E.SubElement(tr, W + 'trPr')
        E.SubElement(trpr, W + 'cantSplit')
        if ri == 0:
            E.SubElement(trpr, W + 'tblHeader')
        for ci, value in enumerate(row):
            tc = E.SubElement(tr, W + 'tc'); tcp = E.SubElement(tc, W + 'tcPr')
            prop(tcp, 'tcW', w=widths[ci], type='dxa'); prop(tcp, 'vAlign', val='center')
            p = E.SubElement(tc, W + 'p'); pr = E.SubElement(p, W + 'pPr')
            prop(pr, 'pStyle', val='11')
            if ri < len(rows) - 1:
                prop(pr, 'keepNext')
            prop(pr, 'keepLines'); prop(pr, 'snapToGrid', val=0)
            prop(pr, 'spacing', before=0, after=0, line=240, lineRule='auto')
            prop(pr, 'ind', firstLine=0, firstLineChars=0)
            prop(pr, 'jc', val='center' if ri == 0 else 'left')
            r = E.SubElement(p, W + 'r'); rp = E.SubElement(r, W + 'rPr')
            prop(rp, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman', eastAsia='宋体')
            prop(rp, 'b', val=int(ri == 0)); prop(rp, 'bCs', val=int(ri == 0))
            prop(rp, 'sz', val=20); prop(rp, 'szCs', val=20)
            for li, line in enumerate(value.split('\n')):
                if li:
                    E.SubElement(r, W + 'br')
                E.SubElement(r, W + 't').text = line
    body.replace(old_table, tbl)
    body.replace(cap, paragraph('表12 方法家族与本文采用方式', cap))
    # Original RRF source supports the reciprocal-rank component only, not the
    # team's support-count hierarchy or any reported numerical performance.
    rank_para = next(p for p in body.findall(W + 'p') if text(p).startswith('总关联共识、分层自助'))
    for t in rank_para.xpath('.//w:t', namespaces=N):
        if t.text and '的倒数秩分数' in t.text:
            t.text = t.text.replace('的倒数秩分数', '的倒数秩分数[14]')
            break
    last_ref = next(p for p in body.findall(W + 'p') if text(p).startswith('[13] '))
    citation = '[14] Cormack G V, Clarke C L A, Buettcher S. Reciprocal rank fusion outperforms condorcet and individual rank learning methods [C]//Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2009: 758-759. DOI:10.1145/1571941.1572114.'
    last_ref.addnext(paragraph(citation, last_ref))

    strengths = [
        '（1）数据与接口统一。三问使用一致的主键、尺度和折外接口，预测、统计排序与生成检验各有对应证据，便于逐层核对。',
        '（2）兼顾互补性与稳定性。感官组收缩利用四类学习器互补，在保留不利排序指标的同时改善连续解释度；四路排名同时检验样本扰动和风格变化。',
        '（3）结论边界明确。化学核验区分预测依赖与缺陷证据；生成方法联合考察边缘、联合、覆盖和离散，并保留候选未知状态。',
    ]
    weaknesses = [
        '（1）数据与量表受限。样本量及风格不均衡限制弱目标和小风格推断，聚合感官信号无法恢复原始评委评分。',
        '（2）区间缺少独立验证。重复验证共享样本，残差校准为样本内检查；新增小规模重训缺逐样本预测，无法完整分解训练不确定性。',
        '（3）前三项存在替代。变量共线、影子字段入选非零，遮蔽检验不能证明正式答案充分。',
        '（4）生成与迁移尚待实测。画像及迁移依赖预测器，未见风格、独立批次和真实酿造尚未验证；两条候选同源且对微小裕量敏感，不能据此作安全或工艺结论。',
    ]
    for prefix, values in [('8.1 ', strengths), ('8.2 ', weaknesses)]:
        h = heading(prefix); old = h.getnext()
        assert old.tag == W + 'p'
        previous = h
        for value in values:
            p = paragraph(value, old)
            prop(p.find(W + 'pPr'), 'pStyle', val='34')
            prop(p.find(W + 'pPr'), 'ind', left=0, firstLine=0, firstLineChars=0)
            previous.addnext(p); previous = p
        body.remove(old)

    # Five figures contain textual long dashes. Change only those labels in
    # R-rendered copies; their numeric data, plot type and canvas are unchanged.
    rels = {r.get('Id'): r.get('Target') for r in E.fromstring(files['word/_rels/document.xml.rels'])}
    figure_edits = {}
    for prefix, stem in {
        '图2 ': '图2_模型比较', '图11 ': '图11_A级关系',
        '图14 ': '图14_生成质量', '图15 ': '图15_三风格画像',
        '附图S1 ': '附图S1_残差与量表',
    }.items():
        caption = next(x for x in body.findall(W + 'p') if text(x).startswith(prefix))
        drawing = caption.getprevious()
        rid = drawing.find('.//a:blip', N).get('{' + N['r'] + '}embed')
        media = 'word/' + rels[rid]
        source = BASE / '图表_v3_字符统一' / (stem + '.png')
        data = source.read_bytes()
        assert Image.open(BytesIO(data)).size == Image.open(BytesIO(files[media])).size, (prefix, source)
        figure_edits[prefix.strip()] = {
            'media': media, 'path': str(source.relative_to(ROOT)),
            'old_sha256': sha256(files[media]).hexdigest(), 'new_sha256': sha256(data).hexdigest(),
            'scope': 'textual dash glyphs only; numeric plot data unchanged',
        }
        files[media] = data

    # Retain valid OpenXML property ordering without changing mother styles.
    orders = {
        'pPr': 'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange',
        'rPr': 'rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath rPrChange',
    }
    for local, order in orders.items():
        rank = {W + n: i for i, n in enumerate(order.split())}
        for pr in root.findall('.//' + W + local):
            pr[:] = sorted(pr, key=lambda x: rank.get(x.tag, 999))
    files['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    with ZipFile(TARGET, 'w', ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    record = {
        'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
        'output': str(TARGET.relative_to(ROOT)),
        'emdash_occurrences_replaced': before_dashes,
        'voice_occurrences_replaced': before_voice,
        'flush_left_formula_explanations': flush_count,
        'native_QED': 3, 'solver_paragraphs': solvers,
        'table12_rows': rows, 'reference14': citation,
        'reference_source': 'https://doi.org/10.1145/1571941.1572114',
        'original_table12': text(next(x.getnext() for x in source_body if text(x).startswith('表12 '))),
        'strength_points': len(strengths), 'weakness_points': len(weaknesses),
        'technical_roadmap_embedded': False,
        'figure_character_edits': figure_edits,
        'no_training': True, 'no_result_edits': True, 'no_git_commit': True,
    }
    (AUDIT / '修改清单.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({k: record[k] for k in ['output', 'emdash_occurrences_replaced', 'voice_occurrences_replaced', 'flush_left_formula_explanations', 'native_QED', 'strength_points', 'weakness_points']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
