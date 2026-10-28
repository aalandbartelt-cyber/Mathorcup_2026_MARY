"""Embed the user-approved roadmap in a v3 copy; preserve frozen evidence.

Only prose linking the three questions, main-figure numbering and image metadata
change. The original formula/table/media parts are retained byte for byte.
"""
from copy import deepcopy as cp
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import json
import re

from lxml import etree as E
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文'
AUDIT = ROOT / 'docs/collab/20261027_总论文v4嵌图'
SOURCE = BASE / '2026_MathorCup_大数据竞赛_总论文_v3.docx'
TARGET = BASE / '2026_MathorCup_大数据竞赛_总论文_v4.docx'
FIGURE = BASE / '图稿_待审计/技术路线/技术路线图_待审.png'
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
     'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
     'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
     'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
     'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture'}
W = '{' + N['w'] + '}'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
CAPTION = '图1 三问技术路线与感官预测接口'
INTRO = ('三问技术路线如图1所示。本文在统一数据口径下衔接感官预测、关键成分识别及风格生成迁移。'
         '第二问的正式Top-3由四路排序确定，NNLS归因仅作独立解释；该排序不作为第三问优化输入。'
         '第三问调用同一预测器形成画像并检验代理约束，0-6七档仅表示相对强度。')


def text(x):
    return ''.join(x.xpath('.//w:t/text() | .//m:t/text()', namespaces=N))


def prop(pr, tag, **attrs):
    x = pr.find(W + tag)
    if x is None:
        x = E.SubElement(pr, W + tag)
    for k, v in attrs.items():
        x.set(W + k, str(v))
    return x


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    snapshot = AUDIT / '旧版本保护快照.json'
    if not snapshot.exists():
        snapshot.write_text(json.dumps({str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                                       for p in BASE.iterdir() if p.is_file() and
                                       (re.search(r'_v[123]\.', p.name) or p.name.startswith('SHA256SUMS'))},
                                      ensure_ascii=False, indent=2))
    with ZipFile(SOURCE) as z:
        files = {n: z.read(n) for n in z.namelist()}
    root = E.fromstring(files['word/document.xml']); body = root.find(W + 'body')
    rels = E.fromstring(files['word/_rels/document.xml.rels'])
    # One-pass substitution also covers appendix references to main figures.
    # Supplemental labels such as 附图S1 are untouched.
    replacements = []
    for t in root.xpath('//w:t', namespaces=N):
        if t.text:
            before = t.text
            t.text = re.sub(r'(?<!附)图\s*(\d+)(?!\d)', lambda m: '图' + str(int(m[1]) + 1), before)
            if before != t.text:
                replacements.append({'before': before, 'after': t.text})
    # These short headings already contain explicit chapter.section numbers.
    # Disable the inherited template list, preventing "1.1 2.1" duplicates.
    for p in body.findall(W + 'p'):
        if text(p) in ['2.1 问题一分析', '2.2 问题二分析', '2.3 问题三分析']:
            pr = p.find(W + 'pPr')
            num = prop(pr, 'numPr'); prop(num, 'numId', val=0)
    oldcap = next(p for p in body.findall(W + 'p') if text(p).startswith('图2 '))
    oldimage = oldcap.getprevious()
    assert oldimage.find('.//w:drawing', N) is not None
    anchor = next(p for p in body.findall(W + 'p') if text(p) == '2.3 问题三分析'
                  and p.find('w:pPr/w:pStyle', N).get(W + 'val') == '3')
    analysis = anchor.getnext()
    assert text(analysis).startswith('风格生成需要')

    def paragraph(value, template):
        p = cp(template); pr = p.find(W + 'pPr')
        rp = p.find('w:r/w:rPr', N)
        for x in list(p):
            if x is not pr:
                p.remove(x)
        for tag in ['sectPr', 'numPr', 'pageBreakBefore', 'keepNext']:
            x = pr.find(W + tag)
            if x is not None:
                pr.remove(x)
        r = E.SubElement(p, W + 'r')
        if rp is not None:
            r.append(cp(rp))
        t = E.SubElement(r, W + 't'); t.text = value
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        return p

    intro = paragraph(INTRO, analysis)
    # Keep the image with its caption. The preceding prose may end on the prior
    # page when necessary, avoiding a large artificial block of white space.
    image = cp(oldimage)
    caption = paragraph(CAPTION, oldcap)
    prop(caption.find(W + 'pPr'), 'ind', firstLine=0, firstLineChars=0)
    rid = 'rIdApprovedTechnicalRoute'
    assert not any(r.get('Id') == rid for r in rels)
    media = 'media/approved_technical_route.png'
    E.SubElement(rels, '{' + REL + '}Relationship', Id=rid,
                 Type=N['r'] + '/image', Target=media)
    raw = FIGURE.read_bytes(); files['word/' + media] = raw
    image.find('.//a:blip', N).set('{' + N['r'] + '}embed', rid)
    width, height = Image.open(BytesIO(raw)).size
    cx = round(15.25 / 2.54 * 914400); cy = round(cx * height / width)
    for x in image.xpath('.//wp:extent | .//a:xfrm/a:ext', namespaces=N):
        x.set('cx', str(cx)); x.set('cy', str(cy))
    nextid = 1 + max(int(x.get('id', 0)) for x in root.xpath('//wp:docPr | //pic:cNvPr', namespaces=N))
    for x in image.xpath('.//wp:docPr | .//pic:cNvPr', namespaces=N):
        x.set('id', str(nextid)); x.set('name', '图1'); x.set('descr', CAPTION)
    previous = analysis
    for x in [intro, image, caption]:
        previous.addnext(x); previous = x
    # The source has no figure SEQ fields. Sync drawing names/alt text directly
    # with captions, rather than incrementing legacy template metadata.
    relmap = {r.get('Id'): r.get('Target') for r in rels}
    figures = []
    for p in body.findall(W + 'p'):
        value = text(p)
        if not re.match(r'^(?:图\d+|附图S\d+)\s', value):
            continue
        drawing = p.getprevious(); blip = drawing.find('.//a:blip', N)
        assert blip is not None, value
        for x in drawing.xpath('.//wp:docPr | .//pic:cNvPr', namespaces=N):
            x.set('name', value.split(' ', 1)[0]); x.set('descr', value)
        part = 'word/' + relmap[blip.get('{' + N['r'] + '}embed')]
        figures.append({'图号': value.split(' ', 1)[0], '图题': value, '包内图片': part,
                        'SHA256': sha256(files[part]).hexdigest(),
                        '来源': str(FIGURE.relative_to(ROOT)) if value == CAPTION else '总论文v3既有图，字节不变'})
    # Ensure new paragraph properties follow the mother schema order.
    order = 'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange'
    rank = {W + tag: i for i, tag in enumerate(order.split())}
    for p in body.findall(W + 'p'):
        pr = p.find(W + 'pPr')
        if pr is not None:
            pr[:] = sorted(pr, key=lambda x: rank.get(x.tag, 999))
    files['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    files['word/_rels/document.xml.rels'] = E.tostring(rels, encoding='UTF-8', xml_declaration=True, standalone=True)
    with ZipFile(TARGET, 'w', ZIP_DEFLATED) as z:
        for n, data in files.items():
            z.writestr(n, data)
    with (BASE / '总论文_v4_图清单.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(figures[0])); writer.writeheader(); writer.writerows(figures)
    record = {'source': str(SOURCE.relative_to(ROOT)), 'output': str(TARGET.relative_to(ROOT)),
              'figure': str(FIGURE.relative_to(ROOT)), 'figure_sha256': sha256(raw).hexdigest(),
              'user_approval': '没问题 可以插入了 然后 注意同步更新文档内容',
              'placement': '2.3问题三分析末尾，第3章模型假设之前', 'caption': CAPTION,
              'added_prose': INTRO, 'main_figure_map': {i: i + 1 for i in range(1, 18)},
              'reference_replacements': replacements, 'old_main_figures': 17, 'new_main_figures': 18,
              'unchanged_supplements': 3, 'image_width_cm': 15.25,
              'heading_numbering_repair': '2.1/2.2/2.3 disable inherited automatic list; explicit numbers retained',
              'old_media_unchanged': True, 'model_result_edits': False, 'git_commit': False}
    (AUDIT / '嵌图清单.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({'output': record['output'], 'renumbered_text_nodes': len(replacements),
                      'main_figures': 18, 'supplements': 3, 'new_image_pixels': [width, height]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
