"""One-page, stand-alone title/abstract/keywords proof, pending user approval.

Never writes the main-paper DOCX. Uses its existing mother styles and page box.
"""
from copy import deepcopy as cp
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import re

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文'
OUT = BASE / '首页_待审计'
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + N['w'] + '}'


def prop(parent, tag, **attrs):
    x = parent.find(W + tag)
    if x is None:
        x = E.SubElement(parent, W + tag)
    for k, v in attrs.items():
        x.set(W + k, str(v))
    return x


def main():
    data = json.loads((OUT / '题目摘要关键词_候选_v1.json').read_text())
    source = BASE / '2026_MathorCup_大数据竞赛_总论文_v5.docx'
    with ZipFile(source) as z:
        files = {n: z.read(n) for n in z.namelist()}
    root = E.fromstring(files['word/document.xml']); body = root.find(W + 'body')
    old = list(body)
    section = cp(root.find('.//w:sectPr', N))
    title, abstract_heading, paragraph, keywords = [cp(old[i]) for i in range(4)]
    for x in list(body):
        body.remove(x)

    def new_paragraph(value, mother, role='body'):
        p = cp(mother); pr = p.find(W + 'pPr')
        for x in list(p):
            if x is not pr:
                p.remove(x)
        for tag in ['numPr', 'pageBreakBefore', 'sectPr', 'keepNext']:
            x = pr.find(W + tag)
            if x is not None:
                pr.remove(x)
        if role == 'body':
            prop(pr, 'pStyle', val='11')
            prop(pr, 'spacing', before=0, after=0)
            prop(pr, 'ind', firstLine=480, firstLineChars=200)
        if role == 'keywords':
            prop(pr, 'pStyle', val='11')
            prop(pr, 'ind', firstLine=0, firstLineChars=0)
        if role == 'title':
            prop(pr, 'keepNext')
        bold = False
        for chunk in re.split(r'(\*\*)', value):
            if not chunk:
                continue
            if chunk == '**':
                bold = not bold; continue
            r = E.SubElement(p, W + 'r'); rp = E.SubElement(r, W + 'rPr')
            prop(rp, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman',
                 eastAsia='黑体' if role in ['title', 'heading'] else '宋体')
            prop(rp, 'b', val=int(bold or role in ['title', 'heading']))
            prop(rp, 'bCs', val=int(bold or role in ['title', 'heading']))
            size = 32 if role == 'title' else 28 if role == 'heading' else 24
            prop(rp, 'sz', val=size); prop(rp, 'szCs', val=size)
            t = E.SubElement(r, W + 't'); t.text = chunk
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        order = 'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange'
        rank = {W + tag: i for i, tag in enumerate(order.split())}
        pr[:] = sorted(pr, key=lambda x: rank.get(x.tag, 999))
        return p

    body.append(new_paragraph(data['title'], title, 'title'))
    body.append(new_paragraph('摘 要', abstract_heading, 'heading'))
    for s in [data['introduction']] + [q['model'] + q['solve_result'] for q in data['questions']] + [data['closing']]:
        body.append(new_paragraph(s, paragraph))
    body.append(new_paragraph('**关键词：**' + '；'.join(data['keywords']), keywords, 'keywords'))
    body.append(section)
    files['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    output = OUT / '首页排版预览_待审_v1.docx'
    with ZipFile(output, 'w', ZIP_DEFLATED) as z:
        for n, raw in files.items():
            z.writestr(n, raw)
    print(output)


if __name__ == '__main__':
    main()
