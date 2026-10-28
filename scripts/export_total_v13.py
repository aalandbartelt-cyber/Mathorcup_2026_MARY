"""Update real TOC, keep mother styles and generate the delivered Word PDF."""
from copy import deepcopy as cp
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import shutil
import subprocess
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / '05_论文/总论文'
SOURCE = PAPER / '2026_MathorCup_大数据竞赛_总论文_v12.docx'
FILE = PAPER / '2026_MathorCup_大数据竞赛_总论文_v13.docx'
AUDIT = ROOT / 'docs/collab/20261028_V13修订'
WORK = Path('/mnt/c/Users/Public/mathorcup_v13_20261028')
N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + N['w'] + '}'
WORK.mkdir(parents=True, exist_ok=True)

with ZipFile(FILE) as z:
    xml = z.read('word/document.xml')
    styles_xml = z.read('word/styles.xml')
preflight = E.fromstring(xml)
for field in preflight.xpath('//w:instrText[contains(text(),"TOC ")]', namespaces=N):
    # Outline-only TOC honors the explicit body-text level on code subsections.
    field.text = ' TOC \\o "1-2" \\h \\z '
xml = E.tostring(preflight, encoding='UTF-8', xml_declaration=True, standalone=True)
with ZipFile(SOURCE) as z:
    original_parts = {name: z.read(name) for name in z.namelist()}
parts = dict(original_parts)
parts['word/document.xml'] = xml
parts['word/styles.xml'] = styles_xml

def write_doc():
    with ZipFile(FILE, 'w', ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)

def text(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=N))

write_doc()
script = WORK / 'export.ps1'
script.write_bytes(b'\xef\xbb\xbf' + (ROOT / 'scripts/export_total_paper_word.ps1').read_bytes())
local = WORK / FILE.name
windows = subprocess.check_output(['wslpath', '-w', str(WORK)], text=True).strip()
cmd = ['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile',
       '-ExecutionPolicy', 'Bypass', '-File', windows + '\\export.ps1',
       '-Source', windows + '\\' + FILE.name, '-OutputDir', windows]
shutil.copy2(FILE, local)
subprocess.run(cmd + ['-UpdateToc'], check=True)

root = E.fromstring(parts['word/document.xml'])
body = root.find('w:body', N)
fragment = E.parse(str(WORK / 'toc_fragment.xml')).find('.//w:body', N)
paragraphs = [cp(p) for p in fragment if p.tag == W + 'p']
for p in paragraphs:
    pp = p.find(W + 'pPr')
    if pp is None:
        pp = E.Element(W + 'pPr');p.insert(0, pp)
    ps = pp.find(W + 'pStyle')
    if ps is None:
        ps = E.Element(W + 'pStyle');pp.insert(0, ps)
    ps.set(W + 'val', '1')
    for attribute in list(p.attrib):
        if E.QName(attribute).localname in {'paraId', 'textId'}:
            del p.attrib[attribute]
old = next(p for p in body.findall(W + 'p') if p.xpath('.//w:instrText[contains(text(),"TOC ")]', namespaces=N))
start = body.index(old)
stop = next(i for i in range(start + 1, len(body)) if body[i].find('w:pPr/w:sectPr', N) is not None)
for p in list(body)[start:stop]:
    body.remove(p)
for p in reversed(paragraphs):
    body.insert(start, p)

record = json.loads((WORK / 'word_pagination.json').read_text(encoding='utf-8-sig'))
old_ids = {x.get(W + 'id') for x in root.findall('.//' + W + 'bookmarkStart') if x.get(W + 'name', '').startswith('_Toc')}
for x in list(root.findall('.//' + W + 'bookmarkStart')) + list(root.findall('.//' + W + 'bookmarkEnd')):
    if x.get(W + 'id') in old_ids:
        x.getparent().remove(x)
next_id = 15000
for heading in record['headings']:
    matches = [p for p in body.findall(W + 'p') if text(p) == heading['text']]
    if len(matches) != 1:
        # Auto-numbered chapter paragraphs have InnerText excluding ListString.
        continue
    p = matches[0]
    hxml = E.fromstring(heading['xml'].encode())
    for name in hxml.xpath('.//w:bookmarkStart[starts-with(@w:name,"_Toc")]/@w:name', namespaces=N):
        a = E.Element(W + 'bookmarkStart');a.set(W + 'id', str(next_id));a.set(W + 'name', name)
        b = E.Element(W + 'bookmarkEnd');b.set(W + 'id', str(next_id))
        p.insert(1, a);p.append(b);next_id += 1
parts['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
write_doc()
shutil.copy2(WORK / 'toc_fragment.xml', AUDIT / '目录缓存.xml')
shutil.copy2(WORK / 'word_pagination.json', AUDIT / '目录更新分页.json')
shutil.copy2(FILE, local)
subprocess.run(cmd, check=True)
shutil.copy2(local.with_suffix('.pdf'), FILE.with_suffix('.pdf'))
shutil.copy2(WORK / 'word_pagination.json', AUDIT / 'Word分页.json')

destination = Path('/mnt/d/上海大学__/26年妈妈杯数模/终稿/论文/A2601605.pdf')
if destination.exists():
    backup = AUDIT / '原A2601605.pdf'
    index = 2
    while backup.exists():
        backup = AUDIT / f'原A2601605_构建{index}.pdf'
        index += 1
    shutil.copy2(destination, backup)
shutil.copy2(FILE.with_suffix('.pdf'), destination)
print(FILE.with_suffix('.pdf'))
print(destination)
