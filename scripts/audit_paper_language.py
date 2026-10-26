"""Scan publication text, filenames and nested archives for internal terminology."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile, is_zipfile
import argparse
import gzip
import json
import re
from lxml import etree

PATTERN=re.compile(r'(?<![A-Za-z0-9])E\d{3}(?![A-Za-z0-9])|参考组|对照组|晋级|门槛|promotion|\bpass\b|BZD|agent(?:2|3)?|学长版',re.I)
TEXT_EXT={'.md','.txt','.csv','.json','.yaml','.yml','.py','.r','.tex','.xml','.svg','.html','.bib','.toml','.ini'}

def text_hits(text,location):
    return [{'location':location,'term':m.group(0),'context':text[max(0,m.start()-35):m.end()+45]} for m in PATTERN.finditer(text)]

def scan_blob(name,data,records):
    records['files'].append(name)
    records['hits']+=text_hits(name,name+' [filename]')
    ext=Path(name).suffix.lower()
    if ext in {'.docx','.zip'}:
        with ZipFile(BytesIO(data)) as z:
            for n in z.namelist():
                if n.endswith('/'):continue
                blob=z.read(n);location=name+'!'+n
                if ext=='.docx' and (n.endswith('.xml') or n.endswith('.rels')):
                    tree=etree.fromstring(blob)
                    records['files'].append(location)
                    records['hits']+=text_hits(n,location+' [filename]')
                    # Join text across Word runs, including formulas, notes, captions and metadata.
                    records['hits']+=text_hits(''.join(tree.itertext()),location)
                    for node in tree.iter():
                        for key,value in node.attrib.items():
                            records['hits']+=text_hits(value,location+' [attribute]')
                elif ext=='.zip':scan_blob(location,blob,records)
                elif n.startswith('word/media/'):
                    records['images'].append(location)
    elif ext=='.gz':scan_blob(name[:-3],gzip.decompress(data),records)
    elif ext=='.pdf':
        import pymupdf
        with pymupdf.open(stream=data,filetype='pdf') as pdf:
            for i,page in enumerate(pdf):records['hits']+=text_hits(page.get_text(),f'{name} [page {i+1}]')
            records['hits']+=text_hits(json.dumps(pdf.metadata,ensure_ascii=False),name+' [metadata]')
    elif ext in TEXT_EXT or Path(name).name in {'SHA256SUMS','LICENSE','.gitkeep'}:
        text=data.decode('utf-8-sig')
        records['hits']+=text_hits(text,name)
    elif ext=='.png':
        from PIL import Image
        im=Image.open(BytesIO(data))
        records['hits']+=text_hits(json.dumps(im.info,default=str,ensure_ascii=False),name+' [metadata]')
        records['images'].append(name)
    elif ext in {'.npz','.npy'}:
        # Machine-readable numeric outputs are checked separately for identity and provenance.
        records['numeric_archives'].append(name)
    else:records['unhandled'].append(name)

def audit(paths):
    records={'files':[],'hits':[],'images':[],'numeric_archives':[],'unhandled':[]}
    for path in paths:
        p=Path(path)
        if not p.exists():raise FileNotFoundError(p)
        files=sorted(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else [p]
        for f in files:scan_blob(str(f),f.read_bytes(),records)
    records['scope']=[str(p) for p in paths]
    records['hit_count']=len(records['hits'])
    records['image_text_method']='PNG metadata scan; visible labels checked from plotting source/CSV and visual rendering, no OCR claim'
    return records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('paths',nargs='+');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.paths)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'hit_count':result['hit_count'],'entries':len(result['files']),'unhandled':result['unhandled'],'report':str(args.output)},ensure_ascii=False))
    raise SystemExit(1 if result['hits'] or result['unhandled'] else 0)

if __name__=='__main__':main()
