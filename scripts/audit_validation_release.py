"""Final checks of the three dated review handoffs. Never trains a model."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import hashlib
import json
import re
import subprocess

from lxml import etree
from PIL import Image, ImageChops
import pandas as pd
import pymupdf

from audit_paper_language import audit

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'docs/collab/20261027_模型检验独立验收'
STAGE=ROOT/'05_论文/交付给论文手/待审_20261027'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
PROFILES=[
 ('Q1_感官评分',ROOT/'05_论文/Q1_感官评分/第一问_论文正文_v5.docx'),
 ('Q2_关键化合物',ROOT/'.worktrees/q2-evidence-opt/05_论文/Q2_关键化合物/第二问_论文正文_v4.docx'),
 ('Q3_风格生成',ROOT/'.worktrees/q3-foundation/05_论文/Q3_风格生成/第三问_论文正文_v2.docx')]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    results=[]; scope=[STAGE]
    cli=r'\\wsl.localhost\Ubuntu-22.04\home\mary\.codex\skills\software-copyright-materials\vendor\docx-toolkit\scripts\dotnet\DocxToolkit.Cli\bin\Debug\net8.0\DocxToolkit.Cli.dll'
    sdk=[]
    for q,base in PROFILES:
        folder=ROOT/'05_论文'/q/'待审_20261027'; figs=ROOT/'04_图表'/q/'待审_20261027'
        scope += [folder,figs]
        main_doc=next(folder.glob('第*正文*.docx'))
        for p in folder.glob('*.docx'):
            win=r'\\wsl.localhost\Ubuntu-22.04'+str(p).replace('/','\\')
            command=['/mnt/c/Program Files/dotnet/dotnet.exe',cli,'validate','--input',win,'--json']
            completed=subprocess.run(command,check=True,capture_output=True)
            record=json.loads(completed.stdout.decode('utf-8-sig'));assert record['isValid'],record
            sdk.append({'path':str(p.relative_to(ROOT)),**record})
        with ZipFile(main_doc) as z, ZipFile(base) as b:
            xml=etree.fromstring(z.read('word/document.xml'))
            equations=xml.xpath('.//m:oMath',namespaces=NS)
            empty=xml.xpath('.//m:oMath//m:r[not(m:t) or not(normalize-space(m:t))]',namespaces=NS)
            assert not empty
            unchanged_style=all(z.read(n)==b.read(n) for n in ['word/styles.xml','word/settings.xml','word/fontTable.xml','word/theme/theme1.xml'])
            assert unchanged_style
            tables=xml.xpath('.//w:tbl',namespaces=NS)
            math_tables=[t for t in tables if t.xpath('.//m:oMath',namespaces=NS)]
            assert all(len(t.findall('w:tr',NS))==1 and len(t.find('w:tr',NS).findall('w:tc',NS))==3 for t in math_tables)
            rels={e.get('Id'):e.get('Target') for e in etree.fromstring(z.read('word/_rels/document.xml.rels'))}
            drawings=xml.xpath('.//w:drawing',namespaces=NS)
            captions={}
            for p in xml.xpath('./w:body/w:p',namespaces=NS):
                t=''.join(p.xpath('.//w:t/text()',namespaces=NS)); m=re.match(r'(附图S\d|图\d)\s+(.+)',t)
                if m:
                    assert m.group(1) not in captions
                    captions[m.group(1)]=m.group(2)
            manifest=pd.read_csv(STAGE/q/'图片'/f'{q[:2]}_图清单.csv')
            assert len(manifest)==len(drawings)
            for row in manifest.itertuples(index=False,name=None):
                assert row[2]==captions[row[0]],row
            for draw in drawings:
                rid=draw.find('.//a:blip',NS).get('{'+NS['r']+'}embed')
                image=Image.open(BytesIO(z.read('word/'+rels[rid]))).convert('RGB')
                matches=[]
                for p in (STAGE/q/'图片').glob('*.png'):
                    other=Image.open(p).convert('RGB')
                    if image.size==other.size and ImageChops.difference(image,other).getbbox() is None:matches.append(p)
                assert len(matches)==1,matches
        with pymupdf.open(main_doc.with_suffix('.pdf')) as pdf:
            pages=len(pdf); fonts={f[3] for page in pdf for f in page.get_fonts()}
            assert any('CambriaMath' in f for f in fonts)
        package=next(STAGE.glob(q+'_交付包*.zip'))
        syntax=0
        with ZipFile(package) as z:
            assert z.testzip() is None
            sums=z.read(q+'/SHA256SUMS.txt').decode().splitlines()
            for line in sums:
                h,n=line.split('  ',1);data=z.read(q+'/'+n)
                assert hashlib.sha256(data).hexdigest()==h,n
                assert data==(STAGE/q/n).read_bytes(),n
            for n in z.namelist():
                if n.endswith('.py'):compile(z.read(n),n,'exec');syntax+=1
        verification=subprocess.run([
            '/home/mary/miniconda3/envs/mathorcup/bin/python',str(STAGE/q/'程序/核验交付.py'),
            '--root',str(STAGE/q)],check=True,capture_output=True,text=True)
        row={'question':q,'pages':pages,'figures':len(drawings),'content_tables':len(tables)-len(math_tables),
             'formulas':len(equations),'empty_math_runs':0,'mother_style_identity':unchanged_style,
             'archive':str(package.relative_to(ROOT)),'archive_files':len(sums)+1,
             'archive_bytes':package.stat().st_size,'archive_sha256':digest(package),
             'python_syntax_checked':syntax,'provided_verifier':json.loads(verification.stdout)}
        results.append(row);print(q,pages,'pages;',len(sums)+1,'files; structural/data checks complete',flush=True)
    pagination=json.loads((AUDIT/'word_pagination.json').read_text(encoding='utf-8-sig'))
    for d in pagination:
        assert all(t['start_page']==t['end_page'] for t in d['tables']),d['source']
        image_caps=[c for c in d['captions'] if c['text'].startswith(('图','附图'))]
        assert [c['page'] for c in image_caps]==[i['page'] for i in d['images']],d['source']
    old=json.loads((AUDIT/'existing_document_snapshot.json').read_text())
    assert all(digest(ROOT/r['path'])==r['sha256'] for r in old)
    language=audit(scope)
    (AUDIT/'publication_language_all.json').write_text(json.dumps(language,ensure_ascii=False,indent=2)+'\n')
    assert language['hit_count']==0 and not language['unhandled'],language['hits']
    (AUDIT/'openxml_sdk_validation.json').write_text(json.dumps(sdk,ensure_ascii=False,indent=2)+'\n')
    report={'status':'ready_for_user_review','manuscripts':results,'sdk_documents':len(sdk),
            'language_scanned_entries':len(language['files']),'language_hits':0,'empty_math_runs':0,
            'original_documents_packages_preserved':len(old),'split_tables':0,'separated_figure_captions':0}
    (AUDIT/'final_release_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='manuscripts'},ensure_ascii=False))


if __name__=='__main__':main()
