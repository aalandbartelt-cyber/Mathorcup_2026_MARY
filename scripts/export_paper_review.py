"""Export current Word deliverables and one bookmarked PDF review volume."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import re
import pymupdf

ROOT=Path(__file__).resolve().parents[1]

def latest(folder,pattern):
    files=list(folder.glob(pattern))
    if not files:raise FileNotFoundError(pattern)
    return max(files,key=lambda p:int(re.search(r'_v(\d+)\.docx$',p.name).group(1)))

def windows(path):
    return subprocess.check_output(['wslpath','-w',str(path.resolve())],text=True).strip()

def export(question):
    folder=ROOT/'05_论文'/question
    n={'Q1':'第一问','Q2':'第二问','Q3':'第三问'}[question.split('_')[0]]
    files=[latest(folder,n+'_论文正文_v*.docx')]
    for pattern in ['模型假设*_v*.docx','问题分析*_v*.docx']:
        if list(folder.glob(pattern)):files.append(latest(folder,pattern))
    command=shutil.which('libreoffice') or shutil.which('soffice')
    native=bool(command)
    if not command:
        command='/mnt/c/Program Files/LibreOffice/program/soffice.exe'
        if not Path(command).is_file():raise RuntimeError('LibreOffice未找到，不能省略PDF交付')
    with tempfile.TemporaryDirectory(prefix='paper_pdf_') as temp:
        stage=Path(temp)
        if native:
            cmd=[command,'-env:UserInstallation='+stage.as_uri()+'/profile','--headless','--convert-to','pdf','--outdir',str(stage),*[str(p) for p in files]]
        else:
            cmd=[command,'-env:UserInstallation=file:///C:/Users/mary3/AppData/Local/Temp/mcup_paper_pdf_export','--headless','--convert-to','pdf','--outdir',windows(stage),*[windows(p) for p in files]]
        process=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        if process.returncode:raise RuntimeError(process.stdout+process.stderr)
        merged=pymupdf.open();bookmarks=[];manifest=[]
        for source in files:
            pdf=stage/source.with_suffix('.pdf').name
            if not pdf.is_file():raise FileNotFoundError(pdf)
            dest=folder/pdf.name;shutil.copy2(pdf,dest)
            with pymupdf.open(pdf) as part:
                offset=len(merged);bookmarks.append([1,source.stem,offset+1])
                bookmarks.extend([[min(level+1,6),text,page+offset] for level,text,page in part.get_toc()])
                merged.insert_pdf(part)
                manifest.append({'word':str(source.relative_to(ROOT)),'word_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pdf':str(dest.relative_to(ROOT)),'pdf_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'pages':len(part),'page_width_points':part[0].rect.width,'page_height_points':part[0].rect.height})
        version=re.search(r'_v\d+$',files[0].stem).group(0)
        out=folder/(question.split('_')[0]+'_论文审计合订本'+version+'.pdf')
        merged.set_toc(bookmarks)
        merged.set_metadata({'title':n+'论文审计合订本','author':'','subject':'正文、模型假设与问题分析','creator':'论文审计导出','producer':'PyMuPDF'})
        merged.save(out,garbage=4,deflate=True);merged.close()
    evidence=ROOT/'03_结果'/question/'publication_audit';evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'pdf_export_manifest.json').write_text(json.dumps({'documents':manifest,'review_pdf':str(out.relative_to(ROOT)),'review_sha256':hashlib.sha256(out.read_bytes()).hexdigest()},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'review_pdf':str(out),'documents':len(files),'pages':sum(x['pages'] for x in manifest)},ensure_ascii=False))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--question',required=True);a=p.parse_args();export(a.question)
