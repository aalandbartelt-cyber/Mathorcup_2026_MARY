"""Export dated review DOCX files through the installed Windows Word."""
from pathlib import Path
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'docs/collab/20261027_模型检验独立验收'
WORK=Path('/mnt/c/Users/Public/mathorcup_validation_20261027')
WORK.mkdir(parents=True,exist_ok=True)
jobs=[]; originals=[]
for q in ['Q1_感官评分','Q2_关键化合物','Q3_风格生成']:
    for p in sorted((ROOT/'05_论文'/q/'待审_20261027').glob('*.docx')):
        dest=WORK/p.name;shutil.copy2(p,dest)
        jobs.append({'source':'C:\\Users\\Public\\mathorcup_validation_20261027\\'+p.name})
        originals.append((p,dest))
job=WORK/'jobs.json'; job.write_text(json.dumps(jobs,ensure_ascii=False),encoding='utf-8-sig')
script=WORK/'export.ps1'
# Windows PowerShell 5 interprets a BOM-less UTF-8 script as an ANSI code page.
script.write_bytes(b'\xef\xbb\xbf'+(ROOT/'scripts/export_validation_word.ps1').read_bytes())
command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
         'C:\\Users\\Public\\mathorcup_validation_20261027\\export.ps1',
         '-JobFile','C:\\Users\\Public\\mathorcup_validation_20261027\\jobs.json']
subprocess.run(command,check=True)
for original,local in originals:
    pdf=local.with_suffix('.pdf')
    assert pdf.exists() and pdf.stat().st_size>1000,pdf
    shutil.copy2(pdf,original.with_suffix('.pdf'))
shutil.copy2(WORK/'word_pagination.json',AUDIT/'word_pagination.json')
print('Copied',len(originals),'Word-native PDFs into review directories.')
