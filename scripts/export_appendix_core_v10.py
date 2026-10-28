"""Render the new appendix edition in Word without rewriting DOCX bytes."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / '05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v10.docx'
AUDIT = ROOT / 'docs/collab/20261028_总论文v10核心代码'
WORK = Path('/mnt/c/Users/Public/mathorcup_appendix_v10_20261028')

WORK.mkdir(parents=True, exist_ok=True)
script = WORK / 'export.ps1'
script.write_bytes(b'\xef\xbb\xbf' + (ROOT / 'scripts/export_total_paper_word.ps1').read_bytes())
local = WORK / DOCX.name
shutil.copy2(DOCX, local)
windows = subprocess.check_output(['wslpath', '-w', str(WORK)], text=True).strip()
subprocess.run([
    '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile',
    '-ExecutionPolicy', 'Bypass', '-File', windows + '\\export.ps1',
    '-Source', windows + '\\' + DOCX.name, '-OutputDir', windows,
], check=True)
shutil.copy2(local.with_suffix('.pdf'), DOCX.with_suffix('.pdf'))
shutil.copy2(WORK / 'word_pagination.json', AUDIT / 'word_pagination.json')
print(DOCX.with_suffix('.pdf'))
