"""Keep the SDK-generated body while preserving every other V8 package byte."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_论文/总论文/2026_MathorCup_大数据竞赛_总论文_v8.docx'
OUTPUT = BASE.with_name(BASE.name.replace('_v8', '_v9'))


def main():
    with ZipFile(OUTPUT) as edited:
        document = edited.read('word/document.xml')
    fd, temporary = tempfile.mkstemp(prefix='appendix_a_', suffix='.docx', dir=OUTPUT.parent)
    os.close(fd)
    temp_path = Path(temporary)
    with ZipFile(BASE) as base, ZipFile(temp_path, 'w', ZIP_DEFLATED) as output:
        for info in base.infolist():
            output.writestr(info, document if info.filename == 'word/document.xml' else base.read(info.filename))
    temp_path.replace(OUTPUT)
    with ZipFile(BASE) as base, ZipFile(OUTPUT) as edited:
        changed = [n for n in base.namelist() if base.read(n) != edited.read(n)]
    assert changed == ['word/document.xml'], changed
    print('Changed package parts:', changed)


if __name__ == '__main__':
    main()
