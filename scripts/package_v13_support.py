"""Package the user's unchanged flat delivery folder and verify actual paths."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import ast
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/collab/20261028_V13修订'
SOURCE = Path('/mnt/d/上海大学__/26年妈妈杯数模/终稿/A2601605附件')
DESTINATIONS = [SOURCE.with_suffix('.zip'), ROOT / '06_交付/A2601605附件.zip']
mapping = json.loads((AUDIT / '附录14节_真实源范围.json').read_text())
files = sorted(p for p in SOURCE.rglob('*') if p.is_file())
assert len(files) == 181, len(files)
staging = AUDIT / 'A2601605附件.zip'
with ZipFile(staging, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
    for path in files:
        archive.write(path, SOURCE.name + '/' + path.relative_to(SOURCE).as_posix())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

commands = []
with tempfile.TemporaryDirectory(prefix='mcup_v13_unpack_') as directory:
    unpack = Path(directory)
    with ZipFile(staging) as archive:
        assert len(archive.infolist()) == 181 and archive.testzip() is None
        archive.extractall(unpack)
        for source in files:
            member = SOURCE.name + '/' + source.relative_to(SOURCE).as_posix()
            assert hashlib.sha256(archive.read(member)).hexdigest() == digest(source)
    work = unpack / SOURCE.name
    paths = sorted(p for p in (work / 'A2601605源程序').iterdir() if p.is_file())
    assert len(paths) == 30
    for path in paths:
        assert path.suffix.lower() in {'.py', '.r'}
    assert (work / 'A2601605软件与运行说明.txt').is_file()
    for block in mapping['sections']:
        relative = block['archive_relative_path']
        assert (work / relative).is_file()
        environment = dict(os.environ, PYTHONPYCACHEPREFIX=str(unpack / 'bytecode'))
        process = subprocess.run([
            '/home/mary/miniconda3/envs/mathorcup/bin/python', '-m', 'py_compile', relative
        ], cwd=work, env=environment, capture_output=True, text=True)
        assert process.returncode == 0, (relative, process.stderr)
        commands.append({'section': block['section'], 'command': block['command'], 'exit_code': process.returncode})
    for path in paths:
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'))
for destination in DESTINATIONS:
    if destination.exists() and digest(destination) != digest(staging):
        backup = AUDIT / (destination.parent.name + '_旧附件.zip')
        assert not backup.exists()
        shutil.copy2(destination, backup)
    shutil.copy2(staging, destination)
    assert digest(destination) == digest(staging)
result = {
    'files': 181, 'program_files': 30, 'appendix_A2_real_paths': 31,
    'commands_run': commands, 'command_count': 14, 'distinct_python_files': 12,
    'bytes': staging.stat().st_size, 'sha256': digest(staging),
    'destinations': [str(p) for p in DESTINATIONS],
    'AI_details_PDF': 'absent; explicitly marked pending in manuscript; not fabricated',
    'source_files_changed': False, 'training_executed': False,
    'residual_path_limit': 'Existing TXT and plotting/default input paths were not rewritten, as source package is frozen by this task.'
}
(AUDIT / '附件打包与14命令检查.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'commands_run'}, ensure_ascii=False, indent=2))
