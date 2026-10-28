"""Repair submission-only paths and verify a fresh archive without fitting or plotting."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import ast
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/collab/20261028_V13修订/P0-4修复'
SOURCE = Path('/mnt/d/上海大学__/26年妈妈杯数模/终稿/A2601605附件')
DESTINATIONS = [SOURCE.with_suffix('.zip'), ROOT / '06_交付/A2601605附件.zip']
TXT = SOURCE / 'A2601605软件与运行说明.txt'
THEME = SOURCE / 'A2601605源程序/绘图主题.R'
CONDA = '/home/mary/miniconda3/bin/conda'
PYTHON = '/home/mary/miniconda3/envs/mathorcup/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): digest(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / '05_论文/总论文' / f'2026_MathorCup_大数据竞赛_总论文_v{v}.{s}'
                 for v in (12, 13) for s in ('docx', 'pdf')]
    protected += [SOURCE.parent / '论文/A2601605.pdf']
    if (AUDIT / '修复前.json').exists():
        before = json.loads((AUDIT / '修复前.json').read_text())
    else:
        before = {'support': inventory(SOURCE), 'delivery': inventory(ROOT / '06_交付'),
                  'papers': {str(p): digest(p) for p in protected},
                  'archives': {str(p): digest(p) for p in DESTINATIONS}}
        assert len(before['support']) == 181
        (AUDIT / '修复前.json').write_text(json.dumps(before, ensure_ascii=False, indent=2))
        for path in (TXT, THEME):
            shutil.copy2(path, AUDIT / ('修复前_' + path.name))
        for index, path in enumerate(DESTINATIONS):
            shutil.copy2(path, AUDIT / f'修复前_附件_{index + 1}.zip')

    # Preserve line endings and all unrelated content in the actual submission source.
    content = (AUDIT / ('修复前_' + TXT.name)).read_bytes()
    assert content.count(b'01_') == 5
    content = content.replace('01_源程序/'.encode(), 'A2601605源程序/'.encode())
    content = content.replace('运行位置：04_支撑材料根目录。'.encode(),
                              '运行位置：解压后的A2601605附件根目录（含A2601605源程序等四个目录及本说明文件）。'.encode())
    content = content.replace('在支撑材料根目录运行：'.encode(), '在A2601605附件根目录运行：'.encode())
    content = content.replace('--attachments 官方附件目录 --execute'.encode(),
                              '--attachments "官方附件目录" --assets "A2601605数据_不含赛题原始数据" --execute'.encode())
    TXT.write_bytes(content)
    theme = (AUDIT / ('修复前_' + THEME.name)).read_bytes()
    replacements = {}
    for old, new in [('03_结果/', 'A2601605结果/'),
                     ('02_数据_不含赛题原始数据/', 'A2601605数据_不含赛题原始数据/')]:
        replacements[old] = theme.count(old.encode())
        assert replacements[old] > 0
        theme = theme.replace(old.encode(), new.encode())
    THEME.write_bytes(theme)
    after = inventory(SOURCE)
    changed = [name for name in before['support'] if before['support'][name] != after[name]]
    assert changed == ['A2601605源程序/绘图主题.R', 'A2601605软件与运行说明.txt'], changed
    assert set(before['support']) == set(after)

    staging = AUDIT / 'A2601605附件.zip'
    with ZipFile(staging, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        for name in after:
            archive.write(SOURCE / name, SOURCE.name + '/' + name)
    checks = {}
    with tempfile.TemporaryDirectory(prefix='mcup_v13_p0_') as folder:
        temp = Path(folder)
        with ZipFile(staging) as archive:
            assert len(archive.infolist()) == 181 and archive.testzip() is None
            archive.extractall(temp)
        work = temp / SOURCE.name
        assert inventory(work) == after
        instructions = (work / TXT.name).read_text()
        commands = [shlex.split(line) for line in instructions.splitlines()
                    if line.startswith('conda run') and 'A2601605源程序/' in line]
        assert len(commands) == 5
        for command in commands:
            script = command[command.index('python') + 1] if 'python' in command else command[command.index('Rscript') + 1]
            assert (work / script).is_file(), script
        checks['five_documented_scripts_exist'] = True
        execute = next(c for c in commands if '--execute' in c)
        assets = work / execute[execute.index('--assets') + 1]
        assert (assets / 'Q1_感官目标字典.csv').is_file()
        assert (assets / 'Q1_外层交叉验证划分.csv').is_file()
        checks['Q1_explicit_assets_exist'] = True
        env = dict(os.environ, PYTHONPYCACHEPREFIX=str(temp / 'bytecode'),
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        command = [CONDA, *commands[0][1:]]
        run = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=120)
        (AUDIT / 'Q1_help.txt').write_text(run.stdout + run.stderr)
        assert run.returncode == 0 and '--assets' in run.stdout, run.stderr
        checks['exact_documented_help_exit_0'] = True
        python_files = sorted((work / 'A2601605源程序').glob('*.py'))
        for path in python_files:
            ast.parse(path.read_text())
            run = subprocess.run([PYTHON, '-m', 'py_compile', str(path)], env=env,
                                 capture_output=True, text=True, timeout=30)
            assert run.returncode == 0, run.stderr
        checks['all_20_Python_files_compile'] = len(python_files) == 20
        # Parse the R scripts, source only the shared definitions, and read all mapped
        # inputs. Do not execute the graph scripts or open any graphics devices.
        rcheck = temp / 'check_inputs.R'
        rcheck.write_text('''
files <- list.files("A2601605源程序", pattern="[.]R$", full.names=TRUE)
stopifnot(length(files) == 10)
for (file in files) invisible(parse(file=file))
.mcup_support <- getwd()
source("A2601605源程序/绘图主题.R")
paths <- unique(unname(.mcup_csv_map))
stopifnot(length(paths) == 120, all(file.exists(paths)))
for (path in paths) {
  data <- utils::read.csv(path, check.names=FALSE)
  stopifnot(ncol(data) > 0)
}
cat("R_PARSE=10\\nINPUT_PATHS=120\\nREAD_OK=120\\nNO_PLOTTING=TRUE\\n")
''')
        run = subprocess.run([CONDA, 'run', '-n', 'r', 'Rscript', str(rcheck)], cwd=work,
                             env=env, capture_output=True, text=True, timeout=120)
        (AUDIT / 'R语法与120输入.txt').write_text(run.stdout + run.stderr)
        assert run.returncode == 0 and 'READ_OK=120' in run.stdout, run.stderr
        checks['all_10_R_scripts_parse'] = True
        checks['all_120_R_input_paths_exist_and_read'] = True
        assert not (work / '绘图复算输出').exists() and not (work / '运行结果').exists()
        checks['no_training_or_plot_outputs_created'] = True

    assert inventory(SOURCE) == after
    for path in protected:
        assert digest(path) == before['papers'][str(path)]
    checks['V12_V13_and_submission_PDF_unchanged'] = True
    for destination in DESTINATIONS:
        shutil.copy2(staging, destination)
        assert digest(destination) == digest(staging)
    # All pre-existing local delivery files remain identical except the new archive.
    local = inventory(ROOT / '06_交付')
    assert set(local) == set(before['delivery'])
    assert [p for p in local if local[p] != before['delivery'][p]] == ['A2601605附件.zip']
    checks['local_delivery_only_archive_changed'] = True
    result = {'passed': all(checks.values()), 'checks': checks, 'files': len(after),
              'bytes': staging.stat().st_size, 'sha256': digest(staging),
              'old_sha256': before['archives'][str(DESTINATIONS[0])],
              'changed_submission_files': changed, 'unchanged_submission_files': len(after) - len(changed),
              'path_replacements': replacements, 'commands': [' '.join(c) for c in commands],
              'destinations': [str(p) for p in DESTINATIONS], 'papers_changed': False,
              'training_executed': False, 'plots_regenerated': False,
              'scope': 'P0-4: documented paths, explicit Q1 assets, R input mappings; no end-to-end model rerun'}
    (AUDIT / '验收.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (AUDIT / 'SHA256SUMS').write_text(f'{result["sha256"]}  A2601605附件.zip\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
