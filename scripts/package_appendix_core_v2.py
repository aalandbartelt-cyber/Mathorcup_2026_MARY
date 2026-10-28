"""Build the appendix core-code handoff with software and command notes.

The previous v1 archive is intentionally left untouched.  This builder creates
an additive v2 package from the neutral staging tree and adds the prose that a
paper editor can place immediately before each code block in Appendices C-E.
Only syntax/import/hash checks are run; no modelling function is called.
"""
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import ast
import csv
import json
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/collab/20261027_总论文v6与核心代码'
STAGING = AUDIT / '代码暂存'
DELIVERY = ROOT / '05_论文/交付给论文手'
FOLDER = DELIVERY / '三问附录核心代码_v2'
ZIP = DELIVERY / '三问附录核心代码_v2.zip'

VERIFY = '''"""Static/hash/import checks only; never call a modelling function."""
from pathlib import Path
from hashlib import sha256
import ast
import importlib.util
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
for line in (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    digest, relative = line.split('  ', 1)
    assert sha256((root / relative).read_bytes()).hexdigest() == digest, relative
modules = []
for path in sorted(root.glob('Q*/*.py')):
    source = path.read_text(encoding='utf-8')
    ast.parse(source)
    compile(source, path.name, 'exec')
    sys.path.insert(0, str(path.parent))
    name = 'appendix_' + path.parent.name[:2] + '_' + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    sys.path.pop(0)
    modules.append(path.relative_to(root).as_posix())
print('SHA256 verified; syntax and imports valid for', len(modules), 'core modules.')
print('No training, sampling, ranking or result regeneration was executed.')
'''

SOFTWARE_DOC = '''# 软件与命令说明（附录代码前置说明）

本文件是附录代码的运行说明。论文手在附录 C、D、E 的每段核心代码前，应先放入对应问题的“软件与命令”段，再接代码块。下面的命令只做环境确认、语法检查、模块导入和文件哈希核对，不会训练模型、抽样生成、搜索候选、重新排名或改写论文结果。

## 共用软件环境

实际使用的软件为 WSL2 Ubuntu 环境中的 `Python 3.10.19`（conda 环境名 `mathorcup`），并使用 `NumPy 2.2.6`、`SciPy 1.15.3`、`pandas 2.3.3` 和 `scikit-learn 1.7.2`。核心代码按 CPU 运行，不需要 GPU、PyTorch 或强化学习框架。完整工程中的缓存读取可能另外需要 `joblib` 和 `PyYAML`，但它们不属于本函数级交接包的静态检查依赖。

在包根目录执行环境核对命令：

```bash
conda activate mathorcup
python --version
python -c "import numpy, scipy, pandas, sklearn; print(numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__)"
python verify_core.py
```

`verify_core.py` 只检查 `SHA256SUMS`、Python AST/字节码编译和模块导入；它不读取官方附件，也不调用任何训练、生成、排序或化学分级函数。

## 附录 C：问题一感官预测

**软件与命令：** 本段代码在 Python 3.10.19、NumPy 2.2.6、SciPy 1.15.3 环境中运行；完整折内流程还依赖 scikit-learn 1.7.2 及支撑工程中的预处理和基学习器包装。进入包根目录后执行 `python -m py_compile Q1_感官预测/nnls_core.py`，再执行 `python verify_core.py`。这两条命令只验证非负堆叠函数的语法、导入和哈希，不重新拟合权重，也不生成 OOF 或结果文件。通过后在附录中接排 `nnls_core.py` 与 `inner_oof_excerpt.txt` 的核心代码。

## 附录 D：问题二关键化合物

**软件与命令：** 本段代码在 Python 3.10.19、NumPy 2.2.6、pandas 2.3.3、scikit-learn 1.7.2 环境中运行；正式四路排序、条件归因和化学核验所需的候选表、冻结预测器及字段字典由完整支撑工程提供。进入包根目录后执行 `python -m py_compile Q2_关键化合物/rank_core.py Q2_关键化合物/frozen_nnls_core.py Q2_关键化合物/conditional_core.py Q2_关键化合物/chemical_core.py`，再执行 `python verify_core.py`。这些命令只检查语法、导入和哈希，不重新排名、不拟合条件模型、不改变正式 Top-3。通过后依次排入四路排序、冻结预测、条件残差置换和 OAV/证据等级代码。

## 附录 E：问题三风格生成

**软件与命令：** 本段代码在 Python 3.10.19、NumPy 2.2.6、SciPy 1.15.3、pandas 2.3.3、scikit-learn 1.7.2 环境中运行；完整生成与迁移工程可能按缓存读取需要 joblib、PyYAML，但本包不加载缓存。进入包根目录后执行 `python -m py_compile Q3_风格生成/*.py`，再执行 `python verify_core.py`。这些命令只检查风格 Copula、浓度约束、迁移筛选、感官推理、证据状态和分布诊断模块，不生成新谱、不搜索新路径、不重新计算正文指标。通过后按 `附录编排建议.md` 的顺序排入核心函数。

## 论文中的边界表述

附录代码是函数级核心展示和审计入口，不是从原始附件一键复现全部正文数字的独立工程。若需要完整复算，应使用支撑材料中的数据契约、冻结配置、划分记录、模型缓存和结果清单；不得把上述静态检查命令表述为训练或结果重算命令。
'''

Q1_PREFIX = '''
## 软件与命令（排在本问题代码之前）

本问题核心代码使用 WSL2 Ubuntu 中 conda 环境 `mathorcup` 的 Python 3.10.19，依赖 NumPy 2.2.6、SciPy 1.15.3；完整折内流程另使用 scikit-learn 1.7.2。代码段前先在包根目录执行 `python -m py_compile Q1_感官预测/nnls_core.py`，随后执行 `python verify_core.py`。这里只做语法、导入和哈希核对，不重新拟合权重、不生成 OOF，故不会改动正文结果。

'''
Q2_PREFIX = '''
## 软件与命令（排在本问题代码之前）

本问题核心代码使用 WSL2 Ubuntu 中 conda 环境 `mathorcup` 的 Python 3.10.19，依赖 NumPy 2.2.6、pandas 2.3.3 和 scikit-learn 1.7.2。代码段前先执行 `python -m py_compile Q2_关键化合物/rank_core.py Q2_关键化合物/frozen_nnls_core.py Q2_关键化合物/conditional_core.py Q2_关键化合物/chemical_core.py`，随后执行 `python verify_core.py`。这里只做语法、导入和哈希核对，不重新排名、不拟合条件模型、不改变正式 Top-3。

'''
Q3_PREFIX = '''
## 软件与命令（排在本问题代码之前）

本问题核心代码使用 WSL2 Ubuntu 中 conda 环境 `mathorcup` 的 Python 3.10.19，依赖 NumPy 2.2.6、SciPy 1.15.3、pandas 2.3.3 和 scikit-learn 1.7.2。代码段前先执行 `python -m py_compile Q3_风格生成/*.py`，随后执行 `python verify_core.py`。这里只做语法、导入和哈希核对，不生成新谱、不搜索新路径、不重算正文指标，也不占用 GPU。

'''


def _prepend(path: Path, prefix: str) -> None:
    original = path.read_text(encoding='utf-8')
    if '## 软件与命令（排在本问题代码之前）' not in original:
        path.write_text(original.split('\n', 1)[0] + '\n' + prefix + original.split('\n', 1)[1], encoding='utf-8')


def _write_docs(folder: Path) -> None:
    (folder / '软件与命令说明.md').write_text(SOFTWARE_DOC, encoding='utf-8')
    _prepend(folder / 'Q1_感官预测' / 'README.md', Q1_PREFIX)
    _prepend(folder / 'Q2_关键化合物' / 'README.md', Q2_PREFIX)
    _prepend(folder / 'Q3_风格生成' / 'README.md', Q3_PREFIX)

    delivery = folder / '交付说明.md'
    text = delivery.read_text(encoding='utf-8')
    insert = ('\n## 官方附录要求：软件名称与命令\n\n'
              '官方要求附录列出参赛者实际使用的软件名称、版本和命令。本包已在根目录 `软件与命令说明.md` 及三个问题 README 的代码段前分别给出该说明；论文手应将对应段落原样放在每段核心代码之前。命令只做静态核验，不声称重新训练或一键复现全部结果。\n')
    if '## 官方附录要求：软件名称与命令' not in text:
        text = text.replace('\n## 输入输出与运行边界', insert + '\n## 输入输出与运行边界', 1)
    delivery.write_text(text, encoding='utf-8')

    layout = folder / '附录编排建议.md'
    text = layout.read_text(encoding='utf-8')
    insert = ('\n## 每段代码之前必须写的软件与命令说明\n\n'
              '在附录 C、D、E 的每个核心代码小节开头，先放对应 README 中的“软件与命令（排在本问题代码之前）”段落，再接代码。该段落至少保留软件名称、版本、实际核验命令以及“不训练、不重算结果”的边界；不要只写“使用 Python”，也不要把静态核验命令写成完整实验命令。\n')
    if '## 每段代码之前必须写的软件与命令说明' not in text:
        text = text.replace('\n## 排版', insert + '\n## 排版', 1)
    layout.write_text(text, encoding='utf-8')


def main() -> None:
    DELIVERY.mkdir(parents=True, exist_ok=True)
    if FOLDER.exists():
        # This is an additive v2 build.  Never delete or alter v1.
        for p in FOLDER.rglob('*'):
            if p.is_dir() and p.name == '__pycache__':
                raise RuntimeError(f'unexpected bytecode directory: {p}')
    for name in ['Q1_感官预测', 'Q2_关键化合物', 'Q3_风格生成']:
        shutil.copytree(STAGING / name, FOLDER / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Start the package-level prose from the existing v1 wording, then add the
    # official software/command requirement below.  v1 itself is untouched.
    for filename in ['交付说明.md', '附录编排建议.md']:
        source = DELIVERY / '三问附录核心代码_v1' / filename
        if source.exists():
            shutil.copy2(source, FOLDER / filename)
    _write_docs(FOLDER)
    requirements = '\n'.join(f'{p}=={version(p)}' for p in ['numpy', 'scipy', 'pandas', 'scikit-learn']) + '\n'
    (FOLDER / 'requirements-core.txt').write_text(requirements, encoding='utf-8')
    (FOLDER / 'verify_core.py').write_text(VERIFY, encoding='utf-8')

    # Keep the original neutral mapping, adding the prose document to the ZIP.
    mapping = [
        ('C', 'Q1_感官预测/nnls_core.py', 'NNLSStackingFit/_solve/fit_nnls_stacking', '训练折内层OOF张量、目标和家族标签', '非负收缩权重、自由截距、融合预测', '式(2)-(5)', '核心'),
        ('C', 'Q1_感官预测/inner_oof_excerpt.txt', '外层训练内拟合流程', '训练样本、内层划分、四模型候选配置', '内层OOF和外层拟合结果', '5.2-5.3', '上下文摘录'),
        ('D', 'Q2_关键化合物/rank_core.py', 'reciprocal_rank_consensus/aggregate_leave_one_style_ranks', '各路完整候选名次、留一风格完整名次', '四路融合排名、稳定性汇总', '式(13)-(14)', '核心'),
        ('D', 'Q2_关键化合物/four_route_excerpt.txt', '四路排名连接', '总关联、分层自助、风格校正、留一风格表', '融合输入和正式排名', '6.1', '上下文摘录'),
        ('D', 'Q2_关键化合物/frozen_nnls_core.py', 'FrozenFoldNNLS/raw_module_permutation', '已拟合4模型、权重、截距、231字段、训练模块', '固定预测器的原始/置换预测', '式(12)', '核心'),
        ('D', 'Q2_关键化合物/conditional_core.py', '条件期望/残差置换/_fold_field_rows', '训练模块同伴和风格、验证观测和固定预测器', '字段条件置换后的MAE差', '式(12)', '核心'),
        ('D', 'Q2_关键化合物/chemical_core.py', '_oav_summary/_grade', '同单位浓度/阈值、既有语义和统计证据', 'OAV汇总、既有A-D证据规则', '式(15)', '核心'),
        ('E', 'Q3_风格生成/style_copula.py', 'StyleGaussianGenerator', '训练风格标签及有限化学矩阵', '条件生成化学谱', '式(16)-(19)', '核心'),
        ('E', 'Q3_风格生成/concentration_constraints.py', '浓度转换/直接投影/秩端点', 'raw或log10字段、源浓度与已知掩码', '候选的直接浓度约束', '式(20)-(21)', '核心'),
        ('E', 'Q3_风格生成/migration_search.py', 'constrained_paths/select_candidates', '固定端点、路径比例、代理与距离', '有限集合内候选和最小改动选择', '式(20)-(23)', '核心'),
        ('E', 'Q3_风格生成/sensory_inference.py', '冻结推理/经验CDF/支持域', '调用者已核验的最终模型及训练参照', '画像预测、相对量表、支持域诊断', '7.1-7.2', '支撑核心'),
        ('E', 'Q3_风格生成/evidence_status.py', 'audit_concentrations/proxy_screen/candidate_status', '直接证据、代理、收益和支持域', '完整候选证据状态', '7.2-7.3', '核心'),
        ('E', 'Q3_风格生成/distribution_diagnostics.py', 'distribution_diagnostics', '真实留出谱、生成谱及训练尺度', '分布距离、几何覆盖等', '7.3', '支撑核心'),
        ('E', 'Q3_风格生成/核心流程摘录.txt', '画像汇总流程', '已生成化学谱和冻结预测器', '风格-目标画像统计', '7.2', '上下文摘录'),
    ]
    with (FOLDER / '核心代码与论文对应.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['附录', '文件', '函数或流程', '输入', '输出', '正文对应', '建议用途'])
        writer.writerows(mapping)

    forbidden = re.compile(r'\bE\d{3}\b|\be\d{3}[_\w]*|BZD|\bagent\d*\b|学长版|参考组|对照组|晋级|门槛|promotion|/home/mary|/mnt/[a-z]/', re.I)
    allfiles = [p for p in sorted(FOLDER.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    hits = []
    for p in allfiles:
        value = p.read_text(encoding='utf-8-sig')
        if forbidden.search(p.relative_to(FOLDER).as_posix()) or forbidden.search(value):
            hits.append(p.relative_to(FOLDER).as_posix())
    if hits:
        raise ValueError(f'internal labels in code package: {hits}')
    codefiles = list(FOLDER.glob('Q*/*.py'))
    for p in codefiles:
        ast.parse(p.read_text(encoding='utf-8'))
        compile(p.read_text(encoding='utf-8'), p.name, 'exec')
    # Write the manifest after all generated text is final.
    (FOLDER / 'SHA256SUMS').write_text(
        ''.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(FOLDER).as_posix()}\n'
                for p in allfiles if p.name != 'SHA256SUMS' or p.parent != FOLDER),
        encoding='utf-8')
    check = subprocess.run([sys.executable, str(FOLDER / 'verify_core.py')], check=True, capture_output=True, text=True)
    payload = [p for p in sorted(FOLDER.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    with ZipFile(ZIP, 'w', ZIP_DEFLATED, compresslevel=9) as z:
        for p in payload:
            z.write(p, arcname=FOLDER.name + '/' + p.relative_to(FOLDER).as_posix())
    with ZipFile(ZIP) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(payload)
        for p in payload:
            assert z.read(FOLDER.name + '/' + p.relative_to(FOLDER).as_posix()) == p.read_bytes()
    result = {
        'zip': str(ZIP.relative_to(ROOT)), 'files': len(payload), 'core_modules': len(codefiles),
        'python_core_lines': sum(len(p.read_text(encoding='utf-8').splitlines()) for p in codefiles),
        'zip_bytes': ZIP.stat().st_size, 'zip_sha256': sha256(ZIP.read_bytes()).hexdigest(),
        'syntax_import_hash_check': check.stdout.strip(), 'internal_label_scan_hits': hits,
        'software_command_document': '软件与命令说明.md',
        'no_training_or_result_recomputation': True,
        'scope': 'function-level appendix cores, not standalone full-data reproduction',
        'manifest': {str(p.relative_to(FOLDER)): {'bytes': p.stat().st_size, 'sha256': sha256(p.read_bytes()).hexdigest()} for p in payload},
    }
    (AUDIT / '代码包核验_v2.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (ZIP.with_suffix('.zip.sha256')).write_text(f'{result["zip_sha256"]}  {ZIP.name}\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'manifest'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
