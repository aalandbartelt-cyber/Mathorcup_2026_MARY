"""Assemble real final algorithm excerpts for a teammate's appendix editing."""
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
FOLDER = DELIVERY / '三问附录核心代码_v1'
ZIP = DELIVERY / '三问附录核心代码_v1.zip'

VERIFY = '''"""Static/hash/import checks only; never call a modeling function."""
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


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name in ['Q1_感官预测', 'Q2_关键化合物', 'Q3_风格生成']:
        destination = FOLDER / name
        if destination.exists():
            # Retain unrelated old contents; the current source list controls
            # the ZIP below. This is not a cleanup of the user's workspace.
            assert not list(destination.rglob('__pycache__'))
        shutil.copytree(STAGING / name, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (FOLDER / '交付说明.md').write_text('''# 三问附录核心代码交付说明

用途：供队友编排总论文的附录C、D、E。按问题分别收录最终实际算法的核心函数/类及流程摘录，保留原始运算、排序、缺失处理和约束。论文文本、结果数字及正式排序不在本包内修改。

## 先看哪些文件

- 附录C（问题一）：`Q1_感官预测/nnls_core.py`，配合内层折外流程摘录，说明非负权重、自由截距与感官组收缩。
- 附录D（问题二）：`rank_core.py`和四路连接摘录说明统计Top-3；`frozen_nnls_core.py`与`conditional_core.py`说明最终预测器的独立解释；`chemical_core.py`说明气味活性值及证据等级。
- 附录E（问题三）：`style_copula.py`说明平均秩、相关收缩与经验边缘反变换；浓度转换、路径选择、感官推理、证据状态和分布检验分别在同目录模块。

建议先读《附录编排建议.md》和各问题README，再挑选主要函数排入附录；不必把所有支撑函数全部贴入论文。

## 输入输出与运行边界

本包是函数级核心代码，不含官方数据、完整工程、30折配置、模型缓存和权重。函数需显式提供其输入；流程TXT为真实代码片段，依赖上下文，不是独立脚本。输入输出与正文公式对应见`核心代码与论文对应.csv`。

完整复现实验应使用已有支撑工程、冻结配置及数据。不要把本包称为从原始附件一键复现全部论文数字的独立程序。冻结预测器的身份/哈希、样本和231字段顺序仍须在完整工程核验。

## 本次检查

在既有环境执行`python verify_core.py`，只核对内容哈希、Python语法与模块导入，不调用训练、抽样、搜索、排名或化学分级。检查不占GPU，也不重新生成论文结果。

Python 3.10及以上；第三方依赖见`requirements-core.txt`。各函数调用方式见相应README。来源中性功能名与原文件哈希保留，内部实验路径不进入此交接包。

## 重要口径

1. 感官预测使用最终感官组收缩NNLS，非负权重不要求和为1，截距自由；0-6是相对量表展示。
2. 问题二的四路统计排序与最终NNLS独立归因分开，不用归因重新排列150条答案；化学等级输入保持既有证据口径。
3. 问题三使用实际收缩Copula及既定最小改动迁移，未知直接证据不记为通过；状态中的英文计算枚举保留其程序意义，不代表论文评价或管理结论。
4. 生成与迁移属于统计设计，不能写成已验证可酿造、安全或全化学空间最优。

附录代码由队友排入论文，当前总论文的对应代码章节不在本轮自动展开。旧交付文件保留，本包不覆盖原模型工程。
''')
    (FOLDER / '附录编排建议.md').write_text('''# 附录编排建议

## 附录C 问题一模型核心程序

优先排`NNLSStackingFit.predict`、`_solve`、`fit_nnls_stacking`，对应正文非负拟合、家族收缩、自由截距及融合预测；另按篇幅给内层OOF流程，标明来自训练折而非验证折。基学习器的库调用和候选配置可留支撑工程。

## 附录D 问题二模型核心程序

先排四路连接和`reciprocal_rank_consensus`，明确完整候选名次、支持数优先、平滑常数和稳定并列规则。随后展示冻结NNLS预测接口、模块联合置换、字段条件残差置换及MAE差；二者是独立解释证据。最后择要列`_oav_summary/_grade`，不要漏掉不可比/缺失返回分支。

留一风格完整汇总、条件期望模型的预处理与类型校验可留作支撑；不能为了缩短代码，把模块构造或条件期望改到验证折。

## 附录E 问题三模型核心程序

按“平均秩正态化及Ledoit-Wolf相关收缩 - 经验分位反变换 - 直接浓度转换与投影 - 路径枚举和最小改动筛选 - 代理/未知证据状态”排列。正式生成调用`style_copula_shrunk`分支，不能把比较分支当主模型。

感官推理复用四个已拟合模型、最终权重及自由截距；无需在附录重复印两份相同预测接口。分布距离函数可按版面保留或移支撑。使用源缺失掩码，勿把填补浓度当真实已知浓度。

## 排版

- 代码保持等宽字体与原缩进，长行在编辑器或Word中排版，不改运算符和参数。
- 各问先写输入、输出、调用条件，再列主要函数。TXT摘录须保留其非独立入口说明。
- 全部源码均在对应问题文件夹，可复制或引用；不要混入其他题目的旧程序。
- 化学约束状态字符串是程序接口的一部分，正文应使用中文证据表述；不要把计算状态当作真实酿造认证。
''')
    requirements = '\n'.join(f'{p}=={version(p)}' for p in ['numpy', 'scipy', 'pandas', 'scikit-learn']) + '\n'
    (FOLDER / 'requirements-core.txt').write_text(requirements)
    (FOLDER / 'verify_core.py').write_text(VERIFY)
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
        writer = csv.writer(f); writer.writerow(['附录', '文件', '函数或流程', '输入', '输出', '正文对应', '建议用途']); writer.writerows(mapping)
    # Names and prose are neutral; actual state-machine enums are retained.
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
        ast.parse(p.read_text()); compile(p.read_text(), p.name, 'exec')
    (FOLDER / 'SHA256SUMS').write_text(''.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(FOLDER).as_posix()}\n'
                                           for p in allfiles if p.name != 'SHA256SUMS' or p.parent != FOLDER))
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
    result = {'zip': str(ZIP.relative_to(ROOT)), 'files': len(payload), 'core_modules': len(codefiles),
              'python_core_lines': sum(len(p.read_text().splitlines()) for p in codefiles),
              'zip_bytes': ZIP.stat().st_size, 'zip_sha256': sha256(ZIP.read_bytes()).hexdigest(),
              'syntax_import_hash_check': check.stdout.strip(), 'internal_label_scan_hits': hits,
              'no_training_or_result_recomputation': True,
              'scope': 'function-level appendix cores, not standalone full-data reproduction',
              'manifest': {str(p.relative_to(FOLDER)): {'bytes': p.stat().st_size, 'sha256': sha256(p.read_bytes()).hexdigest()} for p in payload}}
    (AUDIT / '代码包核验.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (ZIP.with_suffix('.zip.sha256')).write_text(f'{result["zip_sha256"]}  {ZIP.name}\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'manifest'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
