"""Extract source-identical function cores; never run any fit or result writer."""
from __future__ import annotations

import ast
from hashlib import sha256
import importlib
import json
from pathlib import Path
import re
import sys
import textwrap

ROOT = Path('/home/mary/mathorcup')
AUDIT = ROOT / 'docs/collab/20261027_总论文v6与核心代码'
STAGE = AUDIT / '代码暂存'
Q1 = STAGE / 'Q1_感官预测'
Q2 = STAGE / 'Q2_关键化合物'
SRC1 = ROOT / '.worktrees/q1-nnls-opt/02_代码/Q1_感官评分/src'
SRC2 = ROOT / '.worktrees/q2-evidence-opt/02_代码/Q2_关键化合物/src'
records = []
source_meta = {}


def source(path, logical):
    data = path.read_text()
    source_meta[logical] = {'logical_source': logical,
                            'source_path': str(path.relative_to(ROOT)),
                            'source_sha256': sha256(path.read_bytes()).hexdigest()}
    return data, ast.parse(data)


def segment(data, node):
    first = min([node.lineno] + [x.lineno for x in getattr(node, 'decorator_list', [])])
    return '\n'.join(data.splitlines()[first - 1:node.end_lineno])


def select(path, logical, names, substitutions=None):
    data, tree = source(path, logical)
    out = []
    for name in names:
        node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        raw = segment(data, node)
        code = raw
        for a, b in (substitutions or {}).items():
            code = code.replace(a, b)
        expected = ast.dump(ast.parse(code), include_attributes=False)
        assert expected == ast.dump(ast.parse(raw if not substitutions else code), include_attributes=False)
        records.append({**source_meta[logical], 'symbol': name,
                        'first_line': min([node.lineno] + [x.lineno for x in getattr(node, 'decorator_list', [])]),
                        'last_line': node.end_lineno,
                        'source_segment_sha256': sha256(raw.encode()).hexdigest(),
                        'exported_segment_sha256': sha256(code.encode()).hexdigest(),
                        'source_equal_except_documented_labels': not substitutions or raw == code or
                        all(k.startswith('E') for k in substitutions),
                        'metadata_substitutions': substitutions or {}})
        out.append(code)
    return '\n\n\n'.join(out)


def provenance(logicals):
    return '\n'.join(f"# Source logical name: {x}; SHA256: {source_meta[x]['source_sha256']}" for x in logicals)


def write_module(path, imports, body, logicals):
    path.write_text(provenance(logicals) + '\n' + imports.strip() + '\n\n\n' + body + '\n')


for d in [Q1, Q2]:
    d.mkdir(parents=True, exist_ok=True)

q1body = select(SRC1 / 'nnls_stacking.py', 'sensory_group_nnls',
                ['NNLSStackingFit', '_solve', 'fit_nnls_stacking'])
write_module(Q1 / 'nnls_core.py', '''from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import nnls''', q1body, ['sensory_group_nnls'])

flow = select(SRC1 / 'run_e009_nnls_stacking.py', 'nested_oof_stacking', ['fit_outer_nnls'],
              {'E009 base inner OOF is incomplete': 'stacking base inner OOF is incomplete'})
(Q1 / 'inner_oof_excerpt.txt').write_text(
    '内层OOF与外层训练内堆叠流程摘录\n'
    '这是完整函数体摘录，不是独立运行入口。以下函数不应在本次交付中被执行。\n'
    '依赖：numpy、make_inner_splits、_candidates、_fit_base、_parameter_key、matrix_metrics、fit_nnls_stacking。\n'
    '输入：当前外层训练集、当前验证化学输入、固定基学习器候选配置、感官家族、收缩值、内层折数和种子。\n'
    '输出：各共享方式预测、参数、已选基模型、内层划分、内层OOF基预测、外层验证基预测。\n'
    + provenance(['nested_oof_stacking']) + '\n\n' + flow + '\n')

rank = select(SRC2 / 'rank_consensus.py', 'support_aware_rank_fusion', ['reciprocal_rank_consensus'])
leave = select(SRC2 / 'robust_consensus.py', 'leave_style_rank_aggregation', ['aggregate_leave_one_style_ranks'])
as_lens = select(SRC2 / 'run_e111_robust_consensus.py', 'four_evidence_connection', ['_as_lens'])
write_module(Q2 / 'rank_core.py', '''from __future__ import annotations
import numpy as np
import pandas as pd''', rank + '\n\n\n' + leave + '\n\n\n' + as_lens,
             ['support_aware_rank_fusion', 'leave_style_rank_aggregation', 'four_evidence_connection'])

data, tree = source(SRC2 / 'run_e111_robust_consensus.py', 'four_evidence_connection')
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
conn_nodes = main.body[6:9]
assert len(conn_nodes) == 3 and ast.unparse(conn_nodes[0]).startswith('lenses = ')
connection = '\n'.join(textwrap.dedent(segment(data, n)) for n in conn_nodes)
for n in conn_nodes:
    records.append({**source_meta['four_evidence_connection'], 'symbol': 'main:four_route_connection',
                    'first_line': n.lineno, 'last_line': n.end_lineno,
                    'source_segment_sha256': sha256(segment(data, n).encode()).hexdigest(),
                    'exported_segment_sha256': sha256(textwrap.dedent(segment(data, n)).encode()).hexdigest(),
                    'change': 'indentation_only'})
(Q2 / 'four_route_excerpt.txt').write_text(
    '正式四路连接摘录：非独立入口\n'
    '输入total、bootstrap、adjusted、deletion分别是总关联、自助稳定、风格校正、留一风格汇总的完整名次表。\n'
    'deletion由aggregate_leave_one_style_ranks(..., expected_styles=22)得到。\n'
    '四表均须按target_code、feature_code覆盖全部候选；bootstrap使用bootstrap_rank，其他输入列名见函数。\n'
    '下面四个标签仅为可读路由名；支持数、倒数秩和平局顺序保留实际源算法。\n'
    + provenance(['four_evidence_connection']) + '\n\n'
    + 'LENS_TOTAL = "总关联"\nLENS_BOOTSTRAP = "分层自助"\nLENS_STYLE = "风格校正"\nLENS_DELETE = "留一风格"\n\n'
    + connection + '\n')

frozen = select(SRC2 / 'q1_final_adapter.py', 'frozen_sensory_nnls_interface',
                ['FrozenFoldNNLS', 'raw_module_permutation'])
write_module(Q2 / 'frozen_nnls_core.py', '''from __future__ import annotations
import numpy as np
import pandas as pd''', frozen, ['frozen_sensory_nnls_interface'])

cond = select(SRC2 / 'hierarchical_importance.py', 'conditional_residual_model',
              ['ConditionalFeatureModel', 'fit_conditional_feature_model', 'residual_permutation'])
fields = select(SRC2 / 'run_e115_final_nnls.py', 'final_nnls_hierarchical_attribution',
                ['_fold_field_rows'],
                {'E113_fold_local_all_modules_no_validation_gate': 'fold_local_all_modules_no_validation_gate'})
write_module(Q2 / 'conditional_core.py', '''from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
try:
    from .frozen_nnls_core import raw_module_permutation
except ImportError:
    from frozen_nnls_core import raw_module_permutation''', cond + '\n\n\n' + fields,
             ['conditional_residual_model', 'final_nnls_hierarchical_attribution'])

chem = select(SRC2 / 'run_e112_chemical_evidence.py', 'official_chemical_grading',
              ['_oav_summary', '_grade'])
write_module(Q2 / 'chemical_core.py', '''from __future__ import annotations
import numpy as np
import pandas as pd''', chem, ['official_chemical_grading'])

(Q1 / 'README.md').write_text('''# 问题一附录核心

本目录是函数级核心及输入输出说明，不含数据、模型缓存或已选参数，不声称是独立复现工程。没有自动运行入口；导入模块不会训练模型。

- `nnls_core.py`：真实最终非负堆叠计算。`fit_nnls_stacking`输入形状为“训练样本×感官目标×4个基学习器”的内层OOF预测及训练目标；输出冻结权重、自由截距、共享权重及预测方法。非负权重不要求和为1；感官组向共享权重固定收缩0.5。
- `inner_oof_excerpt.txt`：真实外层训练内的完整拟合流程摘录，保留函数体，但依赖划分、候选参数与四种基模型适配，所以不是独立入口。

依赖Python 3.10、NumPy、SciPy。内层流程另依赖scikit-learn、预处理及四种基模型包装、评价函数、固定10次重复3折划分和既有参数配置。单次函数调用不等于整套外层验证。

建议论文附录优先排`_solve`与`fit_nnls_stacking`，保留`NNLSStackingFit`的预测接口；内层OOF流程按篇幅摘录，其依赖留在支撑材料。填写新数据或重训应另行使用完整工程，不能凭本目录声称复现全部论文数字。

来源中性名称及原文件SHA256见各文件头和`source_checksums.json`。校验散列用于追溯，不代表本目录重新运行了训练。
''')

(Q2 / 'README.md').write_text('''# 问题二附录核心

本目录是从真实最终源码抽取的函数级核心，不含官方数据、完整名次表、模型缓存、注册表、参数或结果；不声称是独立复现工程。没有自动执行入口，导入不运行训练或生成排名。

## 正式统计答案

- `rank_core.py`：支持数优先的倒数秩融合、留一风格完整名次汇总，以及四路输入的格式适配。输入表必须覆盖每个感官目标的完整候选范围，不能只把各路前三项送入融合。排序先比较Top-10和Top-3支持数，再比较倒数秩分数及既定平局键。
- `four_route_excerpt.txt`：真实四路连接段摘录；总关联、分层自助、风格校正和留一风格四表先在外部准备，本文正式答案由这四路确定。摘录不是独立入口。

## 最终感官模型的独立解释

- `frozen_nnls_core.py`：`FrozenFoldNNLS`接收四个已训练且顺序正确的基模型、已收缩权重、自由截距和231个原始字段名。它不估计新权重。`raw_module_permutation`移动模块原值及缺失掩码，其他字段保持原样。
- `conditional_core.py`：训练折用模块同伴与风格拟合字段条件期望；验证折只置换观测残差，缺失位置保留；`_fold_field_rows`计算模块与字段两层的置换后MAE差。模块来自训练折，不能按验证目标选择。

独立解释不进入正式四路排序。预测器仍输入231列，218严格单体只限定候选和可置换字段；其余13列保留为预测上下文。调用者必须先用完整工程核对冻结模型身份、OOF哈希、30折参数、样本/字段顺序及划分，本目录不包含这些资产。

条件期望模型只有在调用相关函数时才拟合；本次交付仅完成语法和导入检查，没有执行此拟合，也没有重跑论文结果。独立归因不是因果干预证明。此目录不包含字段遮蔽或重新排序入口。

## 化学核验

- `chemical_core.py`：实际OAV汇总与A-D等级规则。`_oav_summary`输入为已确认单位可比的原始浓度表及正阈值；不适用/缺失返回不可用，不以零代替N/A。`_grade`输入语义证据级别、既有统计稳健等级和既有可预测等级，输出A-D。

语义证据及可比性必须由完整核验资产提供；本目录不包含完整化学语义字典，也不把化学等级解释为因果或安全认证。不能用新的预测指标替换既有等级输入后重分级，更不能因此改变冻结150条关系。

依赖Python 3.10、NumPy、Pandas、scikit-learn；模块之间的条件归因依赖见导入。完整工程读取缓存时另需joblib。建议论文优先展示`reciprocal_rank_consensus`、四路连接段、冻结NNLS预测与条件残差置换，以及`_oav_summary/_grade`；训练管道和验证断言可作为支撑核心按篇幅安排。

来源只标中性逻辑名及原文件SHA256，见各文件头和`source_checksums.json`。元数据标签中的内部编号已去除，数值运算、排序键、权重和约束未改。
''')

for directory, logicals in [(Q1, ['sensory_group_nnls', 'nested_oof_stacking']),
                           (Q2, ['support_aware_rank_fusion', 'leave_style_rank_aggregation',
                                 'four_evidence_connection', 'frozen_sensory_nnls_interface',
                                 'conditional_residual_model', 'final_nnls_hierarchical_attribution',
                                 'official_chemical_grading'])]:
    public = [{k: source_meta[x][k] for k in ['logical_source', 'source_sha256']} for x in logicals]
    (directory / 'source_checksums.json').write_text(json.dumps(public, ensure_ascii=False, indent=2) + '\n')

# Verify extracted functions against original AST with only an explicit string
# relabeling allowance. No selected computational function is executed here.
class CleanLabels(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace('E113_fold_local_all_modules_no_validation_gate',
                                            'fold_local_all_modules_no_validation_gate')
            node.value = node.value.replace('E009 base inner OOF is incomplete',
                                            'stacking base inner OOF is incomplete')
        return node

for module in list(Q1.glob('*.py')) + list(Q2.glob('*.py')):
    code = module.read_text(); compile(code, str(module), 'exec')
    for node in ast.parse(code).body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        rec = next(r for r in records if r['symbol'] == node.name)
        original_tree = ast.parse((ROOT / rec['source_path']).read_text())
        original_node = next(x for x in original_tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == node.name)
        assert ast.dump(CleanLabels().visit(original_node), include_attributes=False) == ast.dump(node, include_attributes=False), node.name

sys.dont_write_bytecode = True
sys.path.insert(0, str(Q1)); sys.path.insert(0, str(Q2))
for name in ['nnls_core', 'rank_core', 'frozen_nnls_core', 'conditional_core', 'chemical_core']:
    importlib.import_module(name)

for path in list(Q1.iterdir()) + list(Q2.iterdir()):
    if path.is_file():
        assert not re.search(r'\bE\d{3}\b|BZD|agent\d*|学长', path.name + '\n' + path.read_text(), re.I), path
        assert '/home/' not in path.read_text() and '.worktrees' not in path.read_text(), path

files = [{'path': str(p.relative_to(STAGE)), 'lines': len(p.read_text().splitlines()),
          'bytes': p.stat().st_size, 'sha256': sha256(p.read_bytes()).hexdigest()}
         for d in [Q1, Q2] for p in sorted(d.iterdir()) if p.is_file()]
manifest = {'scope': 'function_level_core_no_training', 'sources': list(source_meta.values()),
            'extractions': records, 'files': files,
            'checks': {'source_function_ast_equal_after_metadata_labels': True,
                       'syntax_compile_modules': 5, 'imported_modules': 5,
                       'training_functions_called': 0, 'source_or_result_writes': 0,
                       'internal_label_hits': 0}}
(AUDIT / 'Q1Q2提取来源与核验.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'files': files, 'checks': manifest['checks']}, ensure_ascii=False, indent=2))
