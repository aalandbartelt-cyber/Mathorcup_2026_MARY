"""Evaluate frozen Q3 artifacts; do not fit or modify the frozen generators."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path('/home/mary/mathorcup')
WT = ROOT / '.worktrees/q3-foundation'
SOURCE = WT / '03_结果/Q3_风格生成'
DATA = ROOT / '05_论文/Q3_风格生成/待审_20261027/数据与结果/validation_v1'
INTERNAL = Path(__file__).parent
STYLE_ORDER = ['Tripel', 'Kriek', 'Lambic']
MODEL_NAMES = {
    'style_copula_shrunk': '风格收缩Copula',
    'shared_lowrank': '共享低秩',
    'style_residual_shrunk': '风格残差收缩',
    'style_residual_floor': '风格残差与地板状态',
}
METRICS = ['mean_w1', 'energy_v', 'coverage', 'spearman_pair_rmse', 'near_duplicate_fraction']
CHECK_METRICS = ['mean_w1', 'energy_v', 'coverage', 'spearman_pair_rmse']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return pd.read_csv(SOURCE / name)


def csv(frame, name):
    frame.to_csv(DATA / name, index=False)


def json_dump(data, path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    module_path = WT / '02_代码/Q3_风格生成/src/q3_evaluation.py'
    spec = importlib.util.spec_from_file_location('frozen_q3_diagnostics', module_path)
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    chem_path = ROOT / 'data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv'
    cv_path = ROOT / '03_结果/Q1_感官评分/q1_cv_splits.csv'
    chem, splits = pd.read_csv(chem_path), pd.read_csv(cv_path)
    features = [c for c in chem if c not in ['beer', 'beer_id', 'tasting_category_fine']]
    x, styles = chem[features].to_numpy(float), chem.tasting_category_fine.to_numpy(str)
    assert x.shape == (250, 231)
    archived = read('q3_e202_robust_sensitivity.csv')
    archived = archived[archived.method == 'style_copula_shrunk'].set_index(['repeat', 'fold', 'style', 'seed_index'])
    inputs = [chem_path, cv_path, module_path, SOURCE / 'q3_e202_robust_sensitivity.csv']
    original_rows, null_rows, completeness = [], [], []
    max_archived_error = {m: 0.0 for m in CHECK_METRICS}
    max_w1_error = 0.0
    for (repeat, fold), group in splits.groupby(['repeat', 'fold'], sort=True):
        repeat, fold = int(repeat), int(fold)
        slot = SOURCE / 'q3_checkpoints_E202' / f'R{repeat:02d}F{fold}'
        cache, mf = slot / 'generated.csv.gz', slot / 'manifest.json'
        inputs += [cache, mf]
        manifest = json.loads(mf.read_text())
        assert sha(cache) == manifest['files']['generated.csv.gz']
        valid = np.sort(group.row_position.to_numpy(int))
        train = np.setdiff1d(np.arange(len(x)), valid)
        assert set(manifest['train_beer']) == set(chem.beer.iloc[train].astype(int))
        assert set(manifest['validation_beer']) == set(chem.beer.iloc[valid].astype(int))
        median = np.nanmedian(x[train], axis=0)
        tx = np.where(np.isnan(x[train]), median, x[train])
        vx = np.where(np.isnan(x[valid]), median, x[valid])
        low, high = np.quantile(tx, [.01, .99], axis=0)
        clip_train = np.clip(tx, low, high)
        scale = np.std(clip_train, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.)
        cached = pd.read_csv(cache)
        copula = cached[cached.method == 'style_copula_shrunk']
        assert len(copula.groupby(['style', 'seed_index'])) == 15
        assert set(copula['style']) == set(STYLE_ORDER)
        assert copula.seed_index.nunique() == 5
        for (style, seed_index), block in copula.groupby(['style', 'seed_index'], sort=True):
            seed_index = int(seed_index)
            original = block[features].to_numpy(float)
            real = vx[styles[valid] == style]
            assert original.shape == real.shape and np.isfinite(original).all()
            local_train = np.clip(tx[styles[train] == style], low, high)
            g, r = np.clip(original, low, high), np.clip(real, low, high)
            base = diag.distribution_diagnostics(r, g, local_train, scale)
            base.pop('w1_by_feature'); base.pop('scope')
            key = {'repeat': repeat, 'fold': fold, 'style': style, 'generation_seed': seed_index}
            original_rows.append({**key, **base})
            old = archived.loc[(repeat, fold, style, seed_index)]
            for metric in CHECK_METRICS:
                max_archived_error[metric] = max(max_archived_error[metric], abs(float(base[metric]) - float(old[metric])))
            completeness.append({**key, 'n_real': len(real), 'n_generated': len(g), 'n_features': len(features), 'all_finite': bool(np.isfinite(original).all()), 'matched_count': len(real) == len(g), 'train_validation_disjoint': not bool(set(manifest['train_beer']) & set(manifest['validation_beer']))})
            for permutation in range(5):
                rng = np.random.default_rng(np.random.SeedSequence([2026102700, repeat, fold, STYLE_ORDER.index(style), seed_index, permutation]))
                shuffled = np.column_stack([g[rng.permutation(len(g)), j] for j in range(g.shape[1])])
                assert np.array_equal(np.sort(g, axis=0), np.sort(shuffled, axis=0))
                null = diag.distribution_diagnostics(r, shuffled, local_train, scale)
                null.pop('w1_by_feature'); null.pop('scope')
                invariant_error = abs(null['mean_w1'] - base['mean_w1'])
                max_w1_error = max(max_w1_error, invariant_error)
                assert invariant_error <= 1e-12
                null_rows.append({**key, 'permutation': permutation, **null, **{f'delta_{m}': null[m] - base[m] for m in CHECK_METRICS}})
        print(f'[{repeat * 3 + fold + 1:02d}/30] fixed spectra evaluated', flush=True)
    assert len(original_rows) == 450 and len(null_rows) == 2250
    assert max(max_archived_error[m] for m in ['mean_w1', 'energy_v', 'coverage']) <= 1e-9
    original_df, null_df = pd.DataFrame(original_rows), pd.DataFrame(null_rows)
    csv(pd.DataFrame(completeness), '生成缓存完整性.csv')
    csv(original_df, '固定生成谱诊断.csv')
    csv(null_df, '独立置换诊断.csv')
    deltas = [f'delta_{m}' for m in CHECK_METRICS]
    paired = null_df.groupby(['style', 'repeat'], as_index=False)[deltas].mean()
    csv(paired, '独立置换逐重复差.csv')
    agg = paired.groupby('style')[deltas].agg(['mean', 'std', 'min', 'max'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg = agg.reset_index()
    csv(agg, '独立置换汇总.csv')

    final_metrics = read('q3_metrics_by_repeat_E211.csv')
    input_names = ['q3_metrics_by_repeat_E211.csv', 'q3_e203_robust_by_repeat.csv', 'q3_migration_summary_E204.csv', 'q3_primary_candidates_E207.csv', 'q3_proxy_margin_sensitivity_E207.csv', 'q3_generated_profiles_E205.csv.gz']
    inputs += [SOURCE / n for n in input_names]
    final = final_metrics[final_metrics.method == 'style_copula_shrunk'].copy()
    final['method'] = final.method.map(MODEL_NAMES)
    final['metric_scale'] = final.metric_scale.map({'raw': '原始训练尺度', 'robust': '训练折分位稳健尺度'})
    csv(final, '主模型逐风格逐重复.csv')
    values = ['mean_w1', 'energy_v', 'coverage', 'spearman_pair_rmse', 'near_duplicate_fraction', 'pairwise_spread_ratio', 'log_spread_error']
    style_summary = final.groupby(['metric_scale', 'style'])[values].agg(['mean', 'std'])
    style_summary.columns = ['_'.join(c) for c in style_summary.columns]
    csv(style_summary.reset_index(), '主模型逐风格检验.csv')
    global_repeat = final.groupby(['metric_scale', 'repeat'], as_index=False)[values].mean()
    csv(global_repeat, '主模型逐重复检验.csv')
    robust_base = final[final.metric_scale == '训练折分位稳健尺度'][['method', 'style', 'repeat'] + METRICS]
    structures = read('q3_e203_robust_by_repeat.csv')
    structures['method'] = structures.method.map(MODEL_NAMES)
    assert structures.method.notna().all()
    ablation = pd.concat([robust_base, structures[['method', 'style', 'repeat'] + METRICS]], ignore_index=True)
    csv(ablation, '结构消融逐重复.csv')
    ablation_repeat = ablation.groupby(['method', 'repeat'], as_index=False)[METRICS].mean()
    a = ablation_repeat.groupby('method')[METRICS].agg(['mean', 'std'])
    a.columns = ['_'.join(c) for c in a.columns]
    csv(a.reset_index(), '结构消融汇总.csv')
    reference = a.loc['风格收缩Copula']
    table7 = []
    for method in MODEL_NAMES.values():
        s = a.loc[method]
        table7.append({'方法': method, 'W1': f"{s.mean_w1_mean:.4f}", 'energy': f"{s.energy_v_mean:.4f}", '覆盖率': f"{s.coverage_mean:.4f}", '相关RMSE': f"{s.spearman_pair_rmse_mean:.4f}", 'energy差': f"{s.energy_v_mean - reference.energy_v_mean:+.4f}", '说明': '最终生成模型' if method == '风格收缩Copula' else '相关结构略改善，边缘/联合距离增加'})
    csv(pd.DataFrame(table7), '表7_结构消融.csv')

    profiles = read('q3_generated_profiles_E205.csv.gz')
    finite_fields = ['prediction_z', 'grade_0_6', 'model_fold_sd', 'audit_prediction_z', 'predictor_support_fraction']
    profile_finite = bool(np.isfinite(profiles[finite_fields]).all().all())
    m = read('q3_migration_summary_E204.csv')
    primary = read('q3_primary_candidates_E207.csv')
    margin = read('q3_proxy_margin_sensitivity_E207.csv')
    margin = margin[margin.was_main_candidate_eligible].groupby('margin_z', as_index=False).both_proxy_nonworsening.sum()
    margin.columns = ['要求下降裕量_z', '两模型满足数']
    margin['冻结主候选数'] = 2
    csv(margin, '固定候选裕量敏感性.csv')
    stages = [
        ('早期策略记录', 891, '策略/路径记录', '11源×3目标×3情景×9策略路径'),
        ('有生成谱', int(m.candidate_available.sum()), '策略/路径记录', '有谱不是独立啤酒'),
        ('结构拒绝', int((~m.candidate_available).sum()), '策略/路径记录', '无谱保留分母'),
        ('Pareto标记', int(m.pareto_front.sum()), '策略/路径记录', '并行属性，不是下一筛选级'),
    ]
    status_names = {'needs_direct_verification': '直接证据不足', 'no_target_gain': '无目标收益', 'rejected_direct_constraint': '直接约束拒绝', 'rejected_no_structural_candidate': '无结构候选', 'needs_proxy_verification': '等待代理接入'}
    for name, n in m.status.value_counts().items():
        stages.append((status_names[name], int(n), '早期策略/路径记录', '状态互斥；等待接入不写作模型全失败'))
    stages += [('当前固定主任务', len(primary), '源—目标任务', '11源×3目标'), ('当前有谱任务', int(primary.candidate_available.sum()), '源—目标任务', '固定最小改动规则')]
    statuses = {'needs_direct_verification': '当前直接证据不足', 'rejected_no_primary_feasible_candidate': '当前无搜索可行谱', 'eligible_statistical_candidate': '当前待实测统计候选', 'proxy_variation_uncertain': '当前代理不稳定'}
    for name, n in primary.audited_status.value_counts().items():
        stages.append((statuses[name], int(n), '源—目标任务', '两条统计候选来自同一源'))
    csv(pd.DataFrame(stages, columns=['检验项', '结果', '单位', '口径']), '迁移阶段分母核验.csv')

    table6 = [
        {'检验项': '生成缓存与划分一致性', '结果': '30折×3风格×5种子齐全，逐组生成数匹配留出数', '口径': '固定外层缓存；全部231字段有限；训练与验证beer无交叉', '支持程度': '支持程序与数据一致性'},
        {'检验项': '画像完整性', '结果': f'{len(profiles):,}个目标值全部有限；每风格300谱×50目标', '口径': '全数据描述画像；0–6为经验相对等级', '支持程度': '支持画像完整，不是实测评分验证'},
        {'检验项': '精确复制与塌缩', '结果': '近训练复制率0，整组塌缩标记0', '口径': '30外层折×5生成种子；复制容差为标准化RMS≤10⁻⁶', '支持程度': '支持无精确/极近复制；地板恒定维度保留'},
        {'检验项': '三风格留出一致性', '结果': '稳健W1为0.4114/0.5684/0.4487；覆盖0.3356/0.7389/0.5994', '口径': 'Tripel/Kriek/Lambic；折种子先聚合，十次重复描述', '支持程度': '风格间差异明显，小风格相关结构证据较弱'},
        {'检验项': '尺度与极值敏感性', '结果': '原始W1/energy=1.0773/6.3933；稳健0.4762/0.5813', '口径': '稳健边界仅由外层训练折1%–99%分位确定', '支持程度': '极端记录影响显著，稳健数字不能替代原始结果'},
        {'检验项': '三结构消融', '结果': '三变体energy增加0.0658–0.0703，相关RMSE略降', '口径': '共享低秩/风格残差收缩/风格残差与地板状态；与主模型同折', '支持程度': '支持保留主模型，同时保留相关结构不足'},
        {'检验项': '迁移阶段分母', '结果': '891记录=844有谱+47结构拒绝；Pareto标记293为并行属性', '口径': '早期直接约束阶段；15条当时等待代理接入', '支持程度': '支持状态一致性，不把未接入当作全失败'},
        {'检验项': '固定主规则与裕量', '结果': '33任务：2候选/26证据不足/1代理不稳/4无谱；裕量0/0.01/0.02时2/0/0', '口径': '固定候选不重选；两候选来自同一个源', '支持程度': '统计设计对代理裕量敏感，需独立实测'},
        {'检验项': '未见风格与实际酿造', '结果': '尚未检验', '口径': '10×3为已知风格内留出；无独立工艺/新批次感官数据', '支持程度': '不支持可实际酿造、唯一或全局最优主张'},
    ]
    null_summary = agg.set_index('style')
    corr_delta = float(paired.delta_spearman_pair_rmse.mean())
    energy_delta = float(paired.delta_energy_v.mean())
    table6.insert(5, {'检验项': '保持边缘的独立置换', '结果': f'W1最大变化{max_w1_error:.2g}；energy差{energy_delta:+.5f}；相关RMSE差{corr_delta:+.5f}', '口径': '每缓存5次独立置换；差为置换−原谱；折/种子/置换先聚合到重复', '支持程度': '仅联合结构诊断；如有负差，保留为不利证据'})
    csv(pd.DataFrame(table6), '表6_补充检验.csv')
    internal = {'source_sha256': {str(p): sha(p) for p in sorted(set(inputs))}, 'max_robust_recomputation_error_vs_archive': max_archived_error, 'max_edge_w1_invariance_error': max_w1_error, 'fixed_generated_groups': len(original_rows), 'permutation_evaluations': len(null_rows), 'profile_all_finite': profile_finite, 'outputs': {p.name: sha(p) for p in DATA.glob('*.csv')}}
    json_dump(internal, INTERNAL / 'supplemental_audit_sources.json')
    public = {'status': 'pending_user_review', 'scope': 'frozen_spectrum_diagnostics', 'n_outer_repeats': 10, 'n_outer_folds': 30, 'n_generation_seeds_per_fold_style': 5, 'n_independent_column_permutations_per_group': 5, 'n_original_groups': len(original_rows), 'n_permutation_evaluations': len(null_rows), 'max_marginal_w1_invariance_error': max_w1_error, 'model_refitted': False, 'new_spectra_sampled': False, 'uncertainty_scope': 'repeat_variation_not_independent_confidence_guarantee', 'table6': table6, 'table7': table7, 'limitations': ['置换保持边缘但破坏样本内配对，不是打乱风格标签后的重新拟合。', '小风格留出数很少，相关矩阵诊断有较大抽样误差。', '重复训练集相互重叠，重复标准差不等于独立置信区间。', '固定两候选追加0.01 z的双模型下降裕量后均不再满足。', '未知浓度及未知绝对阈值保留未决，未验证实际酿造。'], 'files_sha256': internal['outputs']}
    json_dump(public, DATA / '补充检验摘要.json')
    print(json.dumps({'rows': len(null_rows), 'original_reconstruction_error': max_archived_error, 'max_w1_invariance_error': max_w1_error, 'null_energy_delta': energy_delta, 'null_correlation_delta': corr_delta, 'style_summary': agg.to_dict('records')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
