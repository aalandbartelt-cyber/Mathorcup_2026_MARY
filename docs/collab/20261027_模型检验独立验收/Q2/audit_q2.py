"""Read-only recomputation of handed-over Q2 validation evidence.

Run with the mathorcup Python environment. All writes stay next to this file.
No model fitting, ranking changes, paper generation, or Git mutation occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path('/home/mary/mathorcup')
OUT = Path(__file__).resolve().parent
TREE = ROOT / '.worktrees/q2-evidence-opt'
RES = TREE / '03_结果/Q2_关键化合物'
PAPER_DATA = TREE / '05_论文/Q2_关键化合物/数据与结果'
HANDOVER = ROOT / '05_论文/Q2_关键化合物/数据与结果/validation_v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sources = [
        RES / name for name in [
            'q2_bootstrap_top3_E107.csv', 'q2_bootstrap_selection_E107.csv',
            'q2_bootstrap_ranks_E107.npz', 'q2_bootstrap_jaccard_E107.csv',
            'q2_consensus_rankings_E106.csv', 'q2_final_top3_E112.csv',
            'q2_leave_one_style_rankings_E108.csv.gz',
            'q2_leave_one_style_survival_E108.csv',
            'q2_anchor_recovery_E107.csv', 'q2_cpss_selection_E117.csv',
            'q2_null_target_permutation_E117.csv.gz',
            'q2_shadow_feature_audit_E117.csv', 'q2_final_nnls_sufficiency_E115.csv',
            'q2_run_manifest_E115_final_nnls.json',
        ]
    ] + list(HANDOVER.glob('*')) + [
        PAPER_DATA / '最终感官模型Top3删除敏感性.csv',
        ROOT / 'scripts/build_q2_validation.py',
        ROOT / 'scripts/plot_q2_validation_v2.R',
        TREE / '02_代码/Q2_关键化合物/src/run_e108_style_sensitivity.py',
        TREE / '02_代码/Q2_关键化合物/src/run_e115_final_nnls.py',
        TREE / '04_图表/Q2_关键化合物/v4/plot_q2_paper_v4.R',
        ROOT / '04_图表/Q2_关键化合物/validation_v1/图5_负对照与直接锚点.png',
    ]
    hashes_before = {str(p.relative_to(ROOT)): sha(p) for p in sources if p.is_file()}

    bootstrap = pd.read_csv(RES / 'q2_bootstrap_top3_E107.csv')
    final = pd.read_csv(RES / 'q2_final_top3_E112.csv')
    reference = pd.read_csv(RES / 'q2_consensus_rankings_E106.csv').query('consensus_rank <= 3')
    selection = pd.read_csv(RES / 'q2_bootstrap_selection_E107.csv')
    leave = pd.read_csv(RES / 'q2_leave_one_style_rankings_E108.csv.gz')
    assert leave.shape == (22 * 50 * 218, 13)
    assert not leave[['omitted_style', 'target_code', 'feature_code']].duplicated().any()
    leave_frequency = (
        leave.assign(selected=leave['consensus_rank'].le(3))
        .groupby(['target_code', 'feature_code'], as_index=False)
        .agg(leave_frequency=('selected', 'mean'))
    )
    rank_npz = np.load(RES / 'q2_bootstrap_ranks_E107.npz', allow_pickle=False)
    assert rank_npz['consensus_ranks'].shape == (500, 218, 50)
    selection_from_ranks = (rank_npz['consensus_ranks'] <= 3).mean(axis=0)
    lookup = {
        (str(t), str(f)): float(selection_from_ranks[i, j])
        for i, f in enumerate(rank_npz['feature_codes'])
        for j, t in enumerate(rank_npz['target_codes'])
    }
    bootstrap_max_frequency_error = max(
        abs(float(r.selection_frequency) - lookup[(r.target_code, r.feature_code)])
        for r in bootstrap.itertuples()
    )

    sets = {}
    scope_rows = []
    for label, frame in [
        ('总关联共识参考集', reference),
        ('分层bootstrap频率选出集', bootstrap),
        ('四路融合最终Top-3', final),
    ]:
        pairs = frame[['target_code', 'feature_code']]
        joined = pairs.merge(selection[['target_code', 'feature_code', 'selection_frequency']],
                             on=['target_code', 'feature_code'], validate='one_to_one')
        joined = joined.merge(leave_frequency, on=['target_code', 'feature_code'],
                              validate='one_to_one')
        assert len(joined) == 150
        sets[label] = set(map(tuple, pairs.to_numpy()))
        scope_rows.append({
            'scope': label,
            'n_pairs': len(joined),
            'bootstrap_frequency_median': float(joined.selection_frequency.median()),
            'n_bootstrap_frequency_ge_0_60': int(joined.selection_frequency.ge(.6).sum()),
            'leave_style_frequency_mean': float(joined.leave_frequency.mean()),
            'leave_style_frequency_median': float(joined.leave_frequency.median()),
        })
        joined.to_csv(OUT / f'{label}_逐对频率.csv', index=False)

    overlap = len(sets['分层bootstrap频率选出集'] & sets['总关联共识参考集'])
    differences = bootstrap[['target_code','feature_code']].merge(
        reference[['target_code','feature_code']], how='outer',
        on=['target_code','feature_code'], indicator=True)
    differences.to_csv(OUT / 'bootstrap与总关联参考集_集合差异.csv', index=False)
    pd.DataFrame(scope_rows).to_csv(OUT / '三套稳定性_独立复算.csv', index=False)

    real = pd.read_csv(RES / 'q2_cpss_selection_E117.csv')
    null = pd.read_csv(RES / 'q2_null_target_permutation_E117.csv.gz')
    negative = [{
        'threshold': threshold,
        'real_relation_count': int(real.selection_probability.ge(threshold).sum()),
        'null_relation_count': int(null.null_selection_probability.ge(threshold).sum()),
    } for threshold in [.6, .7, .8, .9]]
    shadow = pd.read_csv(RES / 'q2_shadow_feature_audit_E117.csv')
    assert len(shadow) == 2000
    assert (shadow.real_selected + shadow.shadow_selected).eq(3).all()
    shadow_result = {
        'n_records': len(shadow), 'total_slots': int(3 * len(shadow)),
        'real_selections': int(shadow.real_selected.sum()),
        'shadow_selections': int(shadow.shadow_selected.sum()),
        'real_per_record': float(shadow.real_selected.mean()),
        'shadow_per_record': float(shadow.shadow_selected.mean()),
        'shadow_slot_fraction': float(shadow.shadow_selected.sum() / (3 * len(shadow))),
    }

    anchor = pd.read_csv(RES / 'q2_anchor_recovery_E107.csv')
    recovered = final.loc[final.direct_anchor_match,
                          ['target_code','feature_code','consensus_rank','bootstrap_rank']]
    expected_anchor = {'dms': 29, 'diacetyl': 92, 't2n': 61, 'co2': 4}
    anchor_checks = {}
    for target, expected in expected_anchor.items():
        row = anchor.loc[anchor.target_code.eq(target)].iloc[0]
        direct = selection.loc[selection.target_code.eq(target) & selection.feature_code.eq(row.direct_anchor)].iloc[0]
        anchor_checks[target] = {'bootstrap_rank': int(direct.bootstrap_rank),
                                 'expected': expected,
                                 'matches': int(direct.bootstrap_rank) == expected}

    raw_delete = pd.read_csv(RES / 'q2_final_nnls_sufficiency_E115.csv')
    handed_delete = pd.read_csv(PAPER_DATA / '最终感官模型Top3删除敏感性.csv')
    calc_delta = raw_delete.masked_mae - raw_delete.baseline_mae
    assert np.allclose(calc_delta, raw_delete.delta_mae_masked_minus_baseline, atol=1e-14)
    aggregate = raw_delete.groupby('target_code', as_index=False).agg(
        mean_delta_mae=('delta_mae_masked_minus_baseline','mean'),
        baseline_mae=('baseline_mae','mean'), n_folds=('fold','size'))
    aggregate['relative_change_pct'] = 100 * aggregate.mean_delta_mae / aggregate.baseline_mae
    aggregate.to_csv(OUT / '折内前三字段遮蔽_逐目标复算.csv', index=False)
    final_sets = final.groupby('target_code').feature_code.apply(set).to_dict()
    intersections = [len(set(row.top3_features.split('|')) & final_sets[row.target_code])
                     for row in raw_delete.itertuples()]
    deletion = {
        'rows': len(raw_delete),
        'mean_of_50_target_means': float(aggregate.mean_delta_mae.mean()),
        'median_of_50_target_means': float(aggregate.mean_delta_mae.median()),
        'n_positive_target_means': int(aggregate.mean_delta_mae.gt(0).sum()),
        'top3_source_values': sorted(raw_delete.top3_source.unique().tolist()),
        'evaluation_scope_values': sorted(raw_delete.evaluation_scope.unique().tolist()),
        'same_field_set_as_formal_top3_rows': int(np.count_nonzero(np.array(intersections) == 3)),
        'intersection_with_formal_top3_counts': {str(k): intersections.count(k) for k in range(4)},
        'handed_1500_row_delta_max_error': float(np.max(np.abs(
            handed_delete.delta_mae_masked_minus_baseline - raw_delete.delta_mae_masked_minus_baseline))),
        'largest_effects': aggregate.sort_values('mean_delta_mae', ascending=False).head(3).to_dict('records'),
    }
    numerical_checks = [
        {'item':'bootstrap中位频率','observed':scope_rows[1]['bootstrap_frequency_median'],'expected':.428},
        {'item':'bootstrap频率≥0.60条数','observed':scope_rows[1]['n_bootstrap_frequency_ge_0_60'],'expected':43},
        {'item':'最终Top3中位频率','observed':scope_rows[2]['bootstrap_frequency_median'],'expected':.395},
        {'item':'最终Top3频率≥0.60条数','observed':scope_rows[2]['n_bootstrap_frequency_ge_0_60'],'expected':40},
        {'item':'最终Top3留一风格均值','observed':scope_rows[2]['leave_style_frequency_mean'],'expected':.608},
        {'item':'最终Top3留一风格中位','observed':scope_rows[2]['leave_style_frequency_median'],'expected':.659},
    ]
    report = {
        'status':'BLOCKED_source_scope_mismatch',
        'no_model_fit':True,
        'bootstrap_raw_ranks_frequency_max_error':bootstrap_max_frequency_error,
        'stability_scopes':scope_rows,
        'bootstrap_reference_overlap':overlap,
        'bootstrap_reference_only_each':150-overlap,
        'negative_controls':negative,
        'shadow':shadow_result,
        'anchors':anchor_checks,
        'formal_anchor_recoveries':recovered.to_dict('records'),
        'deletion':deletion,
        'numerical_checks':numerical_checks,
        'blocking_discrepancies':[
            {
                'item':'bootstrap候选集留一风格均值',
                'handover':.641,
                'recomputed':scope_rows[1]['leave_style_frequency_mean'],
                'reason':'0.640606来自总关联共识参考集，该150条与bootstrap候选集仅重合113条。',
                'source_code':'scripts/build_q2_validation.py:79 硬编码0.641；run_e108_style_sensitivity.py:261、343–360明确reference为总关联共识参考集。',
            },
            {
                'item':'删除敏感性的被遮蔽字段对象',
                'handover':'删除Top-3；若理解为正式答案Top-3则对象不符',
                'recomputed':'同折训练集条件效应排名前三字段',
                'reason':'1500条中仅1条遮蔽字段集合等于正式Top-3；数字不能用于声称正式Top-3删除效应。',
                'source_code':'run_e115_final_nnls.py:76、94–95；top3_source字段',
            },
        ],
        'figure_source_checks':{
            'figure2_short_label':'图正则EN',
            'figure2_source':str((TREE/'04_图表/Q2_关键化合物/v4/plot_q2_paper_v4.R').relative_to(ROOT)),
            'figure5_source':'scripts/plot_q2_validation_v2.R',
            'figure5_issue':'左面板把四个阈值的置换关系计数0与每条影子记录平均入选次数并列，单位不同；建议保留精确分母与阈值说明，避免作同尺度高度比较。',
            'figure5_anchor_status':'CO2在bootstrap第4名但在最终Top3中，属于可同时成立的两个排名口径。',
        },
        'source_sha256':hashes_before,
    }
    hashes_after = {str(p.relative_to(ROOT)): sha(p) for p in sources if p.is_file()}
    assert hashes_before == hashes_after, 'source changed while audit ran'
    report['all_source_hashes_unchanged'] = True
    (OUT/'核验结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','stability_scopes','bootstrap_reference_overlap','negative_controls','shadow','deletion']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
