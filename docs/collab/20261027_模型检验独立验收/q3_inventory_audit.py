"""Read-only Q3 inventory. Writes audit provenance only, never model artifacts."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

import numpy as np
import pandas as pd
import pymupdf
from lxml import etree

ROOT = Path('/home/mary/mathorcup')
WT = ROOT / '.worktrees/q3-foundation'
RESULT = WT / '03_结果/Q3_风格生成'
OUT = Path(__file__).parent
NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run():
    sources = []
    relevant = [
        'q3_metrics_by_repeat_E211.csv', 'q3_all_method_summary_E211.csv',
        'q3_generated_profiles_E205.csv.gz', 'q3_primary_candidates_E207.csv',
        'q3_migration_summary_E204.csv', 'q3_migration_proxy_audit_E205.csv',
        'q3_proxy_margin_sensitivity_E207.csv', 'q3_frozen_candidate_sensitivity_E207.csv',
        'q3_e204_audit_summary.json', 'q3_migration_decision_E204.json',
        'q3_sensory_decision_E205.json', 'q3_e205_audit_manifest.json',
        'q3_e205b_audit_manifest.json', 'q3_run_manifest_E207.json',
        'q3_run_manifest_E202.json', 'q3_distribution_metrics_E202.csv',
        'q3_e202_robust_sensitivity.csv', 'q3_e202_robust_by_repeat.csv',
        'q3_e202_robust_summary.csv', 'q3_e203_replacement_decision.json',
        'q3_e203_vs_copula_robust_comparison.csv', 'q3_baseline_summary_E201.csv',
        'q3_replacement_decision_E208.json', 'q3_replacement_decision_E210.json',
        'q3_replacement_decision_E211.json', 'q3_replacement_decision_E213.json',
        'q3_replacement_decision_E214.json', 'q3_outer_profile_bias_E205.csv',
        'q3_outer_profile_bias_by_repeat_E205.csv', 'q3_metrics_by_repeat_E203.csv',
        'q3_e203_robust_by_repeat.csv', 'q3_metrics_by_repeat_E210.csv',
        'q3_candidate_lock_E209.json', 'q3_candidate_lock_E217.json',
        'q3_external_data_eligibility_E212.json', 'q3_external_directions_E212.csv',
    ]
    paths = [RESULT / n for n in relevant]
    paths += [p for p in (WT / '05_论文/Q3_风格生成').iterdir() if p.is_file()]
    paths += [WT / '04_图表/Q3_风格生成/draft_v2' / n for n in [
        'Q3_图清单_draft_v2.csv', 'source_data_manifest.json', 'figure_contract.md',
        'prepare_source_data.py', 'plot_q3_figures.R',
        '图1_生成质量与多样性.png', '图2_三风格感官画像.png', '图3_迁移收益与审计状态.png',
    ]]
    paths += [WT / '02_代码/Q3_风格生成/src' / n for n in [
        'q3_evaluation.py', 'joint_generation.py', 'audit_e202.py',
        'run_e202_gaussian_baselines.py', 'gaussian_baselines.py',
    ]]
    for p in sorted(set(paths)):
        row = {'path': str(p.relative_to(ROOT)), 'exists': p.exists()}
        if p.exists():
            row.update(bytes=p.stat().st_size, sha256=sha(p))
            if p.suffix == '.csv' or p.name.endswith('.csv.gz'):
                d = pd.read_csv(p)
                row.update(rows=len(d), columns=d.columns.tolist())
            elif p.suffix == '.pdf':
                with pymupdf.open(p) as pdf:
                    row['pages'] = len(pdf)
            elif p.suffix == '.docx':
                with ZipFile(p) as z:
                    x = etree.fromstring(z.read('word/document.xml'))
                row.update(
                    omml=len(x.xpath('.//m:oMath', namespaces=NS)),
                    drawings=len(x.xpath('.//w:drawing', namespaces=NS)),
                    tables=len(x.xpath('.//w:tbl', namespaces=NS)),
                )
        sources.append(row)

    metrics = pd.read_csv(RESULT / 'q3_metrics_by_repeat_E211.csv')
    copula = metrics[(metrics.method == 'style_copula_shrunk') & (metrics.metric_scale == 'robust')]
    measures = ['mean_w1', 'energy_v', 'coverage', 'spearman_pair_rmse', 'log_spread_error']
    profiles = pd.read_csv(RESULT / 'q3_generated_profiles_E205.csv.gz')
    migration = pd.read_csv(RESULT / 'q3_migration_summary_E204.csv')
    primary = pd.read_csv(RESULT / 'q3_primary_candidates_E207.csv')
    margins = pd.read_csv(RESULT / 'q3_proxy_margin_sensitivity_E207.csv')
    current = margins[margins.was_main_candidate_eligible]
    nfields = ['prediction_z', 'grade_0_6', 'model_fold_sd', 'audit_prediction_z', 'predictor_support_fraction']
    report = {
        'date': '2026-10-27',
        'scope': 'read_only_inventory_and_frozen_artifact_recalculation',
        'training_performed': False, 'new_generation_performed': False,
        'formal_files_modified': False, 'git_commit_performed': False,
        'copula_robust_style_mean': copula.groupby('style')[measures].mean().reset_index().to_dict('records'),
        'copula_robust_repeat_statistics': copula.groupby('repeat')[measures].mean().agg(['mean', 'std']).to_dict(),
        'profiles': {
            'rows': len(profiles), 'targets': int(profiles.target_code.nunique()),
            'sample_counts_by_style': profiles.groupby('style').sample_id.nunique().astype(int).to_dict(),
            'finite_by_field': np.isfinite(profiles[nfields]).all().to_dict(),
            'grade_range': [float(profiles.grade_0_6.min()), float(profiles.grade_0_6.max())],
        },
        'direct_stage': {
            'rows': len(migration), 'available': int(migration.candidate_available.sum()),
            'pareto_marked': int(migration.pareto_front.sum()),
            'status_counts': migration.status.value_counts().astype(int).to_dict(),
            'unit': 'source_target_scenario_strategy_path_record_not_independent_beers',
            'pareto_is_parallel_label_not_serial_funnel_stage': True,
        },
        'current_primary': {
            'rows': len(primary), 'sources': int(primary.beer.nunique()),
            'available': int(primary.candidate_available.sum()),
            'status_counts': primary.audited_status.value_counts().astype(int).to_dict(),
            'eligible_rows': primary[primary.eligible][['beer', 'target_style', 'relative_target_gain', 'source_change_rms']].to_dict('records'),
            'physical_safety_certified_any': bool(primary.physical_safety_certified.any()),
            'margin_sensitivity_for_main_eligible': current.groupby('margin_z').both_proxy_nonworsening.sum().astype(int).to_dict(),
        },
        'sources': sources,
    }
    (OUT / 'Q3_sources.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'sources': len(sources), 'missing': [x['path'] for x in sources if not x['exists']],
        'profile_rows': len(profiles), 'current_primary_status': report['current_primary']['status_counts'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    run()
