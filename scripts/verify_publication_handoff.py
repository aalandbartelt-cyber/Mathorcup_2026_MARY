"""Read-only verification from a paper handoff root. Does not train a model."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args(); root=args.root
    lines=(root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    for line in lines:
        digest,name=line.split('  ',1)
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    result={'hashed_files':len(lines),'hash_mismatches':0}
    data=root/'数据与结果'
    if (data/'final_model_oof.npz').exists():
        with np.load(data/'final_model_oof.npz',allow_pickle=False) as z:
            y=z['y_true'];p=z['prediction']
            assert p.shape==(10,250,50) and np.isfinite(p).all()
            variance=((y-y.mean(0))**2).sum(0)
            r2=(1-((p-y)**2).sum(1)/variance).mean(1)
            mae=np.abs(p-y).mean((1,2))
            result.update(r2_repeat_mean=float(r2.mean()),r2_repeat_sd=float(r2.std(ddof=1)),
                          mae_repeat_mean=float(mae.mean()),mae_repeat_sd=float(mae.std(ddof=1)))
    elif (data/'关键化合物Top3_当前版.csv').exists():
        table=pd.read_csv(data/'关键化合物Top3_当前版.csv')
        assert len(table)==150 and table.target_code.nunique()==50
        assert table.groupby('target_code').size().eq(3).all()
        mask=pd.read_csv(data/'最终感官模型_折内条件字段遮蔽敏感性.csv')
        target_means=mask.groupby('target_code').delta_mae_masked_minus_baseline.mean()
        result.update(top3_relations=len(table),masked_mean_delta_mae=float(target_means.mean()),
                      masked_median_target_mean=float(target_means.median()),
                      masked_targets_with_increase=int(target_means.gt(0).sum()))
    else:
        val=data/'validation_v1'
        original=pd.read_csv(val/'固定生成谱诊断.csv')
        null=pd.read_csv(val/'独立置换诊断.csv')
        assert len(original)==450 and len(null)==2250
        assert null.delta_mean_w1.abs().max()<=1e-12
        grouped=null.groupby(['style','repeat'])[['delta_energy_v','delta_spearman_pair_rmse','delta_coverage']].mean()
        result.update(frozen_groups=len(original),permutation_evaluations=len(null),
                      marginal_w1_change=float(null.delta_mean_w1.abs().max()),
                      null_energy_delta=float(grouped.delta_energy_v.mean()),
                      null_correlation_delta=float(grouped.delta_spearman_pair_rmse.mean()))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
