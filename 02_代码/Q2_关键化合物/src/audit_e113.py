"""Audit E113 fold checkpoints without refitting graphs or changing results."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(root):
    root=Path(root); progress=json.loads((root/'q2_progress_E113.json').read_text())
    checkpoint_root=root/'q2_graph_checkpoints_E113'; slots=sorted(p for p in checkpoint_root.iterdir() if p.is_dir())
    errors=[]; rows=[]
    for slot in slots:
        try:
            manifest=json.loads((slot/'manifest.json').read_text())
            for name,digest in manifest['files'].items():
                if _sha(slot/name)!=digest:errors.append(f'{slot.name}/{name}: hash')
            summary=pd.read_csv(slot/'summary.csv.gz');edges=pd.read_csv(slot/'edges.csv.gz');membership=pd.read_csv(slot/'membership.csv.gz')
            if set(edges.scope)!={'outer_train_fold'} or set(membership.scope)!={'outer_train_fold'}:errors.append(f'{slot.name}: scope')
            if len(membership)!=218 or membership.feature_code.nunique()!=218:errors.append(f'{slot.name}: feature coverage')
            if edges.source.eq(edges.target).any():errors.append(f'{slot.name}: self edge')
            if not np.isfinite(edges.partial_correlation.to_numpy(float)).all():errors.append(f'{slot.name}: nonfinite edge')
            if len(summary)!=1:errors.append(f'{slot.name}: summary rows')
            else:rows.append(summary.iloc[0].to_dict())
        except Exception as exc:errors.append(f'{slot.name}: {type(exc).__name__}: {exc}')
    expected=progress.get('completed_folds',0)
    if len(slots)!=expected:errors.append(f'progress says {expected}, checkpoints {len(slots)}')
    split_ids={r.get('split_id') for r in rows};
    if len(split_ids)!=len(rows):errors.append('duplicate split ids')
    report={'experiment_id':'E113','status':'partial' if progress.get('status')!='complete' else 'complete',
            'completed_folds':len(slots),'expected_folds':30,'n_features':218,'errors':errors,
            'scope_verified':'outer_train_fold','fold_summary':rows}
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.root);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':result['status'],'completed_folds':result['completed_folds'],'errors':len(result['errors'])},ensure_ascii=False));raise SystemExit(1 if result['errors'] else 0)
