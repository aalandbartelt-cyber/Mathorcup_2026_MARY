"""Hash-verified, fold-scoped graph checkpoints; never accepts descriptive graphs."""
from pathlib import Path
import hashlib,json,os
import pandas as pd

TABLES=('summary','alpha_scores','edges','membership')

def _scope(tables):
    if set(tables)!=set(TABLES):raise ValueError('checkpoint tables incomplete')
    for name in ['summary','edges','membership']:
        if 'scope' not in tables[name] or not tables[name]['scope'].eq('outer_train_fold').all():
            raise ValueError('only outer_train_fold data may enter a fold checkpoint')

def save_fold_checkpoint(root,split_id,identity,tables):
    _scope(tables)
    folder=Path(root)/str(split_id);folder.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name in TABLES:
        path=folder/(name+'.csv.gz');temp=path.with_name(path.name+'.tmp')
        tables[name].to_csv(temp,index=False,compression='gzip')
        with temp.open('rb') as stream:os.fsync(stream.fileno())
        temp.replace(path);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    temp=folder/'manifest.json.tmp'
    temp.write_text(json.dumps({'identity':identity,'files':hashes},sort_keys=True,ensure_ascii=False)+'\n')
    with temp.open('rb') as stream:os.fsync(stream.fileno())
    temp.replace(folder/'manifest.json')

def load_fold_checkpoint(root,split_id,identity):
    folder=Path(root)/str(split_id);path=folder/'manifest.json'
    if not path.exists():return None
    manifest=json.loads(path.read_text())
    if manifest['identity']!=identity:raise ValueError('checkpoint input/source identity changed')
    tables={}
    for name in TABLES:
        file=folder/(name+'.csv.gz')
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=manifest['files'].get(file.name):
            raise ValueError('checkpoint data missing or corrupt')
        tables[name]=pd.read_csv(file)
    _scope(tables)
    return tables
