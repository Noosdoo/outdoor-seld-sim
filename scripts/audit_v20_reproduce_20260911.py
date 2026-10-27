"""Independent local CPU reproduction; shared sources and server are read-only."""
from pathlib import Path
import csv,json,hashlib,importlib.util,sys,pickle,gzip,concurrent.futures,subprocess,re,datetime,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'out/independent_v20_2026-09-11'
EPS=[99,119,139,149];SCENES=['fast','turn','platoon','static']
def module(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def dataset(ds):
    if ds=='self':p=ROOT/'out/dataset_outdoor_siren_v18/plan/assignment_v18.csv';gt=ROOT/'out/dataset_outdoor_siren_v18/metadata_dist'
    elif ds=='cross':p=ROOT/'out/dataset_outdoor_siren_v12/plan/manifest_fold2_val.csv';gt=ROOT/'out/dataset_outdoor_siren_v12/metadata_dist_h'
    else:p=ROOT/f'out/dataset_outdoor_siren_v19/plan/assignment_v19_{ds}.csv';gt=ROOT/'out/dataset_outdoor_siren_v19/metadata_dist'
    rr=[r for r in rows(p) if r.get('split','fold2')=='fold2'];assert len(rr)==len({r['clip_id'] for r in rr});assert all((gt/(r['clip_id']+'.csv')).exists() for r in rr)
    return p,gt,rr
def tasks():
    ts=[]
    for m in ['v20a','v20b']:
        scenes=SCENES[:3] if m=='v20a' else SCENES
        for e in EPS:
            for model,suffix,dss,count in [(m,'selfcausal',['self']+scenes,3600+300*len(scenes)),(m,'v12causal',['cross'],1800),('v18',m+'causal',['self']+scenes,3600+300*len(scenes))]:
                name=f'infer_{model}ft_e{e:03d}_{suffix}';ts.append(dict(name=name,csv=f'out/v20/C/{name}.csv',expected=count,datasets=dss))
    for e in EPS:
        name=f'infer_v18ft_e{e:03d}_v12causal';ts.append(dict(name=name,csv=f'out/v18/C/{name}.csv',expected=1800,datasets=['cross']))
        for sc in SCENES[:3]:
            name=f'infer_v19{sc}ft_e{e:03d}_selfcausal';ts.append(dict(name=name,csv=f'out/v19/C/{name}.csv',expected=3900,datasets=[sc]))
    return ts
def work(t):
    out=OUT/'runs'/(t['name']+'.json');p=ROOT/t['csv']
    if out.exists():
        old=json.loads(out.read_text(encoding='utf-8'));assert old['provenance']['csv_sha256']==sha(p)
        return t['name']+' REUSED_CURRENT_AUDIT'
    S=module('audit20_score',ROOT/'scripts/_score_unified.py');I=module('audit20_integrity',ROOT/'server_sde/_prediction_integrity.py')
    mp=p.with_suffix('.run_meta.json');dp=p.with_suffix('.done.json');sp=p.with_suffix('.sha256')
    meta=json.loads(mp.read_text(encoding='utf-8'));done=json.loads(dp.read_text(encoding='utf-8'))
    I.verify_download(p,t['expected'],dp,mp);csvhash=sha(p)
    if sp.exists():assert re.search(r'[a-fA-F0-9]{64}',sp.read_text(encoding='utf-8-sig')).group(0).lower()==csvhash
    else:raise FileNotFoundError(sp)
    ev=dict(task=t,provenance=dict(csv_sha256=csvhash,meta_sha256=sha(mp),done_sha256=sha(dp),meta=meta,done=done),tables={},warnings={})
    pred=S.load_pred_rows(p,False);stats={}
    for ds in t['datasets']:
        pp,gt,rr=dataset(ds);clips=[r['clip_id'] for r in rr];assert set(clips)<={Path(c).stem for c in meta['clips']}
        per={c:S.clip_stats(pred,gt,c,40) for c in clips};stats[ds]=per;ev['tables'][ds]=S.aggregate(list(per.values()))
        if ds=='self':
            ref=[r for r in rr if int(r['clip_id'].split('mix')[-1])<=9000];ev['tables']['ref']=S.aggregate([per[r['clip_id']] for r in ref])
            for lo,hi in [(1.4,1.6),(1.6,1.85),(1.85,2.1)]:ev['tables'][f'band_{lo}_{hi}']=S.aggregate([per[r['clip_id']] for r in ref if lo<=float(r['mic_z'])<hi or hi==2.1 and float(r['mic_z'])==hi])
        if ds=='static':
            W=module('audit20_warn',ROOT/'scripts/_warn_episode_eval.py');W.S.load_pred_rows=lambda *a,**kw:pred
            assert not ({0,1}&set(S.v4.DIST_CLASSES))
            W.fires_for=lambda frames,cls:sorted((k,az) for k,c,az in S.H.warn_fires({k:[(c,az,el) for c,az,el,d in vv] for k,vv in frames.items()},True) if c==cls)
            for cls in [0,1]:
                groups={c:'有' if W.gt_episodes(gt,c,cls) else '無' for c in clips}
                ev['warnings'][str(cls)]=W.evaluate(p,clips,gt,cls,groups.__getitem__,40,5,30.)
    statpath=OUT/'stats'/(t['name']+'.pkl.gz');statpath.parent.mkdir(exist_ok=True)
    with gzip.open(statpath,'wb') as f:pickle.dump(stats,f,protocol=5)
    ev['stats_sha256']=sha(statpath);dump(out,ev)
    return t['name']+' OK'
def snapshot():
    OUT.mkdir(parents=True,exist_ok=True);snap=dict(time=datetime.datetime.now().astimezone().isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={},datasets={})
    for folder in ['scripts','server_sde']:
        for p in (ROOT/folder).iterdir():
            if p.is_file() and p.suffix in ['.py','.sh','.sbatch']:snap['sources'][str(p.relative_to(ROOT))]=sha(p)
    for ds in ['self','cross']+SCENES:
        pp,gt,rr=dataset(ds);snap['datasets'][ds]=dict(plan_sha256=sha(pp),gt_hashes={r['clip_id']:sha(gt/(r['clip_id']+'.csv')) for r in rr})
    snap['winner']=sha(ROOT/'out/notify_v43_sweep/winner.json');dump(OUT/'snapshot.json',snap)
def main():
    if not (OUT/'snapshot.json').exists():snapshot()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for x in pool.map(work,tasks()):print(x,flush=True)
    print('ALL_40_COMPLETE',flush=True)
if __name__=='__main__':main()
