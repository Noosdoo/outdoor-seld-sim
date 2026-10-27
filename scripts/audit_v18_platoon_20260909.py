"""Isolated, read-only-source reproduction and paired scene cluster bootstrap.

Outputs only to out/independent_v18_platoon_2026-09-09. No pipeline imports
execute main(), no fetch/repair/submission, no published score tables overwritten.
"""
from pathlib import Path
import csv, hashlib, importlib.util, json, re, subprocess, sys
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/independent_v18_platoon_2026-09-09'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    return list(csv.DictReader(Path(path).open(encoding='utf-8-sig', newline='')))

def clusters(plan, paired=True):
    groups = defaultdict(list)
    if len({r['clip_id'] for r in plan}) != len(plan):
        raise ValueError('duplicate clip')
    for r in plan:
        if not r.get('seed') or not r.get('scenario'):
            raise ValueError('missing scene identity')
        groups[(r['scenario'], r['seed'])].append(r)
    result = []
    for key, rr in sorted(groups.items()):
        if paired and (len(rr) != 2 or {r['h_copy'] for r in rr} != {'1', '2'}):
            raise ValueError(f'incomplete height pair: {key}')
        if not paired and len(rr) != 1:
            raise ValueError(f'non-independent clip identity: {key}')
        result.append([r['clip_id'] for r in sorted(rr, key=lambda r:r['clip_id'])])
    return result

def bootstrap(a, b, draws=10000, seed=20260909):
    """Each Nx2 array = cluster successes/denominator; delta is b-a in pp.
    Unstratified scene resampling; both models use exactly the same scene draws.
    Ratios of pooled counts, never averages of per-clip percentages.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(a) == 0:
        raise ValueError('paired Nx2 counts required')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError('invalid counts')
    if not np.array_equal(a[:,1], b[:,1]) or (a[:,0] > a[:,1]).any() or (b[:,0] > b[:,1]).any():
        raise ValueError('same GT denominator required')
    if a[:,1].sum() == 0:
        raise ValueError('undefined zero-denominator estimand')
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 128):
        ii = rng.integers(len(a), size=(min(128, draws-start), len(a)))
        aa, bb = a[ii].sum(axis=1), b[ii].sum(axis=1)
        good = aa[:,1] > 0
        samples.extend((100*(bb[good,0]-aa[good,0])/aa[good,1]).tolist())
    if not samples:
        raise ValueError('all bootstrap denominators zero')
    point = 100*(b[:,0].sum()-a[:,0].sum())/a[:,1].sum()
    return dict(delta_pp=float(point), ci95_pp=np.quantile(samples,[.025,.975]).tolist(),
                draws=draws, valid_draws=len(samples), seed=seed, clusters=len(a),
                denominator=int(a[:,1].sum()))

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    S = module('audit_score', ROOT/'scripts/_score_unified.py')
    I = module('audit_integrity', ROOT/'server_sde/_prediction_integrity.py')
    p18 = ROOT/'out/dataset_outdoor_siren_v18/plan/assignment_v18.csv'
    p19 = ROOT/'out/dataset_outdoor_siren_v19/plan/assignment_v19_platoon.csv'
    m12 = ROOT/'out/dataset_outdoor_siren_v12/plan/manifest_fold2_val.csv'
    ext = rows(ROOT/'out/dataset_outdoor_siren_v12/plan/assignment_v12ext.csv')
    byid = {r['clip_id']:r for r in ext}
    specs = {
        'self': ([r for r in rows(p18) if r['split']=='fold2'], ROOT/'out/dataset_outdoor_siren_v18/metadata_dist', True),
        'new': ([r for r in rows(p19) if r['split']=='fold2'], ROOT/'out/dataset_outdoor_siren_v19/metadata_dist', True),
        'cross': ([dict(r, scenario=byid.get(r['clip_id'],{}).get('scenario','v11core'), seed=r['clip_id']) for r in rows(m12)], ROOT/'out/dataset_outdoor_siren_v12/metadata_dist_h', False),
    }
    evidence = dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    inputs={}, datasets={}, tables={}, comparisons=[], provenance={})
    for p in [p18,p19,m12,ROOT/'out/dataset_outdoor_siren_v12/plan/assignment_v12ext.csv',ROOT/'out/notify_v43_sweep/winner.json']:
        evidence['inputs'][str(p.relative_to(ROOT))] = sha(p)
    for p in (ROOT/'scripts').glob('*.py'):
        if p.name in ['_score_unified.py','_notify_v43_attrib.py','_notify_v42_diag.py','step12_notify_v43.py','step12_notify_v4_ttc.py','step12_notify_v9b_hold.py','audit_v18_platoon_20260909.py']:
            evidence['inputs'][str(p.relative_to(ROOT))] = sha(p)
    groupmap = {}
    for name,(rr,gt,pair) in specs.items():
        groupmap[name] = clusters(rr,pair)
        gh = hashlib.sha256()
        for r in sorted(rr,key=lambda x:x['clip_id']):
            p = gt/(r['clip_id']+'.csv')
            gh.update((r['clip_id']+':'+sha(p)+'\n').encode())
        evidence['datasets'][name] = dict(clips=len(rr),scenes=len(groupmap[name]),gt_manifest_sha256=gh.hexdigest(),groups=groupmap[name])
    stats = {}
    for model in ['v17b','v18','v18bundle','v19platoon']:
        for epoch in [99,119,139,149]:
            for split in ['self','cross']:
                if model=='v18bundle' and split=='cross': continue
                actual = 'v18' if model=='v18bundle' else model
                folder = 'v19' if model in ['v18bundle','v19platoon'] else model
                suffix = ('v19platooncausal' if model=='v18bundle' else 'selfcausal') if split=='self' else 'v12causal'
                p = ROOT/f'out/{folder}/C/infer_{actual}ft_e{epoch:03d}_{suffix}.csv'
                key = f'{model}_e{epoch:03d}_{split}'
                mp, dp = p.with_suffix('.run_meta.json'), p.with_suffix('.done.json')
                meta = json.loads(mp.read_text(encoding='utf-8'))
                I.verify_download(p,len(meta['clips']),dp,mp)
                evidence['provenance'][key] = dict(csv=str(p.relative_to(ROOT)),sha256=sha(p),meta_sha256=sha(mp),done_sha256=sha(dp),ckpt_sha256=meta['ckpt_sha256'],code_sha256=meta['code_sha256'])
                pred = S.load_pred_rows(p,False)
                for dataset in ([split,'new'] if split=='self' and model in ['v18bundle','v19platoon'] else [split]):
                    rr,gt,_ = specs[dataset]
                    ids={Path(c).stem for c in meta['clips']}
                    if not {r['clip_id'] for r in rr} <= ids: raise ValueError('manifest not covered')
                    per = {r['clip_id']:S.clip_stats(pred,gt,r['clip_id'],40) for r in rr}
                    statkey = f'{model}_e{epoch:03d}_{dataset}'
                    agg = S.aggregate(list(per.values()))
                    evidence['tables'][statkey] = agg
                    stats[statkey] = {c:{k:v for k,v in s.items() if k not in ['rel','close_est','leads']} for c,s in per.items()}
                    print(statkey,agg['strong'],agg['dist_err'],flush=True)
                    if dataset=='self':
                        ref = [r for r in rr if int(re.search(r'mix(\d+)$',r['clip_id'])[1])<=9000]
                        evidence['tables'][statkey+'_ref'] = S.aggregate([per[r['clip_id']] for r in ref])
                        for lo,hi in [(1.4,1.6),(1.6,1.85),(1.85,2.1)]:
                            sub=[per[r['clip_id']] for r in ref if lo<=float(r['mic_z'])<hi or hi==2.1 and float(r['mic_z'])==hi]
                            evidence['tables'][statkey+f'_ref_{lo}_{hi}'] = S.aggregate(sub)
                del pred
    (OUT/'per_clip_counts.json').write_text(json.dumps(stats),encoding='utf-8')
    published = [('v18_self_all','self','v17b','v18'),('v18_cross_v12h','cross','v17b','v18'),('v19_platoon_self_all','self','v18bundle','v19platoon'),('v19_platoon_new','new','v18bundle','v19platoon'),('v19_platoon_cross_v12h','cross','v18','v19platoon'),('v18_on_v15val','self','v17b','v18'),('v19_platoon_on_v15val','self','v18bundle','v19platoon')]
    checks=[]
    for title,dataset,base,candidate in published:
        original=ROOT/f'out/audit_rescore/{title}.md'
        evidence['inputs'][str(original.relative_to(ROOT))]=sha(original)
        old=original.read_text(encoding='utf-8')
        lines=[S.HEADER,S.SEP]
        for model in [base,candidate]:
            labelmodel='v18' if model=='v18bundle' else model
            for epoch in [99,119,139,149]:
                key=f'{model}_e{epoch:03d}_{dataset}'
                choices=[('',key)] if 'on_v15val' not in title else [('',key+'_ref')]+[(f' {lo}〜{hi} m',key+f'_ref_{lo}_{hi}') for lo,hi in [(1.4,1.6),(1.6,1.85),(1.85,2.1)]]
                for suffix,kk in choices:
                    line=S.fmt(f'{labelmodel}_e{epoch:03d}'+suffix,evidence['tables'][kk]); lines.append(line)
                    checks.append(dict(table=title,label=f'{labelmodel}_e{epoch:03d}'+suffix,exact_row_match=line in old))
        (OUT/(title+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        if 'on_v15val' in title: continue
        for epoch in [99,119,139,149]:
            aa=stats[f'{base}_e{epoch:03d}_{dataset}']; bb=stats[f'{candidate}_e{epoch:03d}_{dataset}']
            for metric in ['strong','cap_all']:
                def counts(ss):
                    return [[sum(ss[c]['n_strong' if metric=='strong' else 'n_close_cap'] for c in group),sum(ss[c]['crit'][1] if metric=='strong' else ss[c]['n_close_gt'] for c in group)] for group in groupmap[dataset]]
                result=bootstrap(counts(aa),counts(bb))
                evidence['comparisons'].append(dict(table=title,epoch=epoch,metric=metric,**result))
    evidence['row_checks']=checks
    evidence['all_rows_match']=all(c['exact_row_match'] for c in checks)
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE rows',len(checks),'match',evidence['all_rows_match'],flush=True)

if __name__=='__main__': run()
