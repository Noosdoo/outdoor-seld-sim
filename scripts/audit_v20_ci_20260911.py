"""Independent scene-paired bootstrap incl. exact pooled median; CPU only."""
from audit_v20_reproduce_20260911 import *
import numpy as np
from collections import defaultdict
SEED=20260909;DRAWS=10000
def weighted_medians(values,labels,weights):
    """Exact resampled pooled median without expanding/sorting each bootstrap.
    Tree descent finds both central order statistics using cluster multiplicities.
    """
    values=np.asarray(values,float);labels=np.asarray(labels,int);weights=np.asarray(weights)
    if not len(values):return np.full(len(weights),np.nan)
    order=np.argsort(values,kind='stable');v=values[order];labs=labels[order];g=weights.shape[1]
    total=weights@np.bincount(labs,minlength=g);valid=np.flatnonzero(total>0)
    def kth(target):
        out=np.full(len(weights),np.nan)
        def visit(lo,hi,ix,prefix):
            if not len(ix):return
            if lo==hi or v[lo]==v[hi]:out[ix]=v[lo];return
            mid=(lo+hi)//2;le=prefix+np.bincount(labs[lo:mid+1],minlength=g)
            lower=(weights[ix]@le)>=target[ix]
            visit(lo,mid,ix[lower],prefix);visit(mid+1,hi,ix[~lower],le)
        visit(0,len(v)-1,valid,np.zeros(g,dtype=np.int64));return out
    return .5*(kth((total+1)//2)+kth(total//2+1))
def grouping(ds):
    _,_,rr=dataset(ds);groups=defaultdict(list)
    for r in rr:groups[(r['scenario'],r['seed']) if ds!='cross' else (r['clip_id'],)].append(r)
    result=[]
    for key,rs in sorted(groups.items()):
        if ds!='cross':assert len(rs)==2 and {r['h_copy'] for r in rs}=={'1','2'},key
        else:assert len(rs)==1
        result.append(sorted(r['clip_id'] for r in rs))
    return result
def multiplicities(n,draws=DRAWS,seed=SEED):
    rng=np.random.default_rng(seed);w=np.empty((draws,n),dtype=np.int16)
    for start in range(0,draws,128):
        picks=rng.integers(n,size=(min(128,draws-start),n))
        for j,idx in enumerate(picks):w[start+j]=np.bincount(idx,minlength=n)
    return w
def statistics(per,groups,weights):
    def totals(fn):return np.array([sum(fn(per[c]) for c in group) for group in groups])
    denom=totals(lambda s:s['crit'][1]);close=totals(lambda s:s['n_close_gt']);safe=totals(lambda s:s['safe'][1])
    defs={'strong':(totals(lambda s:s['n_strong']),denom),'crit':(totals(lambda s:s['crit'][0]),denom),'safe':(totals(lambda s:s['safe'][0]),safe),'cap_all':(totals(lambda s:s['n_close_cap']),close),'det_close':(totals(lambda s:s['n_close_det']),close)}
    out={};point={}
    for k,(num,den) in defs.items():
        d=weights@den;n=weights@num;out[k]=np.divide(100*n,d,out=np.full(len(d),np.nan),where=d>0);point[k]=100*num.sum()/den.sum() if den.sum() else float('nan')
    vals=[];labels=[]
    for i,group in enumerate(groups):
        for c in group:vals.extend(per[c]['rel']);labels.extend([i]*len(per[c]['rel']))
    out['dist_err']=100*weighted_medians(vals,labels,weights);point['dist_err']=100*float(np.median(vals)) if vals else float('nan')
    return point,out
def runname(model,e,suffix):return f'infer_{model}ft_e{e:03d}_{suffix}'
def compare_specs():
    for name in ['v20a','v20b']:
        for e in EPS:
            for ds in ['self','cross','fast','turn','platoon']:
                a=runname('v18',e,'v12causal' if ds=='cross' else name+'causal');b=runname(name,e,'v12causal' if ds=='cross' else 'selfcausal')
                yield dict(name=name,epoch=e,dataset=ds,base='v18',a=a,b=b)
                if ds in SCENES[:3]:yield dict(name=name,epoch=e,dataset=ds,base='v19'+ds,a=runname('v19'+ds,e,'selfcausal'),b=b)
    for ds in ['self','cross','fast','turn','platoon']:
        for e in EPS:yield dict(name='v20b',epoch=e,dataset=ds,base='v20a',a=runname('v20a',e,'v12causal' if ds=='cross' else 'selfcausal'),b=runname('v20b',e,'v12causal' if ds=='cross' else 'selfcausal'))
def main():
    specs=list(compare_specs());results=[];(OUT/'ci_cache').mkdir(exist_ok=True)
    for ds in ['self','cross','fast','turn','platoon']:
        groups=grouping(ds);w=multiplicities(len(groups));cache={}
        names=sorted({x[k] for x in specs if x['dataset']==ds for k in ['a','b']})
        for name in names:
            target=OUT/'ci_cache'/f'{name}_{ds}.npz';statsfile=OUT/'stats'/(name+'.pkl.gz')
            if not (OUT/'runs'/(name+'.json')).exists(): print('WAIT_STATS',name,flush=True)
            import time
            while not (OUT/'runs'/(name+'.json')).exists(): time.sleep(5)
            h=sha(statsfile)
            if target.exists():
                z=np.load(target);assert str(z['stats_sha'])==h;cache[name]=({k:float(z['p_'+k]) for k in ['strong','crit','safe','cap_all','det_close','dist_err']},{k:z[k] for k in ['strong','crit','safe','cap_all','det_close','dist_err']})
            else:
                with gzip.open(statsfile,'rb') as f:per=pickle.load(f)[ds]
                point,boot=statistics(per,groups,w);np.savez_compressed(target,stats_sha=h,**boot,**{'p_'+k:v for k,v in point.items()});cache[name]=(point,boot)
            print('BOOT',ds,name,flush=True)
        for spec in [x for x in specs if x['dataset']==ds]:
            pa,aa=cache[spec['a']];pb,bb=cache[spec['b']]
            for metric in aa:
                diffs=bb[metric]-aa[metric];valid=diffs[np.isfinite(diffs)];interval=np.quantile(valid,[.025,.975]).tolist() if len(valid) else [None,None]
                results.append(dict(**spec,metric=metric,a_value=pa[metric],b_value=pb[metric],delta=pb[metric]-pa[metric],ci95=interval,draws=DRAWS,valid=len(valid),seed=SEED,clusters=len(groups),edge_under_0_1=bool(len(valid) and min(abs(x) for x in interval)<.1)))
        dump(OUT/'ci_results.json',results)
    lines=['# 独立対応付き場面bootstrap（10,000回・seed 20260909）','','差=B−A。距離は各再抽出の全対応ペアの中央値。区間はpercentile 2.5/97.5。学習seed変動・多重比較補正は含まない。0近傍の端は二値判断しない。','','| 候補 | epoch | 集合 | 基準 | 指標 | 差pt | 95%区間pt | 場面数 |','|---|---:|---|---|---|---:|---|---:|']
    for x in results:
        lo,hi=x['ci95'];lines.append(f"| {x['name']} | {x['epoch']} | {x['dataset']} | {x['base']} | {x['metric']} | {x['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {x['clusters']} |")
    (OUT/'ci_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('CI_DONE',len(results),flush=True)
def test():
    rng=np.random.default_rng(404);cases=0
    for _ in range(100):
        g=int(rng.integers(1,15));values=rng.integers(0,50,size=int(rng.integers(1,200)))/7;labs=rng.integers(g,size=len(values));w=rng.integers(0,5,size=(20,g));actual=weighted_medians(values,labs,w)
        expected=np.array([np.median(np.repeat(values,ww[labs])) if np.sum(ww[labs]) else np.nan for ww in w]);assert np.allclose(actual,expected,equal_nan=True);cases+=20
    assert np.isnan(weighted_medians([],[],np.ones((2,3)))).all();dump(OUT/'ci_algorithm_tests.json',dict(explicit_expansion_cases=cases,all_match=True,empty_handled=True));print('TEST_PASS',cases)
if __name__=='__main__':test() if '--test' in sys.argv else main()
