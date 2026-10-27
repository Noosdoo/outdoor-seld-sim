"""Platoon scoring sensitivity and diagnostic counterfactuals, never model inference."""
from audit_v20_reproduce_20260911 import *
import numpy as np,math
from collections import defaultdict
def matching(frames,gt,gate):
    for k,gts in gt.items():
        ps=[(i,c,az,d) for i,(c,az,el,d) in enumerate(frames.get(k,[])) if c in {4,6,7} and math.isfinite(d)]
        candidates=sorted((abs((gaz-paz+180)%360-180),gi,pi,pd) for gi,(gc,tr,gaz,gd) in enumerate(gts) for pi,pc,paz,pd in ps if pc==gc and abs((gaz-paz+180)%360-180)<=gate)
        usedg=set();usedp=set()
        for angle,gi,pi,pd in candidates:
            if gi in usedg or pi in usedp:continue
            usedg.add(gi);usedp.add(pi);gc,tr,gaz,gd=gts[gi];yield k,gi,pi,gc,tr,gaz,gd,pd
def notify(S,fires,evs,az=30,t=5):
    S.A.AZ_TOL=az;S.A.T_TOL=t;att=S.A.attribute(fires,evs);got=defaultdict(set)
    for f,i in zip(fires,att):
        if i is not None and f[0]<=evs[i]['cpa']+S.FPS:got[i].add(f[2])
    ids=[i for i,e in enumerate(evs) if e['tier']=='critical']
    return [sum(bool(got[i]) for i in ids),sum('強' in got[i] for i in ids),len(ids)]
def work(spec):
    name,e=spec;S=module('sens_score',ROOT/'scripts/_score_unified.py');_,gt,rr=dataset('platoon')
    model='v18' if name=='v18' else name;suffix='v20bcausal' if name=='v18' else 'selfcausal';folder='v19' if name=='v19platoon' else 'v20'
    p=ROOT/f'out/{folder}/C/infer_{model}ft_e{e:03d}_{suffix}.csv';pred=S.load_pred_rows(p,False)
    gates=[10,15,20,25,30];starts=[0,30,40,50];dist={(a,f):[] for a in gates for f in starts};close_signed=[];notifs={(a,t):np.zeros(3,dtype=int) for a in [20,25,30,35,40] for t in [0,3,5,7,10]}
    mechanism={k:np.zeros(3,dtype=int) for k in ['original','matched_distance_oracle','matched_azimuth_oracle','no_episode_merge']};detail=[]
    for r in rr:
        c=r['clip_id'];frames=pred.get(c,{});g=S.load_gt(gt,c);evs=[S.DG.mk_event(cl,tr,fr) for cl,tr,fr in S.DG.gt_tracks(gt,c)];fires=S.fires_of(frames)
        for key in notifs:notifs[key]+=notify(S,fires,evs,*key)
        for a in gates:
            for k,gi,pi,cl,tr,gaz,gd,pd in matching(frames,g,a):
                err=abs(pd-gd)/max(gd,.1)
                for f in starts:
                    if k>=f:dist[(a,f)].append(err)
                if a==20 and k>=40 and gd<=1.5:close_signed.append((pd-gd)/max(gd,.1))
        orig=notify(S,fires,evs);mechanism['original']+=orig
        if name in ['v20a','v20b','v18'] and e in [99,139]:
            for mode in ['matched_distance_oracle','matched_azimuth_oracle']:
                changed={k:list(v) for k,v in frames.items()}
                for k,gi,pi,cl,tr,gaz,gd,pd in matching(frames,g,20):
                    pc,paz,pel,dd=changed[k][pi];changed[k][pi]=(pc,gaz if mode=='matched_azimuth_oracle' else paz,pel,gd if mode=='matched_distance_oracle' else dd)
                mechanism[mode]+=notify(S,S.fires_of(changed),evs)
            old=S.V43.v4._episodes_with_upgrade
            S.V43.v4._episodes_with_upgrade=lambda mid,strong:sorted([(k,a,'中',d) for k,a,d in mid]+[(k,a,'強',d) for k,a,d in strong])
            try:mechanism['no_episode_merge']+=notify(S,S.fires_of(frames),evs)
            finally:S.V43.v4._episodes_with_upgrade=old
        detail.append(dict(clip=c,seed=r['seed'],original=orig))
    out=dict(name=name,epoch=e,csv_sha256=sha(p),distance=[dict(az_gate=a,minframe=f,median=100*float(np.median(v)),pairs=len(v)) for (a,f),v in dist.items()],notifications=[dict(az_gate=a,time_frames=t,reached=int(v[0]),strong=int(v[1]),critical=int(v[2])) for (a,t),v in notifs.items()],close_signed_median=100*float(np.median(close_signed)),mechanism={k:v.tolist() for k,v in mechanism.items()},per_clip=detail)
    dump(OUT/'sensitivity'/f'{name}_e{e:03d}.json',out);return name+' '+str(e)+' OK'
def main():
    specs=[(m,e) for e in EPS for m in ['v18','v19platoon','v20a','v20b']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for x in pool.map(work,specs):print(x,flush=True)
    print('SENSITIVITY_DONE',flush=True)
if __name__=='__main__':main()
