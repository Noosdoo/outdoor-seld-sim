"""Executable specification examples, NOT a production checkpoint selector.
Tiny fixtures only; no real fold20 data or shared pipeline mutations.
"""
from audit_v20_reproduce_20260911 import OUT,dump
import math,statistics
EPOCHS=[19,39,59,79,99,119,139,149];DIST={4,6,7}
def measure(gt,pred):
    # rows=(frame,class,track,azimuth,horizontal_distance), canonical tie order.
    for r in gt+pred:
        if not (0<=r[0]<100 and math.isfinite(r[3]) and -180<=r[3]<=180):raise ValueError('invalid row')
        if math.isinf(r[4]) or (math.isfinite(r[4]) and r[4]<0):raise ValueError('invalid distance')
    gs=sorted(gt);ps=sorted(pred);rels=[];allrels=[];n=0;det=0;cap=0
    for k in range(40,100):
        gg=[x for x in gs if x[0]==k and x[1] in DIST];pp=[x for x in ps if x[0]==k and x[1] in DIST and math.isfinite(x[4])]
        assert all(math.isfinite(x[4]) for x in gg)
        n+=sum(x[4]<=1.5 for x in gg)
        cand=sorted((abs((g[3]-p[3]+180)%360-180),i,j) for i,g in enumerate(gg) for j,p in enumerate(pp) if g[1]==p[1] and abs((g[3]-p[3]+180)%360-180)<=20)
        ug=set();up=set()
        for az,i,j in cand:
            if i in ug or j in up:continue
            ug.add(i);up.add(j);g,p=gg[i],pp[j];er=abs(p[4]-g[4])/max(g[4],.1);allrels.append(er)
            if g[4]<=1.5:rels.append(er);det+=1;cap+=p[4]<=1.5
    return dict(error=statistics.median(rels) if rels else None,all_error=statistics.median(allrels) if allrels else None,n=n,det=det,cap=cap,coverage=det/n if n else None,capture=cap/n if n else None)
def select(metrics,verified):
    if set(metrics)!=set(EPOCHS) or set(verified)!=set(EPOCHS) or not all(verified.values()):raise ValueError('incomplete or unverified candidates')
    if len({m['n'] for m in metrics.values()})!=1 or next(iter(metrics.values()))['n']==0:raise ValueError('invalid common GT denominator')
    valid=[e for e,m in metrics.items() if m['error'] is not None]
    return min(valid,key=lambda e:(metrics[e]['error'],-metrics[e]['cap'],-e)) if valid else None
def main():
    results=[]
    def check(name,ok):assert ok,name;results.append(dict(test=name,passed=True))
    def R(k=40,c=4,t=0,a=0,d=1):return (k,c,t,a,d)
    g=[R(d=1)]+[R(k=41+i,d=10) for i in range(4)]
    a=measure(g,[R(d=1.5)]+[R(k=41+i,d=10) for i in range(4)])
    b=measure(g,[R(d=1.1)]+[R(k=41+i,d=15) for i in range(4)])
    check('all-band-best-differs-from-near-best',a['all_error']<b['all_error'] and a['error']>b['error'])
    z=measure([R()],[]);check('valid-no-detection-still-in-denominator',z['n']==1 and z['det']==0 and z['error'] is None)
    check('GT-distance-selects-near-band',measure([R(d=1)],[R(d=5)])['error']==4 and measure([R(d=5)],[R(d=1)])['error'] is None)
    check('inclusive-1.5m-boundary',measure([R(d=1.5)],[R(d=1.5)])['cap']==1)
    check('outside-1.5m-boundary',measure([R(d=1.500001)],[R(d=1)])['n']==0)
    check('frame39-excluded-frame40-included',measure([R(k=39),R(k=40)],[R(k=39,d=99),R(k=40)])['error']==0)
    check('class-and-frame-exact',measure([R()],[R(c=6),R(k=41)])['det']==0)
    check('angular20-inclusive',measure([R()],[R(a=20)])['det']==1 and measure([R()],[R(a=20.00001)])['det']==0)
    check('wraparound',measure([R(a=179)],[R(a=-179)])['det']==1)
    check('denominator-floor',measure([R(d=.05)],[R(d=.15)])['error']<1.0000001)
    check('one-prediction-cannot-match-two-GT',measure([R(t=0),R(t=1,a=1)],[R()])['det']==1)
    check('match-all-GT-before-near-filter',measure([R(t=0,a=0,d=3),R(t=1,a=5,d=1)],[R(a=0,d=1)])['det']==0)
    tiegt=[R(t=0,a=-5,d=1),R(t=1,a=5,d=1.4)];tiepred=[R(t=1,a=0,d=2),R(t=0,a=0,d=1)]
    check('row-order-independent-ties',measure(tiegt,tiepred)==measure(tiegt[::-1],tiepred[::-1]))
    check('nan-distance-is-not-valid-match',measure([R()],[R(d=float('nan'))])['det']==0)
    try:measure([R()],[R(d=-1)]);bad=False
    except ValueError:bad=True
    check('negative-distance-rejected',bad)
    good=measure([R()],[R()]);metrics={e:dict(good) for e in EPOCHS};ver={e:True for e in EPOCHS}
    check('complete-tie-latest-epoch',select(metrics,ver)==149)
    metrics[139]['cap']=0;metrics[149]['cap']=0;check('secondary-all-GT-capture',select(metrics,ver)==119)
    for e in EPOCHS:metrics[e]['error']=0.1
    metrics[19]['error']=0.1-1e-12;check('no-rounding-before-ranking',select(metrics,ver)==19)
    # Artificial quality values exercise ranking only; measured errors themselves are nonnegative.
    for failure in ['missing','unverified']:
        mm=dict(metrics);vv=dict(ver)
        if failure=='missing':mm.pop(149)
        else:vv[149]=False
        try:select(mm,vv);bad=False
        except ValueError:bad=True
        check(failure+'-stops-selection',bad)
    check('all-unmatched-selects-none',select({e:z for e in EPOCHS},ver) is None)
    # This is an acknowledged limitation of report-only coverage, not a passing safety guarantee.
    sparse=measure([R(k=40),R(k=41)],[R(k=40)])
    dense=measure([R(k=40),R(k=41)],[R(k=40,d=1.1),R(k=41,d=1.1)])
    check('conditional-error-can-prefer-sparse-detector',sparse['error']<dense['error'] and sparse['coverage']<dense['coverage'])
    dump(OUT/'selector_spec_tests.json',dict(tests=results,total=len(results),production_selector=False));print('SPEC_EXAMPLES_PASS',len(results))
if __name__=='__main__':main()
