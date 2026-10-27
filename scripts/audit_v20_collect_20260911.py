"""Independent declarative gate evaluation from freshly reproduced statistics."""
from audit_v20_reproduce_20260911 import *
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def run(m,e,s):return read(OUT/'runs'/f'infer_{m}ft_e{e:03d}_{s}.json')
def tab(m,e,s,ds):return run(m,e,s)['tables'][ds]
def main():
    S=module('collect20_score',ROOT/'scripts/_score_unified.py');W=module('collect20_warn',ROOT/'scripts/_warn_episode_eval.py');checks={};gates={};judgechecks={}
    (OUT/'reproduced').mkdir(exist_ok=True)
    def compare(name,lines,pattern):
        p=ROOT/'out/audit_rescore'/name;old=[l.strip() for l in p.read_text(encoding='utf-8').splitlines() if re.match(pattern,l)]
        checks[name]=dict(rows=len(old),expected=len(lines),match=old==lines,sha256=sha(p),diff=[dict(i=i,old=a,new=b) for i,(a,b) in enumerate(zip(old,lines)) if a!=b][:5])
        (OUT/'reproduced'/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for model in ['v20a','v20b']:
        scenes=SCENES[:3] if model=='v20a' else SCENES
        for part,ds in [('self_all','self'),('cross_v12h','cross'),('on_v15val','ref')]+[(s,s) for s in scenes]:
            lines=[];bands=[]
            for m in ['v18',model]:
                for e in EPS:
                    suffix='v12causal' if ds=='cross' else model+'causal' if m=='v18' else 'selfcausal'
                    r=run(m,e,suffix);lines.append(S.fmt(f'{m}_e{e:03d}',r['tables'][ds]))
                    if ds=='ref':
                        for lo,hi in [(1.4,1.6),(1.6,1.85),(1.85,2.1)]:bands.append(S.fmt(f'{m}_e{e:03d} {lo}〜{hi} m',r['tables'][f'band_{lo}_{hi}']))
            compare(f'{model}_{part}.md',lines+bands,r'^\| v(?:18|20)')
        if model=='v20b':
            for cls,part in [(0,'warn_siren'),(1,'warn_horn')]:
                lines=[]
                for m in ['v18',model]:
                    for e in EPS:
                        w=run(m,e,model+'causal' if m=='v18' else 'selfcausal')['warnings'][str(cls)]
                        for g in W.GROUPS:
                            if w[g]['n']:lines.append(W.fmt_row(f'{m}_e{e:03d}',g,w[g]))
                compare(f'{model}_{part}.md',lines,r'^\| v(?:18|20)')
        gates[model]={};lines=[]
        for e in EPS:
            decisions=[]
            for ds in SCENES[:3]:
                cand=tab(model,e,'selfcausal',ds)['dist_err']
                for base,suffix,threshold in [('v18',model+'causal',-1.),('v19'+ds,'selfcausal',1.)]:
                    a=tab(base,e,suffix,ds)['dist_err'];decisions.append(dict(dataset=ds,metric='dist_err',base=base,a=a,b=cand,delta=cand-a,threshold=threshold,pass_condition=cand-a<=threshold))
            for ds in ['self','cross']:
                suffix='selfcausal' if ds=='self' else 'v12causal';a=tab('v18',e,model+'causal' if ds=='self' else suffix,ds);b=tab(model,e,suffix,ds)
                for metric,threshold in [('dist_err',.5),('strong',-1.)]:
                    delta=b[metric]-a[metric];decisions.append(dict(dataset=ds,metric=metric,base='v18',a=a[metric],b=b[metric],delta=delta,threshold=threshold,pass_condition=delta<=threshold if metric=='dist_err' else delta>=threshold,den=b.get('n_crit') if metric=='strong' else None,num_a=round(a['strong']*a['n_crit']/100) if metric=='strong' else None,num_b=round(b['strong']*b['n_crit']/100) if metric=='strong' else None))
            gates[model][str(e)]=dict(conditions=decisions,all_pass=all(x['pass_condition'] for x in decisions))
        # Independent comparison of numerical cells and boolean gate cells, without importing _v20_judge.
        old=ROOT/f'out/audit_rescore/{model}_judge.md';parsed=[]
        for l in old.read_text(encoding='utf-8').splitlines():
            cells=[x.strip() for x in l.strip('|').split('|')]
            if not re.match(r'^\| e[0-9]{3} ',l) or len(cells)!=7 or 'すべて' in cells[1]:continue
            parsed.append(cells)
        expected=[(e,x) for e in EPS for x in gates[model][str(e)]['conditions']]
        assert len(parsed)==len(expected)==40
        diff=[]
        for cells,(e,x) in zip(parsed,expected):
            nums=[float(re.search(r'[-+]?\d+(?:\.\d+)?',cells[i]).group(0)) for i in [2,3,4]]
            wanted=[float(f"{x[k]:.4f}") for k in ['a','b','delta']]
            ok='満たさない' not in cells[6]
            if nums!=wanted or ok!=x['pass_condition']:diff.append(dict(cells=cells,expected=x))
        totals=[l for l in old.read_text(encoding='utf-8').splitlines() if l.startswith('| e') and 'すべて' in l]
        assert len(totals)==4
        totalmatch=all(('満たさない' not in l)==gates[model][str(e)]['all_pass'] for e,l in zip(EPS,totals))
        judgechecks[model]=dict(condition_rows=40,total_rows=4,all_match=not diff and totalmatch,diffs=diff)
    passed={m:[int(e) for e,v in ee.items() if v['all_pass']] for m,ee in gates.items()};selected={m:max(eps) if eps else None for m,eps in passed.items()};choice=None;ab=[]
    if all(e is not None for e in selected.values()):
        for ds in ['self','cross']:
            s='selfcausal' if ds=='self' else 'v12causal';a=tab('v20a',selected['v20a'],s,ds);b=tab('v20b',selected['v20b'],s,ds)
            for k,tol in [('dist_err',.3),('strong',1.)]:ab.append(dict(dataset=ds,metric=k,a=a[k],b=b[k],delta=b[k]-a[k],tolerance=tol,equivalent=abs(b[k]-a[k])<=tol))
        choice='v20b' if all(x['equivalent'] for x in ab) else 'v20a'
    else:choice=next((m for m,e in selected.items() if e is not None),None)
    evidence=dict(score_tables=checks,score_rows=sum(x['rows'] for x in checks.values()),score_all_match=all(x['match'] for x in checks.values()),judge_checks=judgechecks,gates=gates,passed=passed,selected=selected,ab_comparison=ab,rule_choice=choice)
    dump(OUT/'summary.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['score_tables','gates']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
