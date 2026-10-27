"""Summarize audit evidence, raw-threshold gates and provenance join. No pipeline edits."""
from audit_v18_platoon_20260909 import ROOT,OUT,sha,rows
import json, subprocess
from fractions import Fraction

def read(p):
    b=p.read_bytes()
    return json.loads(b.decode('utf-16' if b[:2]==b'\xff\xfe' else 'utf-8-sig'))

def run():
    e=read(OUT/'evidence.json'); w=read(OUT/'warning_evidence.json'); remote=read(OUT/'remote_evidence.json')
    check=[]
    for k,p in e['provenance'].items():
        name=__import__('pathlib').Path(p['csv']).stem; r=remote['runs'][name]
        check.append(dict(name=name,ok=r['ok'] and p['sha256']==r['sha256'] and p['meta_sha256']==r['meta_sha256']))
    gates=[]
    for candidate,base,prefix in [('v18','v17b','v18'),('v19platoon','v18bundle','platoon')]:
        for ep in [99,119,139,149]:
            g={}
            for dataset in ['self','cross']:
                bm='v18' if base=='v18bundle' and dataset=='cross' else base
                a=e['tables'][f'{bm}_e{ep:03d}_{dataset}']; b=e['tables'][f'{candidate}_e{ep:03d}_{dataset}']
                g[dataset+'_distance_delta_pp']=b['dist_err']-a['dist_err']
                g[dataset+'_strong_delta_pp']=b['strong']-a['strong']
                g[dataset+'_distance_pass']=g[dataset+'_distance_delta_pp']<=.5
                g[dataset+'_strong_pass']=g[dataset+'_strong_delta_pp']>=-1
            if candidate=='v18':
                for dataset in ['self','v12']:
                    a=w[dataset]['models'][f'v17b_e{ep:03d}'];b=w[dataset]['models'][f'v18_e{ep:03d}']
                    g[dataset+'_bell_reach_pass']=100*(Fraction(b['有']['reached'],b['有']['ep'])-Fraction(a['有']['reached'],a['有']['ep']))>=-1
                    for group in ['有','無','train']:
                        g[dataset+'_'+group+'_false_pass']=Fraction(b[group]['false'],b[group]['n'])<=Fraction(a[group]['false'],a[group]['n'])*(Fraction(1,2) if group=='train' else 1)
            else:
                a=e['tables'][f'v18bundle_e{ep:03d}_new'];b=e['tables'][f'v19platoon_e{ep:03d}_new']
                g['new_distance_delta_pp']=b['dist_err']-a['dist_err'];g['new_capture_delta_pp']=b['cap_all']-a['cap_all']
                g['new_improvement_pass']=g['new_distance_delta_pp']<=-1 or g['new_capture_delta_pp']>=2
            gates.append(dict(candidate=prefix,epoch=ep,all_point_conditions=all(v for k,v in g.items() if k.endswith('_pass')),conditions=g))
    conf=ROOT/'out/dataset_outdoor_siren_v12_conf/plan/assignment_v12conf.csv'
    rr=rows(conf); core=[r for r in rr if int(r['clip_id'].split('mix')[-1])<=1200]
    snapshot={}
    for p in list((ROOT/'scripts').glob('audit_*20260909.py'))+[ROOT/'scripts/test_audit_v18_platoon_20260909.py']+[ROOT/'server_sde'/n for n in remote['scripts']]+[conf,ROOT/'scripts/_v12_conf_notify_server.py',ROOT/'scripts/step10_v12_conf_plan.py',ROOT/'md/design/確定評価v2_事前登録_下書き_2026-09-07.md']:
        snapshot[str(p.relative_to(ROOT))]=sha(p)
    unchanged={p:sha(ROOT/p)==h for p,h in e['inputs'].items()}
    tests=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/test_audit_v18_platoon_20260909.py')],capture_output=True,text=True)
    (OUT/'bootstrap_tests.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    summary=dict(prediction_pairs=check,all_local_remote_match=all(c['ok'] for c in check),
                 distance_rows=len(e['row_checks']),distance_all_match=e['all_rows_match'],
                 warning_rows=sum(len(x['rows_match']) for x in w.values()),warning_all_match=all(all(x['rows_match']) for x in w.values()),
                 gates=gates,source_hashes=snapshot,inputs_unchanged=unchanged,tests_returncode=tests.returncode,
                 local_server_scripts_match={n:sha(ROOT/'server_sde'/n)==h for n,h in remote['scripts'].items()},
                 source_changes_on_server=[(n,p) for n,r in remote['runs'].items() for p,v in r['sources'].items() if v['recorded']!=v['current']],
                 fold20_design=dict(all_clips=len(rr),core_clips=len(core),train_all=sum(r.get('scenario')=='v12train' for r in rr),train_core=sum(r.get('scenario')=='v12train' for r in core)))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 対応付き場面bootstrap — 2026-09-09','',
           '固定モデル・固定採点器に条件付けた場面の再標本化。10,000回、seed=20260909、percentile 95%区間。差は候補−比較相手、単位pt。高さ違いを分割せず、両モデル・全epochで同じ場面抽出を使う。多重比較・epoch選択の補正なし。採否の追加ゲートではない。','',
           '| 比較表 | epoch | 指標 | 差 [95%区間] pt | 場面数 | GT分母 |','| --- | --- | --- | --- | --- | --- |']
    for r in e['comparisons']:
        lo,hi=r['ci95_pp'];lines.append(f"| {r['table']} | e{r['epoch']:03d} | {'強到達' if r['metric']=='strong' else '至近捕捉(全GT)'} | {r['delta_pp']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {r['clusters']} | {r['denominator']} |")
    lines+=['','## 点推定の条件（未丸め値で比較）','','| 候補 | epoch | 全条件 | 条件外 |','| --- | --- | --- | --- |']
    for g in gates:
        bad=', '.join(k for k,v in g['conditions'].items() if k.endswith('_pass') and not v)
        lines.append(f"| {g['candidate']} | e{g['epoch']:03d} | {'条件内' if g['all_point_conditions'] else '条件外'} | {bad or 'なし'} |")
    (OUT/'paired_intervals.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_hashes','inputs_unchanged','prediction_pairs','gates']},ensure_ascii=False,indent=2))
    if not (summary['all_local_remote_match'] and summary['distance_all_match'] and summary['warning_all_match'] and all(unchanged.values()) and tests.returncode==0): raise SystemExit(1)

if __name__=='__main__':
    import sys
    run()
