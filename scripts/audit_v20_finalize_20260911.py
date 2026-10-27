"""Extra edge checks, readable evidence, and final consistency verification."""
from audit_v20_ci_20260911 import *

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def edges():
    # A preselected near-zero endpoint, not a search for a favorable seed.
    ds='self';e=139;groups=grouping(ds);w=multiplicities(len(groups),seed=20260910);boot={}
    for m,s in [('v18','v20bcausal'),('v20b','selfcausal')]:
        name=runname(m,e,s)
        with gzip.open(OUT/'stats'/(name+'.pkl.gz'),'rb') as f:per=pickle.load(f)[ds]
        _,b=statistics(per,groups,w);boot[m]=b
    rr=[]
    for metric in boot['v18']:
        d=boot['v20b'][metric]-boot['v18'][metric];valid=d[np.isfinite(d)]
        rr.append(dict(dataset=ds,epoch=e,base='v18',candidate='v20b',metric=metric,seed=20260910,draws=10000,valid=len(valid),ci95=np.quantile(valid,[.025,.975]).tolist()))
    dump(OUT/'ci_endpoint_second_seed.json',rr);print('EDGE_CHECK_DONE',flush=True)

def summaries():
    summary=read(OUT/'summary.json');lines=['# 独立した宣言条件の再計算','','単位pt。数値は表示のみ丸め、判定は元のfloat値。staticは参考。','','| 候補 | epoch | 集合 | 指標 | 比較 | 基準% | 候補% | 差pt | 閾値 | 条件内 |','|---|---:|---|---|---|---:|---:|---:|---|---|']
    for m,ee in summary['gates'].items():
        for e,v in ee.items():
            for x in v['conditions']:
                sign='≥' if x['metric']=='strong' else '≤'
                lines.append(f"| {m} | {e} | {x['dataset']} | {x['metric']} | {x['base']} | {x['a']:.9f} | {x['b']:.9f} | {x['delta']:+.9f} | {sign}{x['threshold']} | {x['pass_condition']} |")
            lines.append(f"| {m} | {e} | 全条件 | | | | | | | {v['all_pass']} |")
    static=[]
    for e in EPS:
        a=read(OUT/'runs'/f'infer_v18ft_e{e:03d}_v20bcausal.json')['tables']['static']['dist_err']
        b=read(OUT/'runs'/f'infer_v20bft_e{e:03d}_selfcausal.json')['tables']['static']['dist_err']
        static.append(f'| e{e:03d} | {a:.4f}% | {b:.4f}% | {b-a:+.4f}pt |')
    existing=[l for l in (ROOT/'out/audit_rescore/v20b_judge.md').read_text(encoding='utf-8').splitlines() if re.match(r'^\| e\d{3} ',l) and len(l.strip('|').split('|'))==4]
    assert static==existing
    lines+=['','## static参考4行（既存判定表と完全一致）','']+static
    lines+=['',f"条件を満たすepoch: {summary['passed']}",f"各候補: {summary['selected']} / 宣言に従う選択: {summary['rule_choice']}"]
    (OUT/'reproduced/independent_judge.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(OUT/'judge_static_check.json',dict(rows=4,all_match=True))
    out=[];lines=['# Platoon感度分析','','距離帰属：同frameで方位差10/15/20/25/30度、開始frame0/30/40/50。通知帰属：20/25/30/35/40度、時間幅0/3/5/7/10frame。1frame=0.1秒。距離と通知のパラメータは別。','','| epoch | 標準距離差 b−v19 pt | 全20設定の最小 | 最大 | 条件外設定数 |','|---:|---:|---:|---:|---:|']
    for e in EPS:
        a=read(OUT/'sensitivity'/f'v19platoon_e{e:03d}.json');b=read(OUT/'sensitivity'/f'v20b_e{e:03d}.json');v=read(OUT/'sensitivity'/f'v18_e{e:03d}.json')
        d=[dict(az_gate=x['az_gate'],minframe=x['minframe'],delta=y['median']-x['median']) for x,y in zip(a['distance'],b['distance'])]
        assert all((x['az_gate'],x['minframe'])==(y['az_gate'],y['minframe']) for x,y in zip(a['distance'],b['distance']))
        nn=[100*y['strong']/y['critical']-100*x['strong']/x['critical'] for x,y in zip(v['notifications'],b['notifications'])]
        std=next(x['delta'] for x in d if x['az_gate']==20 and x['minframe']==40)
        x=dict(epoch=e,standard_distance_delta=std,distance_min=min(x['delta'] for x in d),distance_max=max(x['delta'] for x in d),distance_failures=[x for x in d if x['delta']>1],notification_strong_delta_min=min(nn),notification_strong_delta_max=max(nn));out.append(x)
        lines.append(f"| {e} | {std:+.9f} | {x['distance_min']:+.9f} | {x['distance_max']:+.9f} | {len(x['distance_failures'])} |")
    dump(OUT/'sensitivity_summary.json',out)
    lines+=['','e099は標準設定で条件内だが、20度・開始frame30では条件外。e139は試した20設定で条件内。これはplatoonの距離条件だけの再判定で、他の集合・全体採否を感度設定で再選択してはいない。','','## 通知の診断操作（到達数 / 強到達数 / 至近イベント数）','', '| モデル | epoch | 原値 | 対応予測距離のみGT化 | 対応予測方位のみGT化 | エピソード結合を除く | 至近対応の符号付き距離誤差中央値% |','|---|---:|---|---|---|---|---:|']
    for e in [99,139]:
        for m in ['v18','v20a','v20b']:
            x=read(OUT/'sensitivity'/f'{m}_e{e:03d}.json');mm=x['mechanism'];lines.append(f"| {m} | {e} | {mm['original']} | {mm['matched_distance_oracle']} | {mm['matched_azimuth_oracle']} | {mm['no_episode_merge']} | {x['close_signed_median']:+.6f} |")
    lines+=['','GT置換は診断用の非実用的な介入で、static学習が原因だと証明する試験ではない。検出・対応されない予測は修復しない。方位変更は通知追跡にも作用するため、各要因の差を足し合わせる分解はできない。']
    (OUT/'sensitivity_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('SUMMARIES_DONE',flush=True)

def validate():
    snap=read(OUT/'snapshot.json');changes=[];counts={}
    for p,h in snap['sources'].items():
        if Path(p).name.startswith('audit_') and '20260911' in p:continue
        if sha(ROOT/p)!=h:changes.append(p)
    for ds,v in snap['datasets'].items():
        p,gt,rr=dataset(ds);assert sha(p)==v['plan_sha256'],str(p)
        for c,h in v['gt_hashes'].items():assert sha(gt/(c+'.csv'))==h,c
        counts[ds]=len(rr)
    assert not changes,changes
    assert sha(ROOT/'out/notify_v43_sweep/winner.json')==snap['winner']
    for t in tasks():
        r=read(OUT/'runs'/(t['name']+'.json'));p=ROOT/t['csv'];assert sha(p)==r['provenance']['csv_sha256']
        for ext,key in [('.done.json','done_sha256'),('.run_meta.json','meta_sha256')]:assert sha(p.with_suffix(ext))==r['provenance'][key]
    ci=read(OUT/'ci_results.json');assert len(ci)==504
    mismatches=[]
    for x in ci:
        for which in ['a','b']:
            r=read(OUT/'runs'/(x[which]+'.json'))['tables'][x['dataset']];a=x[which+'_value'];b=r[x['metric']]
            if not np.isclose(a,b,rtol=0,atol=1e-10,equal_nan=True):mismatches.append([x['metric'],x[which],a,b])
    assert not mismatches,mismatches[:5]
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf-8');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert head==snap['head']
    dump(OUT/'final_validation.json',dict(head=head,source_hashes_unchanged=True,gt_and_manifest_hashes_unchanged=True,datasets=counts,run_files_unchanged=40,ci_rows=504,ci_point_values_match=True,ci_valid_draw_counts=sorted(set(x['valid'] for x in ci)),python=sys.version,numpy=np.__version__,git_status=status))
    print('FINAL_VALIDATION_OK',flush=True)

if __name__=='__main__':
    if '--edges' in sys.argv:edges()
    else:summaries();validate()
