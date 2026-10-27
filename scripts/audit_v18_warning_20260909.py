"""Reproduce warning tables with raw integer evidence, separate audit output."""
from audit_v18_platoon_20260909 import ROOT, OUT, module, rows, sha
import json

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    W=module('audit_warning',ROOT/'scripts/_warn_episode_eval.py')
    evidence={}
    for dataset in ['self','v12']:
        if dataset=='self':
            plan=rows(ROOT/'out/dataset_outdoor_siren_v18/plan/assignment_v18.csv')
            clips=[r['clip_id'] for r in plan if r['split']=='fold2']
            gt=ROOT/'out/dataset_outdoor_siren_v18/metadata_dist'
        else:
            plan=rows(ROOT/'out/dataset_outdoor_siren_v12/plan/assignment_v12ext.csv')
            clips=[r['clip_id'] for r in rows(ROOT/'out/dataset_outdoor_siren_v12/plan/manifest_fold2_val.csv')]
            gt=ROOT/'out/dataset_outdoor_siren_v12/metadata_dist'
        scen={r['clip_id']:r.get('scenario','v11core') for r in plan}
        groups={c:('train' if scen.get(c)=='v12train' else '有' if W.gt_episodes(gt,c,5) else '無') for c in clips}
        old=ROOT/f'out/audit_rescore/v18_warn_{dataset}.md'
        oldtext=old.read_text(encoding='utf-8'); result=dict(original_sha256=sha(old),models={},rows_match=[])
        lines=list(W.HEADER)
        for model in ['v17b','v18']:
            for ep in [99,119,139,149]:
                label=f'{model}_e{ep:03d}'
                p=ROOT/f'out/{model}/C/infer_{model}ft_e{ep:03d}_{"selfcausal" if dataset=="self" else "v12causal"}.csv'
                acc=W.evaluate(p,clips,gt,5,groups.__getitem__,40,5,30.)
                result['models'][label]=acc
                for g in W.GROUPS:
                    line=W.fmt_row(label,g,acc[g]); lines.append(line); result['rows_match'].append(line in oldtext)
                print(dataset,label,'done',flush=True)
        evidence[dataset]=result
        (OUT/f'v18_warn_{dataset}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'warning_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': run()
