"""Refresh only warning cues in the existing development demo; preserve distance cues.
Default is a dry run. --apply updates repo copies; --unity-assets PATH also deploys
only cue CSVs and the replay source. No recordings/GT/model outputs are rewritten.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, sys
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import step12_notify_v9c_multitrack as C
DEMO=ROOT/'out/joycon_demo_v2'
SPEC=importlib.util.spec_from_file_location('organize',ROOT/'scripts_archive/_organize_demo_v2.py')
ORG=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ORG)
NAMES={0:'siren',1:'horn',2:'backup_beep',3:'bike_bell',5:'crossing'}
def sha(b):return hashlib.sha256(b).hexdigest()
def body(text):
    return [r for r in csv.reader(text.splitlines()) if len(r)==5 and r[0]!='t_s' and not r[0].startswith('#')]
def warnings(frames):
    return C.warn_fires_v9c(frames,45.,nframes=100,confirm=3,expire=50,ema=.5,dedup_deg=30.)
def prepare():
    custom=DEMO/'model_infer/val_all_causal.csv'
    tune=ROOT/'out/predictions_v43tune/val_all_causal.csv'
    model=C.load_pred_all(custom);dev=C.load_pred_all(tune)
    planned={};report=[];inputs={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [custom,tune,ROOT/'scripts/step12_notify_v9c_multitrack.py']}
    for p in sorted(DEMO.glob('*_cues.csv')):
        stem=p.name[:-len('_cues.csv')];oracle=stem.startswith('custom_') and not stem.endswith('_model')
        if oracle:
            gt=DEMO/'model_infer/metadata_dist'/(stem+'.csv');inputs[str(gt.relative_to(ROOT))]=sha(gt.read_bytes())
            frames=defaultdict(list)
            with gt.open(encoding='utf-8') as f:
                for row in csv.reader(f):
                    k,c=int(row[0]),int(row[1])
                    if c in NAMES:frames[k].append((c,float(row[3]),float(row[4])))
        else:
            key=stem[:-6] if stem.endswith('_model') else C.normalize(stem)
            source=model if stem.endswith('_model') else dev
            if key not in source:
                # This pre-existing quiet demo is the only confirmed no-detection clip.
                detect=DEMO/(stem+'_detect.csv')
                assert key=='custom_13_shizuka_nanimonai' and len(detect.read_text(encoding='utf-8').strip().splitlines())==1, ('Missing prediction',key)
            frames=source.get(key,{})
        before=p.read_text(encoding='utf-8');old=body(before)
        unchanged=[r for r in old if r[2]!='警告']
        newwarn=[[f'{(k+1)/10:.1f}','L' if az>0 else 'R','警告',NAMES[c],f'{az:.0f}'] for k,c,az in warnings(frames)]
        rows=sorted(unchanged+newwarn,key=lambda r:float(r[0]))
        text='t_s,side,tier,class,az_deg\n'+'\n'.join(','.join(r) for r in rows)
        text+='\n# '+('オラクル（保存したGT系列）' if oracle else '本物のモデル検出')+' → v4.3＋v9c (gate45 dedup30 confirm3 expire50 EMA0.5)\n'
        assert [r for r in body(text) if r[2]!='警告']==unchanged
        planned[p]=text.encode('utf-8')
        report.append(dict(clip=stem,oracle=oracle,old_warning_count=sum(r[2]=='警告' for r in old),new_warning_count=len(newwarn),distance_cues_unchanged=True,before_sha256=sha(p.read_bytes()),after_sha256=sha(planned[p])))
    assert len(planned)==62, 'Unexpected demo inventory; inspect before applying'
    for src,cat,jp in ORG.resolve_entries(ORG.MAP):
        source=DEMO/(src+'_cues.csv');dest=DEMO/'場面'/cat/(jp+'_cues.csv')
        assert dest.exists() and source in planned,(source,dest)
        assert [r for r in body(dest.read_text(encoding='utf-8')) if r[2]!='警告']==[r for r in body(planned[source].decode('utf-8')) if r[2]!='警告'], ('Local distance cues diverged',dest)
        planned[dest]=planned[source]
    return planned,report,inputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');ap.add_argument('--unity-assets',type=Path)
    args=ap.parse_args();planned,report,inputs=prepare()
    if args.unity_assets:
        assets=args.unity_assets.resolve();assert (assets.parent/'ProjectSettings/ProjectVersion.txt').exists()
        for p,data in list(planned.items()):
            if (DEMO/'場面') in p.parents:
                target=assets/'StreamingAssets/joycon_demo_v2'/p.relative_to(DEMO/'場面')
                assert target.exists(),target
                assert [r for r in body(target.read_text(encoding='utf-8')) if r[2]!='警告']==[r for r in body(data.decode('utf-8')) if r[2]!='警告'], ('Unity distance cues diverged',target)
                planned[target]=data
        planned[assets/'JoyconDemoPlayer.cs']=(DEMO/'unity/JoyconDemoPlayer.cs').read_bytes()
    out=ROOT/'out/joycon_warn_v9c_2026-10-01';out.mkdir(exist_ok=True)
    if args.apply:
        # All inputs and targets have been checked before the first write. Preserve the originals.
        backup=out/'backup';backup.mkdir(exist_ok=True)
        for i,(p,data) in enumerate(planned.items()):
            key=sha(str(p.resolve()).encode())[:16]+'_'+p.name
            b=backup/key
            if not b.exists() and p.exists():b.write_bytes(p.read_bytes())
            p.write_bytes(data)
        for p,data in planned.items():assert p.read_bytes()==data,p
    result=dict(applied=args.apply,rule=dict(gate=45,dedup=30,confirm=3,expire=50,ema=.5),inputs_sha256=inputs,clips=report,targets=[dict(path=str(p.resolve()),sha256=sha(data)) for p,data in planned.items()])
    (out/('applied.json' if args.apply else 'planned.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(applied=args.apply,clips=len(report),targets=len(planned),warnings_before=sum(r['old_warning_count'] for r in report),warnings_after=sum(r['new_warning_count'] for r in report),distance_cues_unchanged=True),ensure_ascii=False))
if __name__=='__main__':main()
