"""Run the home-demo recipe with synthetic note exports only. Never reads evaluation data.

First run test_home_demo_note.js <sandbox>/out/field_demo. Then pass that sandbox
as the sole argument. Audio/video and all outputs stay inside it. --before-fix
records the original recipe's behavior without asserting the session fallback.
"""
from pathlib import Path
import csv, json, os, subprocess, sys
import argparse
import numpy as np
import soundfile as sf
import cv2
import step19h_video_marks as marks

SCRIPTS=Path(__file__).resolve().parent

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(p, rows, fields=None):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description="合成デモの通し検査。先に test_home_demo_note.js <sandbox>/out/field_demo を実行。")
    parser.add_argument('sandbox',type=Path)
    parser.add_argument('--before-fix',action='store_true')
    args=parser.parse_args()
    root=args.sandbox.resolve(); out=root/'out/field_demo';raw=root/'raw_demo';raw.mkdir(exist_ok=True)
    session=read(out/'session.csv')[0];events=read(out/'events.csv')
    assert session['用途']=='調整用' and len(events)==5
    assert all('合成デモ' in e['特記'] or e['class']=='none' for e in events)
    sr=96000; t=np.arange(sr)/sr
    base=.0001*np.sin(2*np.pi*1000*t)
    check_files={session['点検原本']}|{r['回転後点検原本'] for r in events if r['回転後点検原本']}
    files={session['校正原本']:60,**{p:6 for p in check_files},**{r['原本']:60 if r['class']=='none' else 20 for r in events}}
    for name,seconds in files.items():
        assert Path(name).name==name and name.endswith('.WAV')
        with sf.SoundFile(raw/name,'w',samplerate=sr,channels=4,subtype='PCM_24') as f:
            for sec in range(seconds):
                block=np.column_stack((base,base*.2,base*.1,base*.3))
                if name in check_files and 1<=sec<=4:
                    az=np.deg2rad([0,-90,180,90][sec-1]);pulse=.2*np.sin(2*np.pi*1500*t[:1920])*np.hanning(1920)
                    block[:1920]+=pulse[:,None]*np.array([1,np.sin(az),0,np.cos(az)])
                f.write(block)
    vp=raw/'video/synthetic.mp4';vp.parent.mkdir(exist_ok=True)
    writer=cv2.VideoWriter(str(vp),cv2.VideoWriter_fourcc(*'mp4v'),30,(320,240))
    assert writer.isOpened()
    for i in range(600):
        im=np.zeros((240,320,3),np.uint8); x=int(300-(i%120)*2.4)
        cv2.rectangle(im,(x,100),(x+15,130),(255,255,255),-1);writer.write(im)
    writer.release()
    runlog=[]
    def run(script,*args):
        cmd=[sys.executable,'-X','utf8','-B',str(SCRIPTS/script),*map(str,args)]
        r=subprocess.run(cmd,cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace',env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
        runlog.append(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        (out/('run_before.json' if '--before-fix' in sys.argv else 'run_after.json')).write_text(json.dumps(runlog,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{script}: exit {r.returncode}',flush=True)
        if r.returncode:print(r.stdout[-1800:]+r.stderr[-1800:]);raise RuntimeError(script)
        return r
    for name in sorted(check_files):run('step19e_check_azimuth.py','--in',raw/name)
    run('step19_realsmoke_convert.py','--in',raw,'--calib',raw/session['校正原本'],'--laeq',session['LAeq_dB'],'--gain-only','--out',out/'conv')
    run('step19d_field_csv_to_ann.py','--session',out/'session.csv','--events',out/'events.csv','--audio-dir',raw,'--out',out/'ann_orig.csv')
    assert not read(out/'ann_orig.csv')  # tuning never enters the final-evaluation annotation
    assert len(read(out/'ann_orig_tuning.csv'))==5
    common=['--ann',out/'ann_orig_tuning.csv','--calib-dir',out/'conv','--pitch','0','--roll','0','--yaw','0','--out',out/'clips','--ann-out',out/'ann_all.csv']
    result=run('step19b_realsmoke_cut.py','--mode','event','--in',raw,*common)
    if '--before-fix' in sys.argv:
        print('Before fix: missing session table =', 'セッション表 ann_orig_sessions.csv:' not in result.stdout)
    else:assert 'セッション表 ann_orig_sessions.csv:' in result.stdout
    assert len(read(out/'ann_all.csv'))==4
    run('step19c_ann_validate.py','--ann',out/'ann_all.csv','--cut')
    # Only the 60-second none take is negative. Append so names cannot overwrite positive clips.
    none=next(r for r in events if r['class']=='none')
    run('step19b_realsmoke_cut.py','--mode','negative','--in',raw/none['原本'],*common,'--append')
    run('step19c_ann_validate.py','--ann',out/'ann_all.csv','--cut')
    config=read(out/'ann_orig_video_takes.csv');assert len(config)==3
    if '--before-fix' in sys.argv:
        for row in config:row['video']=str(vp)
        write(out/'ann_orig_video_takes.csv',config)
    else:
        assert all(Path(row['video']).resolve()==vp for row in config)
    takes=marks.load_takes(out/'ann_orig_video_takes.csv')
    # Exercise the actual Tk/Pillow/OpenCV GUI, hidden, and its normal q handler.
    # No user window or recording is touched; deterministic marks are supplied below.
    import tkinter as tk
    from types import SimpleNamespace
    original_tk=tk.Tk; gui_errors=[]
    def demo_tk(*a,**kw):
        r=original_tk(*a,**kw);r.withdraw();bind=r.bind
        r.report_callback_exception=lambda *exc:gui_errors.append(str(exc))
        def auto_close(sequence,callback,*args,**kwargs):
            if sequence=='<Key>':
                r.after(200,lambda:callback(SimpleNamespace(keysym='q',state=0)))
            return bind(sequence,callback,*args,**kwargs)
        r.bind=auto_close
        r.after(5000,r.destroy)
        return r
    tk.Tk=demo_tk
    try:assert marks.run_gui(takes,out/'gui_smoke','GUI')==0
    finally:tk.Tk=original_tk
    assert not gui_errors and (out/'gui_smoke/reads_GUI.csv').exists()
    print('step19h GUI: video decoded, Tk/Pillow frame rendered, q save/exit PASS',flush=True)
    state=out/'synthetic_video_reads.state.json'
    if state.exists():state.unlink()  # only this harness's synthetic state
    reader=marks.Session(takes,state,'DEMO');v=marks.Video(vp)
    for row in takes:
        fr,_,sec,_=v.get(90);reader.rec(row,'clap',fr,sec)
        for j,x in enumerate(row['markers_list']):
            fr,_,sec,_=v.get(120+36*j);reader.rec(row,'mark',fr,sec,x)
    v.release();reader.save();p,_,warnings=reader.write_outputs(out);assert not warnings
    rows=read(p);assert len(rows)==33
    fixed=next(r for r in events if r['class']=='horn')
    for sec in (8,11):
        row={k:'' for k in rows[0]};row.update(take=fixed['原本'],cls='horn',track=fixed['event_id'],t=str(sec),frame=str(sec*10));rows.append(row)
    write(p,rows)
    run('step19k_field_perception.py','--reads',p,'--ann',out/'ann_all.csv','--out',out/'perception')
    manifest=read(out/'perception/manifest.csv');assert len(manifest)==4
    annotation=read(out/'ann_all.csv')
    for e in events[:4]:
        a=next(r for r in annotation if Path(r['orig_file']).stem.lower()==Path(e['原本']).stem.lower())
        gt=list(csv.reader((out/'perception/gt'/f"{a['clip_id']}.csv").open()))
        nearest=min(gt,key=lambda r:float(r[5]));az=float(nearest[3])
        assert az<0 if e['三脚の向き']=='180' else az>0,(e['原本'],az)
    negative=[r for r in annotation if r['class']=='none']
    assert abs(sum(float(r['t_cpa'])-float(r['t_start']) for r in negative)-60)<.001
    print('PASS: 8 synthetic WAV / 5 takes / 4 GT clips / negative exposure 60s; no model inference',flush=True)

if __name__=='__main__':main()
