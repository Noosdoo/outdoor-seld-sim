"""Exercise exact Bash retry functions with a simulated producer; no GPU/jobs."""
from audit_v18_platoon_20260909 import ROOT, OUT, sha
import json, re, subprocess, tempfile
from pathlib import Path

def run():
    OUT.mkdir(parents=True,exist_ok=True); results=[]
    for filename,fun in [('v19_causal.sbatch','infer_retry'),('v18_v19_reinfer.sbatch','run_infer')]:
        source=ROOT/'server_sde'/filename
        text=source.read_text(encoding='utf-8')
        body=re.search(r'^'+fun+r'\(\).*?^}',text,re.M|re.S)[0]
        for mode,expected_rc,expected_calls in [('collision_once',0,2),('fatal',1,1),('collision_always',1,13)]:
            with tempfile.TemporaryDirectory(prefix='audit_retry_',dir=OUT) as tmp:
                t=Path(tmp); (t/'.venv/bin').mkdir(parents=True); (t/'out').mkdir()
                fake='''#!/bin/bash
if [ "$1" = integ ]; then exit 0; fi
n=0; [ ! -f calls ] || n=$(cat calls); n=$((n+1)); echo $n > calls
case "$TEST_MODE" in
collision_once) [ "$n" -gt 1 ] && exit 0; echo 'RuntimeError: run is in use: mock'; exit 1;;
collision_always) echo 'RuntimeError: run is in use: mock'; exit 1;;
fatal) echo 'ValueError: identity changed'; exit 1;;
esac
'''
                (t/'.venv/bin/python').write_bytes(fake.encode())
                script='set -uo pipefail\nE2E=$PWD\nSLURM_JOB_ID=mock\nINFER=infer\nINTEG=integ\nCAUSAL_RUNS_DIR=$PWD/runs\nCAUSAL_OUT=run\nsleep() { echo sleep >> sleeps; }\n'+body+'\n'+fun+(' 1' if fun=='run_infer' else '')+'\n'
                (t/'probe.sh').write_bytes(script.encode())
                import os
                env=dict(os.environ,TEST_MODE=mode)
                p=subprocess.run(['C:/Program Files/Git/bin/bash.exe','probe.sh'],cwd=t,env=env,capture_output=True,text=True,encoding='utf-8',timeout=30)
                n=int((t/'calls').read_text()) if (t/'calls').exists() else 0
                result=dict(file=filename,function=fun,source_sha256=sha(source),mode=mode,returncode=p.returncode,calls=n,passed=p.returncode==expected_rc and n==expected_calls,output=p.stdout+p.stderr)
                results.append(result)
    (OUT/'retry_evidence.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('retry tests',sum(r['passed'] for r in results),'/',len(results))
    if not all(r['passed'] for r in results): raise SystemExit(1)

if __name__=='__main__': run()
