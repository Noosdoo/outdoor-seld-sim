"""Actual provenance identity after a competing job completes, isolated fixture."""
from audit_v18_platoon_20260909 import ROOT,OUT,module
from pathlib import Path
import copy,json,tempfile

I=module('collision_integrity',ROOT/'server_sde/_prediction_integrity.py')
sample=ROOT/'out/v19/C/infer_v18ft_e099_v19platooncausal.run_meta.json'
meta=json.loads(sample.read_text(encoding='utf-8'))
record=[]
with tempfile.TemporaryDirectory(prefix='audit_identity_',dir=OUT) as tmp:
    run=Path(tmp)/'infer_v18ft_e099_v19platooncausal';run.mkdir()
    a=copy.deepcopy(meta); b=copy.deepcopy(meta)
    # Only the job-specific entry path changes; code bytes and every input identical.
    a['code_sha256']={'/same/entry_job101.py':'a'*64}
    b['code_sha256']={'/same/entry_job102.py':'a'*64}
    with I.acquire_lock(run):
        I.begin_run(run,a)
        I.atomic_json(run/'val_all_causal.done.json',{'fixture':'first producer completed'})
    try:
        with I.acquire_lock(run): I.begin_run(run,b)
    except ValueError as exc:
        record.append(dict(second_job_rejected=True,error=str(exc),only_difference='code_sha256 path; digest unchanged',completion_preserved=(run/'val_all_causal.done.json').exists()))
    else:
        record.append(dict(second_job_rejected=False))
(OUT/'identity_collision_evidence.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))
