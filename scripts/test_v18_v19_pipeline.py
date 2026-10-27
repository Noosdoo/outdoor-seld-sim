"""Exercise actual shell suffixes/transport and generation completeness without SSH/GPU."""
from pathlib import Path
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'server_sde'),str(ROOT/'scripts')]
import _prediction_integrity as I
import _run_v19_gen as G
import numpy as np
import soundfile as sf

BASH=Path('C:/Program Files/Git/bin/bash.exe') if sys.platform=='win32' else Path('/bin/bash')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='seld_pipeline_');self.addCleanup(self.tmp.cleanup)
        self.d=Path(self.tmp.name)

    def shell(self, source):
        return subprocess.run([str(BASH),'--noprofile','--norc'],input=source,cwd=self.d,
                              capture_output=True,text=True,encoding='utf-8',errors='replace')

    def test_all_scorers_failing_never_report_done(self):
        (self.d/'out/audit_rescore').mkdir(parents=True)
        for version in ('v18','v19'):
            script=(ROOT/f'server_sde/_fetch_{version}_results.sh').read_text(encoding='utf-8')
            tail=script[script.index('pids=(); names=()'):]
            pre='set -euo pipefail\nPY=false\nS=static\nfail=0\nP18=x\nG18=x\nP12=x\nM12=x\nPLAN_S=x\nG19=x\na=x\nb=x\nc=x\nd=x\n'
            p=self.shell(pre+tail)
            self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
            self.assertNotIn('FETCH_SCORE_DONE',p.stdout)
            self.assertIn('SCORE_FAILED',p.stdout)

    def test_gen_pipeline_failure_stops(self):
        script=(ROOT/'server_sde/v19_gen.sbatch').read_text(encoding='utf-8')
        setting=next(l for l in script.splitlines() if l.startswith('set -'))
        p=self.shell(setting+'\nfalse | tail -2\necho V19_GEN_DONE\n')
        self.assertNotEqual(p.returncode,0)
        self.assertNotIn('V19_GEN_DONE',p.stdout)

    def test_remote_failure_rejects_cached_file(self):
        cache=self.d/'cache';cache.mkdir()
        (cache/'infer_test.csv').write_text('cached',encoding='utf-8')
        s=f'''source {shlex.quote((ROOT/'server_sde/_fetch_verified.sh').as_posix())}
SSH=(false)
RUNS=/unused
REMOTE_VERIFY=/unused
FETCH_DIR={shlex.quote(cache.as_posix())}
fetch_one infer_test 1
'''
        self.assertNotEqual(self.shell(s).returncode,0)

    def test_corrupt_cache_is_fetched_again_despite_correct_sidecar(self):
        name='infer_v18ft_e099_selfcausal'; run=self.d/'remote'/name
        (run/'submissions').mkdir(parents=True)
        foa=self.d/'foa';foa.mkdir();(foa/'a.flac').write_bytes(b'audio')
        meta={'schema':2,'ckpt_sha256':'a'*64,'ckpt_epoch':99,'clips':['a.flac'],
              'foa':str(foa),'audio_sha256':I.audio_manifest(foa,['a.flac']),'height':{},'settings':{},'code_sha256':{}}
        I.begin_run(run,meta);(run/'submissions/a.csv').write_text('0,0,0,0,0,10\n',encoding='utf-8')
        done=I.publish(run,meta);response=I.verify(run,1)
        cache=self.d/'cache';cache.mkdir()
        (cache/f'{name}.csv').write_text('WRONG\n',encoding='utf-8')
        (cache/f'{name}.sha256').write_text(done['sha256'],encoding='utf-8')
        q=lambda p:shlex.quote(str(p).replace('\\','/'))
        script=f'''set -euo pipefail
source {q(ROOT/'server_sde/_fetch_verified.sh')}
ssh_mock() {{ printf '%s\\n' {shlex.quote(response)}; }}
scp() {{ [ "$1" != -q ] || shift; cp "${{1#is-server:}}" "$2"; }}
SSH=(ssh_mock)
RUNS={q(run.parent)}
REMOTE_VERIFY=unused
FETCH_DIR={q(cache)}
PY={q(sys.executable)}
LOCAL_VERIFY={q(ROOT/'server_sde/_prediction_integrity.py')}
fetch_one {name} 1
'''
        p=self.shell(script)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertEqual((cache/f'{name}.csv').read_bytes(),(run/'val_all_causal.csv').read_bytes())
        self.assertTrue((cache/f'{name}.run_meta.json').exists())

    def make_clip(self):
        old=G.m9.DS;G.m9.DS=self.d/'dataset_outdoor_siren_v19'
        self.addCleanup(setattr,G.m9,'DS',old)
        clip='train_syn_clip20001';ds=G.m9.DS
        for sub in ('foa','metadata','masks',f'work/{clip}'):(ds/sub).mkdir(parents=True,exist_ok=True)
        self.row={'clip_id':clip,'scenario':'v19fast','seed':'1'}
        scene={'row':self.row,'sources':[{'class':'car_drive'}],'stats':{'n_label_rows':1}}
        (ds/'work'/clip/'scene.json').write_text(json.dumps(scene),encoding='utf-8')
        (ds/'metadata'/f'{clip}.csv').write_text('0,4,0,0,0\n',encoding='utf-8')
        (ds/'masks'/f'{clip}.csv').write_text('frame,class,snr_a_db\n0,4,1\n',encoding='utf-8')
        sf.write(ds/'foa'/f'{clip}.flac',np.zeros((240000,4),np.float32),24000)
        return ds,clip

    def test_complete_generated_clip_accepted(self):
        _,clip=self.make_clip();self.assertTrue(G.clip_complete(clip,self.row))

    def test_generation_rejects_wrong_samplerate(self):
        ds,clip=self.make_clip()
        sf.write(ds/'foa'/f'{clip}.flac',np.zeros((240000,4),np.float32),48000)
        self.assertFalse(G.clip_complete(clip,self.row))

    def test_generation_rejects_short_labels_and_changed_plan(self):
        ds,clip=self.make_clip()
        self.assertFalse(G.clip_complete(clip,{**self.row,'seed':'2'}))
        (ds/'metadata'/f'{clip}.csv').write_bytes(b'')
        self.assertFalse(G.clip_complete(clip,self.row))

    def test_generation_rejects_wrong_scene_and_old_turn(self):
        ds,clip=self.make_clip();p=ds/'work'/clip/'scene.json'
        scene=json.loads(p.read_text(encoding='utf-8'))
        scene['row']['clip_id']='wrong';p.write_text(json.dumps(scene),encoding='utf-8')
        self.assertFalse(G.clip_complete(clip))
        scene['row']['clip_id']=clip;scene['row']['scenario']='v19turn';scene['v19']={'turn_rev':1}
        p.write_text(json.dumps(scene),encoding='utf-8');self.assertFalse(G.clip_complete(clip))

    def test_generation_rejects_partial_audio_and_missing_scene(self):
        ds,clip=self.make_clip();(ds/'foa'/f'{clip}.flac').write_bytes(b'partial')
        self.assertFalse(G.clip_complete(clip))
        (ds/'work'/clip/'scene.json').unlink();self.assertFalse(G.clip_complete(clip))

    def test_purge_cannot_escape_dataset(self):
        self.make_clip()
        with self.assertRaises(ValueError):G.purge_partial('../../other')


if __name__=='__main__':unittest.main(verbosity=2)
