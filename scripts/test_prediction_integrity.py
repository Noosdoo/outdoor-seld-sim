"""Regression tests for the second v18/v19 audit; CPU-only, isolated fixtures."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'server_sde'),str(ROOT/'scripts')]
import _prediction_integrity as I
import _warn_episode_eval as W


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='seld_integrity_')
        self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)/'infer_v18ft_e099_selfcausal'
        (self.run/'submissions').mkdir(parents=True)
        self.foa=Path(self.temp.name)/'foa'; self.foa.mkdir()
        for c in ('a','b'):(self.foa/f'{c}.flac').write_bytes(c.encode())
        self.meta={'schema':2,'ckpt_sha256':'a'*64,'ckpt_epoch':99,
                   'clips':['a.flac','b.flac'],'foa':str(self.foa),
                   'audio_sha256':I.audio_manifest(self.foa,['a.flac','b.flac']),
                   'height':{'table_sha256':'b'*64},'settings':{'batch':25},'code_sha256':{'infer':'c'*64}}

    def complete(self, empty=False):
        I.begin_run(self.run,self.meta)
        (self.run/'submissions/a.csv').write_text('' if empty else '0,0,0,0,0,10\n',encoding='utf-8')
        (self.run/'submissions/b.csv').write_text('',encoding='utf-8')
        return I.publish(self.run,self.meta)

    def test_normal_and_empty_predictions(self):
        self.complete(); self.assertTrue(I.verify(self.run,2).startswith('OK'))
        self.assertEqual(I.verify_download(self.run/'val_all_causal.csv',2,self.run/'val_all_causal.done.json',self.run/'run_meta.json'),'DOWNLOAD_OK')

    def test_class8_bicycle_accepted_class9_rejected(self):   # 2026-10-08 9 クラス化（bicycle = 8）
        I.begin_run(self.run,self.meta)
        (self.run/'submissions/a.csv').write_text('0,8,0,10,0,3\n',encoding='utf-8')
        (self.run/'submissions/b.csv').write_text('',encoding='utf-8')
        I.publish(self.run,self.meta); self.assertTrue(I.verify(self.run,2).startswith('OK'))
        (self.run/'submissions/a.csv').write_text('0,9,0,10,0,3\n',encoding='utf-8')
        with self.assertRaises(ValueError): list(I.canonical_rows(self.run,self.meta))

    def test_all_empty_is_valid(self):
        done=self.complete(empty=True)
        self.assertEqual(done['n_rows'],0)
        self.assertTrue(I.verify(self.run,2).startswith('OK'))

    def test_equal_row_corruption_repaired_only_from_completed_digest(self):
        self.complete(); p=self.run/'val_all_causal.csv';expected=p.read_bytes()
        p.write_text('WRONG,0,0,0,180,0,99\n',encoding='utf-8')
        with self.assertRaises(ValueError):I.verify(self.run,2)
        self.assertTrue(I.verify(self.run,2,repair=True).startswith('REBUILT'))
        self.assertEqual(p.read_bytes(),expected)

    def test_download_tampered_cache_rejected(self):
        self.complete();(self.run/'val_all_causal.csv').write_text('a,0,0,0,180,0,10\n',encoding='utf-8')
        with self.assertRaises(ValueError):I.verify_download(self.run/'val_all_causal.csv',2,self.run/'val_all_causal.done.json',self.run/'run_meta.json')

    def test_changed_individuals_are_not_recertified(self):
        self.complete();(self.run/'submissions/a.csv').write_text('0,0,0,180,0,10\n',encoding='utf-8')
        with self.assertRaises(ValueError):I.verify(self.run,2,repair=True)

    def test_missing_done_not_repaired(self):
        self.complete();(self.run/'val_all_causal.done.json').unlink()
        with self.assertRaises(FileNotFoundError):I.verify(self.run,2,repair=True)

    def test_unknown_provenance_rejected(self):
        (self.run/'submissions/a.csv').write_text('',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'unattributed'):I.begin_run(self.run,self.meta)
        self.assertFalse((self.run/'run_meta.json').exists())

    def test_old_schema_rejected(self):
        I.atomic_json(self.run/'run_meta.json',{'ckpt_sha256':'a'*64})
        with self.assertRaisesRegex(ValueError,'legacy'):I.begin_run(self.run,self.meta)

    def test_changed_audio_height_code_manifest_rejected(self):
        self.complete()
        for key, value in [('audio_sha256',{'a.flac':'0'*64,'b.flac':'1'*64}),
                           ('height',{'table_sha256':'d'*64}),('settings',{'batch':1}),
                           ('code_sha256',{'infer':'e'*64}),('clips',['b.flac','a.flac'])]:
            with self.subTest(key=key):
                meta=copy.deepcopy(self.meta);meta[key]=value
                with self.assertRaises(ValueError):I.begin_run(self.run,meta)

    def test_epoch_name_mismatch_rejected(self):
        self.meta['ckpt_epoch']=119
        with self.assertRaises(ValueError):I.begin_run(self.run,self.meta)

    def test_same_count_wrong_clip_ids_rejected(self):
        self.complete();(self.run/'submissions/b.csv').rename(self.run/'submissions/c.csv')
        with self.assertRaises(ValueError):I.verify(self.run,2,repair=True)

    def test_partial_files_rejected(self):
        self.complete();(self.run/'submissions/b.csv.part').write_bytes(b'')
        with self.assertRaises(ValueError):I.verify(self.run,2)

    def test_malformed_output_not_published(self):
        I.begin_run(self.run,self.meta)
        (self.run/'submissions/a.csv').write_text('0,0\n',encoding='utf-8')
        (self.run/'submissions/b.csv').write_bytes(b'')
        with self.assertRaises(ValueError):I.publish(self.run,self.meta)
        self.assertFalse((self.run/'val_all_causal.done.json').exists())

    def test_completed_run_rejects_audio_replaced_under_same_name(self):
        self.complete()
        (self.foa/'a.flac').write_bytes(b'changed audio')
        with self.assertRaisesRegex(ValueError,'superseded audio'):I.verify(self.run,2)

    def test_completed_run_rejects_height_table_changed_after_inference(self):
        table=Path(self.temp.name)/'height.csv';table.write_text('a,1.5',encoding='utf-8')
        self.meta['height']={'table':str(table),'table_sha256':I.sha256(table)}
        self.complete();table.write_text('a,1.8',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'height table'):I.verify(self.run,2)

    def test_wrong_count_rejected(self):
        self.complete()
        with self.assertRaises(ValueError):I.verify(self.run,3)

    def test_live_writer_blocks_verification(self):
        self.complete()
        with I.acquire_lock(self.run):
            with self.assertRaises(RuntimeError):I.verify(self.run,2,repair=True)
        self.assertTrue(I.verify(self.run,2).startswith('OK'))

    def test_frames_bound_from_identity(self):   # Astra RN20-04 (2026-10-20): 長尺 60 s = 600 フレームの来歴
        self.meta['frames']=600
        I.begin_run(self.run,self.meta)
        (self.run/'submissions/a.csv').write_text('599,0,0,0,0,10\n',encoding='utf-8')
        (self.run/'submissions/b.csv').write_text('',encoding='utf-8')
        I.publish(self.run,self.meta); self.assertTrue(I.verify(self.run,2).startswith('OK'))
        self.assertEqual(I.verify_download(self.run/'val_all_causal.csv',2,self.run/'val_all_causal.done.json',self.run/'run_meta.json'),'DOWNLOAD_OK')
        (self.run/'submissions/a.csv').write_text('600,0,0,0,0,10\n',encoding='utf-8')
        with self.assertRaises(ValueError): list(I.canonical_rows(self.run,self.meta))

    def test_default_frames_is_100_and_bad_frames_rejected(self):
        I.begin_run(self.run,self.meta)
        (self.run/'submissions/a.csv').write_text('100,0,0,0,0,10\n',encoding='utf-8')
        (self.run/'submissions/b.csv').write_text('',encoding='utf-8')
        with self.assertRaises(ValueError): I.publish(self.run,self.meta)
        for bad in (0,-1,'600',600.0):
            meta=copy.deepcopy(self.meta); meta['frames']=bad
            with self.assertRaises(ValueError): I.check_identity(meta)

    def test_parallel_episodes_do_not_compare_dictionaries(self):
        p=Path(self.temp.name)/'clip.csv'
        p.write_text('0,0,0,0,0,10\n1,0,0,0,0,10\n0,0,1,90,0,10\n1,0,1,90,0,10\n',encoding='utf-8')
        eps=W.gt_episodes(p.parent,p.stem,0)
        self.assertEqual(len(eps),2)
        first,false=W.attribute([(1,0),(1,90)],eps,5,30)
        self.assertEqual(first,[1,1]);self.assertEqual(false,[])

    def test_missing_gt_is_not_negative_clip(self):
        with self.assertRaises(FileNotFoundError):W.gt_episodes(Path(self.temp.name),'missing',0)


if __name__=='__main__':unittest.main(verbosity=2)
