"""Prediction provenance, exclusive publication, and byte-exact verification (no GPU).

Legacy submissions without a versioned identity are deliberately NOT certified.
Re-infer into a fresh CAUSAL_RUNS_DIR; retain the original run for comparison.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

SCHEMA = 2


N_CLASSES = 9   # 2026-10-08 変更票 #48: bicycle = 8 を足して 9 クラス（旧 8 のままだと bicycle の予測で止まる・勉強チャットの指摘 10/8）


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix='.part') as f:
        tmp = Path(f.name)
        f.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode('utf-8'))
        f.flush()
        os.fsync(f.fileno())
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def acquire_lock(run):
    """OS-owned lock: released even on process death; caller must keep/close handle."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    f = (run/'.prediction.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if f.seek(0, 2) == 0:
                f.write(b'0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except OSError:
        f.close()
        raise RuntimeError(f'run is in use: {run}') from None


def audio_manifest(foa, clips):
    if not clips or len(set(clips)) != len(clips):
        raise ValueError('empty or duplicate input manifest')
    for c in clips:
        if Path(c).name != c or '/' in c or '\\' in c or not c.endswith('.flac'):
            raise ValueError(f'invalid clip name: {c}')
    return {c: sha256(Path(foa)/c) for c in sorted(clips)}


def check_identity(meta):
    required = {'schema', 'ckpt_sha256', 'ckpt_epoch', 'clips', 'audio_sha256',
                'height', 'settings', 'code_sha256', 'foa'}
    if not required <= meta.keys() or meta['schema'] != SCHEMA:
        raise ValueError('unverified legacy identity; re-infer into a fresh CAUSAL_RUNS_DIR')
    clips = meta['clips']
    if not clips or len(set(clips)) != len(clips) or set(clips) != set(meta['audio_sha256']):
        raise ValueError('invalid identity manifest')
    if any(Path(c).name != c or '/' in c or '\\' in c or not c.endswith('.flac') for c in clips):
        raise ValueError('invalid identity clip name')
    if type(meta['ckpt_epoch']) is not int or meta['ckpt_epoch'] < 0:
        raise ValueError('missing checkpoint epoch')
    if not re.fullmatch('[0-9a-f]{64}', meta['ckpt_sha256']):
        raise ValueError('invalid checkpoint digest')
    # Astra RN20-04 (2026-10-20): long takes (60 s = 600 frames) carry their frame count in the identity;
    # absent = 100 (10 s clips, unchanged behaviour for every existing run_meta).
    frames = meta.get('frames', 100)
    if type(frames) is not int or frames < 1:
        raise ValueError('invalid frame count')


def frames_of(meta):
    return int(meta.get('frames', 100))


def begin_run(run, meta):
    """Call with exclusive lock held. Never assign new provenance to old CSVs."""
    run = Path(run)
    check_identity(meta)
    name_epoch = re.search(r'_e(\d+)_', run.name)
    if name_epoch and int(name_epoch[1]) != meta['ckpt_epoch']:
        raise ValueError('output name disagrees with checkpoint epoch')
    path = run/'run_meta.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        check_identity(old)
        if old != meta:
            raise ValueError('run identity changed; choose a fresh CAUSAL_RUNS_DIR')
    else:
        if any((run/'submissions').glob('*')) or (run/'val_all_causal.csv').exists():
            raise ValueError('unattributed existing predictions; choose a fresh CAUSAL_RUNS_DIR')
        atomic_json(path, meta)
    (run/'val_all_causal.done.json').unlink(missing_ok=True)


def canonical_rows(run, meta):
    subs = Path(run)/'submissions'
    expected = {Path(c).stem+'.csv' for c in meta['clips']}
    if {p.name for p in subs.glob('*.csv')} != expected or any(subs.glob('*.part')):
        raise ValueError('individual prediction manifest incomplete or contains extra/partial files')
    nfr = frames_of(meta)
    for c in sorted(meta['clips']):
        clip = Path(c).stem
        with (subs/(clip+'.csv')).open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split(',')
                if len(fields) != 6:
                    raise ValueError(f'malformed prediction: {clip}')
                k, cls, track = map(int, fields[:3])
                az, el, dist = map(float, fields[3:])
                if not (0 <= k < nfr and 0 <= cls < N_CLASSES and 0 <= track < 3
                        and math.isfinite(az) and -180 <= az <= 180
                        and math.isfinite(el) and -90 <= el <= 90
                        and (math.isnan(dist) or math.isfinite(dist) and dist >= 0)):
                    raise ValueError(f'invalid prediction values: {clip}')
                yield f'{clip},{line}\n'.encode('utf-8')


def publish(run, meta):
    """Call after inference with the same lock and checked identity still held."""
    run = Path(run)
    check_identity(meta)
    tmp = run/'val_all_causal.csv.part'
    n = 0
    try:
        with tmp.open('wb') as f:
            for line in canonical_rows(run, meta):
                f.write(line); n += 1
            f.flush(); os.fsync(f.fileno())
        sha = sha256(tmp)
        tmp.replace(run/'val_all_causal.csv')
        done = {'schema': SCHEMA, 'n_clips':len(meta['clips']), 'n_files':len(meta['clips']),
                'n_rows':n, 'sha256':sha, 'identity_sha256':digest(meta),
                'ckpt_epoch':meta['ckpt_epoch'], 'ckpt_sha256':meta['ckpt_sha256']}
        atomic_json(run/'val_all_causal.done.json', done)
        return done
    finally:
        tmp.unlink(missing_ok=True)


def verify(run, expected_count, *, repair=False):
    """Require a completed, attributed run. Count equality alone is insufficient."""
    run = Path(run)
    if expected_count <= 0 or not run.is_dir():
        raise ValueError('positive expected clip count and existing run required')
    with acquire_lock(run):
        meta = json.loads((run/'run_meta.json').read_text(encoding='utf-8'))
        check_identity(meta)
        done = json.loads((run/'val_all_causal.done.json').read_text(encoding='utf-8'))
        epoch = re.search(r'_e(\d+)_', run.name)
        if (len(meta['clips']) != expected_count or done.get('schema') != SCHEMA
                or done.get('identity_sha256') != digest(meta)
                or done.get('n_clips') != expected_count or done.get('n_files') != expected_count
                or done.get('ckpt_epoch') != meta['ckpt_epoch']
                or done.get('ckpt_sha256') != meta['ckpt_sha256']
                or epoch and int(epoch[1]) != meta['ckpt_epoch']):
            raise ValueError('completion identity/count/epoch mismatch')
        if audio_manifest(meta["foa"], meta["clips"]) != meta["audio_sha256"]:
            raise ValueError("completed predictions refer to superseded audio")
        table = meta["height"].get("table")
        if table and sha256(table) != meta["height"].get("table_sha256"):
            raise ValueError("completed predictions refer to a superseded height table")
        h = hashlib.sha256(); n = 0
        for line in canonical_rows(run, meta):
            h.update(line); n += 1
        # Changed individual results must never be re-certified by a repair.
        if done.get('sha256') != h.hexdigest() or done.get('n_rows') != n:
            raise ValueError('individual results disagree with completed digest')
        path = run/'val_all_causal.csv'
        status = 'OK'
        if not path.exists() or sha256(path) != h.hexdigest():
            if not repair:
                raise ValueError('aggregate content differs from completed individual predictions')
            publish(run, meta)
            status = 'REBUILT'
        return f'{status} n_files={expected_count} n_rows={n} sha256={h.hexdigest()}'


def verify_download(csv_path, expected_count, done_path, meta_path):
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    done = json.loads(Path(done_path).read_text(encoding="utf-8"))
    check_identity(meta)
    epoch = re.search(r"_e(\d+)_", Path(csv_path).stem)
    if (expected_count <= 0 or len(meta["clips"]) != expected_count
            or done.get("schema") != SCHEMA or done.get("identity_sha256") != digest(meta)
            or done.get("n_clips") != expected_count or done.get("n_files") != expected_count
            or done.get("ckpt_epoch") != meta["ckpt_epoch"] or done.get("ckpt_sha256") != meta["ckpt_sha256"]
            or epoch and int(epoch[1]) != meta["ckpt_epoch"]):
        raise ValueError("download provenance mismatch")
    ids = {Path(c).stem for c in meta["clips"]}
    n = 0
    with Path(csv_path).open(encoding="utf-8") as f:
        for line in f:
            if len(line.strip().split(",")) != 7 or line.split(",", 1)[0] not in ids:
                raise ValueError("unexpected downloaded clip/row")
            n += 1
    if n != done.get("n_rows") or sha256(csv_path) != done.get("sha256"):
        raise ValueError("download content mismatch")
    return "DOWNLOAD_OK"


def logical_code_name(path, root=None):
    """Identity key for a code file that does not depend on the job-specific copy or its location
    (audit 2026-09-09 R1/A3: `_causal_infer_v17_job4090.py` under `_job4090/` must equal `_causal_infer_v17.py`).
    Files under `root` keep their relative directory (minus a leading `_job<id>` directory); others use the basename."""
    p = Path(path).resolve()
    name = re.sub(r'_job\d+(?=\.py$)', '', p.name)
    if root is not None:
        try:
            rel = p.relative_to(Path(root).resolve())
            parts = list(rel.parts[:-1]) + [name]
            if parts and re.fullmatch(r'_job\d+', parts[0]):
                parts = parts[1:]
            return '/'.join(parts)
        except ValueError:
            pass
    return name


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('expected_count', type=int)
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--done', type=Path)
    parser.add_argument('--meta', type=Path)
    args=parser.parse_args()
    try:
        if args.download:
            print(verify_download(args.run,args.expected_count,args.done,args.meta))
        else:
            print(verify(args.run,args.expected_count,repair=args.repair))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f'NG {exc}\n')
