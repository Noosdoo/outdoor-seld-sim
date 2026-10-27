"""Independent cleanup audit. Only writes under out/independent_cleanup_2026-09-13.

No production script is imported by collect. Git and backup are read-only.
Snapshots contain tracked source/config/docs, not dataset audio. Test execution
is a separate, explicit mode after source and side effects have been inspected.
"""
from __future__ import annotations
import argparse
import ast
import csv
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / 'out/independent_cleanup_2026-09-13'
BACKUP = Path(r'C:\Users\satos\seld_backup_2026-09-12')
BEFORE, AFTER = '188121ae2', 'e96d36417'
PREFIX = 'outdoor_seld_e2e/'

def sha(b): return hashlib.sha256(b).hexdigest()
def dec(b): return b.decode('utf-8-sig', errors='replace')
def write(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
def git(*args):
    return subprocess.check_output(['git', '-c', 'core.quotepath=false', *args], cwd=REPO)
def tree(commit):
    entries = git('ls-tree', '-r', '-z', commit, '--', PREFIX).split(b'\0')
    chosen = {}
    for e in entries:
        if not e: continue
        head, path = e.split(b'\t', 1)
        mode, kind, obj = head.split()
        name = path.decode('utf-8')[len(PREFIX):]
        if kind == b'blob' and not name.startswith('out/'):
            chosen[name] = obj.decode()
    p = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = {}
    for name, obj in chosen.items():
        p.stdin.write((obj+'\n').encode()); p.stdin.flush()
        header = p.stdout.readline().split()
        n = int(header[2]); result[name] = p.stdout.read(n); p.stdout.read(1)
    p.stdin.close(); p.wait()
    return result
def lines(s): return ''.join(f'{i}: {l}\n' for i,l in enumerate(s.splitlines(),1))

def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    old, new = tree(BEFORE), tree(AFTER)
    write('git_log.txt', dec(git('log','--format=%H %s', BEFORE+'..'+AFTER)))
    write('git_status_initial.txt', dec(git('status','--short')))
    write('git_head.txt', dec(git('rev-parse','HEAD')))
    for tag, data in [('before', old), ('after', new)]:
        for name,b in data.items():
            dest = OUT / 'snapshots' / tag / PREFIX / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b)
    manifest_path = next(BACKUP.glob('配置マニフェスト_*.csv'))
    manifest = list(csv.DictReader(io.StringIO(dec(manifest_path.read_bytes()))))
    man = {r['path'].replace('\\','/'):r for r in manifest}
    zpath = next(BACKUP.glob('*.zip'))
    with zipfile.ZipFile(zpath) as z:
        zs = {n.replace('\\','/'):z.read(n) for n in z.namelist() if not n.endswith('/')}
    zipped = {}
    for name,b in zs.items():
        relative = name[len(PREFIX):] if name.startswith(PREFIX) else name
        zipped[relative] = b
    zbad=[n for n,r in man.items() if n not in zipped or sha(zipped[n]) != r['sha256']]
    keep = []
    for name,b in sorted(new.items()):
        if name.startswith('scripts/') and '/' not in name[len('scripts/'):] and name.endswith('.py'):
            r=man.get(name); cur=(ROOT/name).read_bytes() if (ROOT/name).exists() else None
            keep.append(dict(path=name,manifest_sha=r['sha256'] if r else None,before_sha=sha(old[name]) if name in old else None,after_sha=sha(b),current_sha=sha(cur) if cur is not None else None,backup_equal=bool(r and sha(cur)==r['sha256'])))
    prov=[]
    for base in ['_causal_infer_v17.py','_prediction_integrity.py','sde_patch.py','causal_patch.py','height_patch.py']:
        n='server_sde/'+base
        prov.append(dict(path=n,manifest=man[n]['sha256'],before=sha(old[n]),after=sha(new[n]),current=sha((ROOT/n).read_bytes())))
    live={Path(n).name:n for n in new if n.startswith('scripts/') and n.endswith('.py')}
    archive={Path(n).name:n for n in new if n.startswith('scripts_archive/') and n.endswith('.py')}
    moves=[]; diff_text=[]
    for base,n in sorted(archive.items()):
        candidates=['scripts/'+base,'md/'+base]
        candidates += [o for o in old if Path(o).name==base and o not in candidates]
        orig=next((o for o in candidates if o in old),None)
        rec=dict(path=n,old=orig,sha_after=sha(new[n]))
        if orig:
            rec.update(sha_before=sha(old[orig]),identical=old[orig]==new[n])
            diff_text.extend(difflib.unified_diff(dec(old[orig]).splitlines(True),dec(new[n]).splitlines(True),fromfile=orig,tofile=n))
        moves.append(rec)
    write('archive_changes.diff',''.join(diff_text))
    write('moves.json',moves)
    md=[]
    def norm(s): return s.replace('scripts_archive/','scripts/').replace('server_v11_old/','server/')
    md_diff=[]
    for n,b in sorted(new.items()):
        if not n.endswith('.md'): continue
        orig=n if n in old else ('server/env/'+Path(n).name if n.startswith('md/results/') else '')
        if orig not in old:
            orig=next((x for x,v in old.items() if x.endswith('.md') and v==b),None)
        if orig is None or orig not in old:
            md.append(dict(path=n,kind='added'));continue
        if b==old[orig] and n==orig: continue
        eq = norm(dec(b))==norm(dec(old[orig]))
        md.append(dict(path=n,old=orig,kind='path_only' if eq else 'other'))
        if not eq:
            md_diff.extend(difflib.unified_diff(norm(dec(old[orig])).splitlines(True), norm(dec(b)).splitlines(True),fromfile=orig,tofile=n))
    write('md_comparison.json',md);write('md_nonpath.diff',''.join(md_diff))
    index='study/読解の索引_生きているコード_2026-09-09.md'
    idx=dec(new[index]); links=re.findall(r'\]\(([^)]+)\)',idx)
    indexed={Path(x).name for x in links if x.startswith('../scripts/')}
    actual={Path(n).name for n in new if n.startswith('scripts/') and '/' not in n[len('scripts/'): ]}
    refs=[]; imports=[]; dynamic=[]; syntax=[]
    bases=set(archive)
    for n,b in sorted(new.items()):
        if n.endswith(('.py','.sh','.sbatch','.ipynb','.md','.json','.yaml','.yml')):
            src=dec(b)
            for i,l in enumerate(src.splitlines(),1):
                for token in re.findall(r'[\w.-]+\.py\b', l):
                    if token in bases:
                        refs.append(dict(source=n,line=i,target=archive[token],text=l))
            if n.endswith('.py'):
                try: t=ast.parse(src)
                except SyntaxError as e: syntax.append(dict(source=n,error=str(e))); continue
                for node in ast.walk(t):
                    if isinstance(node,(ast.Import,ast.ImportFrom)):
                        names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                        for mod in names:
                            imports.append(dict(source=n,line=node.lineno,module=mod,level=getattr(node,'level',0),archive=archive.get(mod.split('.')[0]+'.py')))
                    if isinstance(node,ast.Call):
                        fn=ast.unparse(node.func)
                        if fn.endswith(('_sc','_load','spec_from_file_location','insert','append','run','Popen','system')):
                            dynamic.append(dict(source=n,line=node.lineno,call=ast.unparse(node)))
    write('archive_references.json',refs);write('imports.json',imports);write('calls.json',dynamic)
    write('numbered_sources.txt','\n'.join('\nFILE '+n+'\n'+lines(dec(b)) for n,b in sorted(new.items()) if n.startswith('server_sde/v2final') or n in ['server_sde/_causal_infer_v17.py','server_sde/_prediction_integrity.py','scripts/_v12_conf_notify_server.py','scripts/_placement_check.py']))
    report=dict(before=BEFORE,after=AFTER,backup_manifest_sha=sha(manifest_path.read_bytes()),backup_zip_sha=sha(zpath.read_bytes()),manifest_count=len(man),zip_count=len(zipped),zip_manifest_bad=zbad,keep=keep,provenance=prov,archive_count=len(archive),archive_modified=sum(not m.get('identical',False) for m in moves),duplicate_basenames=sorted(live.keys()&archive.keys()),index=dict(actual=len(actual),indexed=len(indexed),missing=sorted(actual-indexed),extra=sorted(indexed-actual)),syntax_errors=syntax,source_current_drift=[n for n,b in new.items() if n.endswith(('.py','.sh','.sbatch')) and (not (ROOT/n).exists() or (ROOT/n).read_bytes()!=b)])
    write('inventory.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['keep','provenance']},ensure_ascii=False,indent=2))
    print('KEEP',len(keep),'backup_equal',sum(r['backup_equal'] for r in keep),'new',[r['path'] for r in keep if r['manifest_sha'] is None])
    print('MD', {k:sum(x['kind']==k for x in md) for k in ['added','path_only','other']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['collect']);args=parser.parse_args()
    collect()
