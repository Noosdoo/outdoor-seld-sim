"""Bounded reproducers: AST-only helper extraction, path lookup, metadata reads.

Does not import or execute archived production programs; writes only audit OUT.
"""
from audit_cleanup_checks_20260913 import *
import importlib.machinery
from collections import Counter, defaultdict

def md_blobs(commit):
    result={}
    for e in git('ls-tree','-r','-z',commit,'--',PREFIX).split(b'\0'):
        if not e:continue
        h,n=e.split(b'\t',1);name=n.decode()[len(PREFIX):]
        if name.endswith('.md'):
            result[name]=git('cat-file','blob',h.split()[2].decode())
    return result

def md_compare():
    old,new=md_blobs(BEFORE),md_blobs(AFTER)
    def norm(s,ext=False):
        s=s.replace('scripts_archive/','scripts/').replace('server_v11_old/','server/')
        if ext:s=s.replace('md/results/server_env_','server/env/server_env_')
        return s
    rows=[];diffs=[]
    for n,b in sorted(new.items()):
        orig=n
        if n.startswith('server_v11_old/'):orig=n.replace('server_v11_old/','server/',1)
        if n.startswith('md/results/server_env_'):orig='server/env/'+Path(n).name
        if orig not in old:
            rows.append(dict(path=n,kind='added'));continue
        if b==old[orig] and orig==n:continue
        exact=norm(dec(b))==norm(dec(old[orig]));ext=norm(dec(b),True)==norm(dec(old[orig]),True)
        rows.append(dict(path=n,old=orig,kind='two_paths_only' if exact else 'env_path_only' if ext else 'other'))
        if not ext:diffs.extend(difflib.unified_diff(norm(dec(old[orig]),True).splitlines(True),norm(dec(b),True).splitlines(True),fromfile=orig,tofile=n))
    write('md_full_comparison.json',rows);write('md_full_nonpath.diff',''.join(diffs))
    with zipfile.ZipFile(next(BACKUP.glob('*.zip'))) as z:
        backup={n:z.read(n) for n in z.namelist() if n.endswith('.md')}
    br=[];bd=[]
    for orig,b in backup.items():
        if orig.startswith(PREFIX):orig=orig[len(PREFIX):]
        n=orig.replace('server/env/','md/results/',1) if orig.startswith('server/env/') else orig.replace('server/','server_v11_old/',1) if orig.startswith('server/') else orig
        if n not in new:br.append(dict(old=orig,path=n,kind='missing'));continue
        if dec(b).replace('\r\n','\n')==dec(new[n]).replace('\r\n','\n') and n==orig:continue
        eq=norm(dec(b),True).replace('\r\n','\n')==norm(dec(new[n]),True).replace('\r\n','\n')
        br.append(dict(old=orig,path=n,kind='path_only' if eq else 'other'))
        if not eq:bd.extend(difflib.unified_diff(norm(dec(b),True).splitlines(True),norm(dec(new[n]),True).splitlines(True),fromfile=orig,tofile=n))
    write('md_backup_comparison.json',br);write('md_backup_nonpath.diff',''.join(bd))
    print('MD FULL',dict(Counter(r['kind'] for r in rows)),flush=True)
    print('MD BACKUP',dict(Counter(r['kind'] for r in br)),flush=True)

def probes():
    old,new=snapshot('before'),snapshot('after');inv=load('inventory.json')
    archive={p.name:p for p in (new/'scripts_archive').glob('*.py')}; live={p.stem:p for p in (new/'scripts').glob('*.py')}
    imported=[]
    for base,p in archive.items():
        s=read(p);t=ast.parse(s)
        for node in ast.walk(t):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
                for n in names:
                    if n in live:
                        imported.append(dict(source=p.relative_to(new).as_posix(),line=node.lineno,module=n,explicit_scripts_search=bool(re.search(r'(?:ROOT|R)\s*/\s*[\"\']scripts[\"\']',s))))
    write('archive_imports_of_live.json',imported)
    fixture=OUT/'resolver_fixture';(fixture/'scripts').mkdir(parents=True,exist_ok=True);(fixture/'scripts_archive').mkdir(exist_ok=True)
    (fixture/'scripts/audit_target.py').write_text('value="live"\n',encoding='utf-8')
    (fixture/'scripts_archive/audit_target.py').write_text('value="archive"\n',encoding='utf-8')
    results=[]
    for base,p in archive.items():
        t=ast.parse(read(p));fn=next((n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_sc'),None)
        if fn is None:continue
        ns={'Path':Path,'ROOT':fixture,'__file__':str(fixture/'scripts_archive'/base)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
        choices={name:str(ns['_sc'](name).relative_to(fixture)) for name in ['audit_target.py','scripts/audit_target.py','different/audit_target.py','missing.py']}
        results.append(dict(path=p.relative_to(new).as_posix(),line=fn.lineno,choices=choices))
    write('resolver_probe.json',results)
    spec=[]
    for tag,folder in [('before','scripts'),('after','scripts_archive')]:
        root=snapshot(tag);pp=[str(root/folder)]+[s for s in sys.path if not s.startswith(str(ROOT))]
        got=importlib.machinery.PathFinder.find_spec('step11_v11_render',pp)
        spec.append(dict(tag=tag,file=f'{folder}/_test_v11_peak_resample.py',line=14,search=pp,resolved=got.origin if got else None))
    write('peak_import_resolution.json',spec)
    imp=load('imports.json');refs=load('live_archive_candidates.json')
    shell=[r for r in refs if r['source'].endswith(('.sbatch','.sh')) and not r['text'].lstrip().startswith('#')]
    # v11_audit_logs:30 is a hash input, not a Python launch.
    launches=[r for r in shell if not (r['source']=='server_sde/v11_audit_logs.sbatch' and r['line']==30)]
    write('live_archive_shell_edges.json',launches)
    print('ARCHIVE -> LIVE',len(imported),'no explicit script path', [r for r in imported if not r['explicit_scripts_search']],flush=True)
    print('HELPER PROBE',len(results),'silent collision/subpath collapse',flush=True)
    print('SHELL EDGES',len(launches),'callers',len({r['source'] for r in launches}),'targets',len({r['target'] for r in launches}),flush=True)
    print('PEAK RESOLUTION',[(r['tag'],r['resolved']) for r in spec],flush=True)
    # Frozen provenance manifests: only read, never invoke verify() (it takes a lock).
    examples=[]
    for folder in ['v17b','v18','v19','v20']:
        for p in (ROOT/'out'/folder).rglob('*.run_meta.json'):
            m=json.loads(read(p));keys=sorted(m.get('code_sha256',{}))
            examples.append(dict(path=p.relative_to(ROOT).as_posix(),count=len(keys),keys=keys))
    write('provenance_manifest_keys.json',examples)
    print('PROVENANCE MANIFESTS',len(examples),'key counts',dict(Counter(x['count'] for x in examples)),flush=True)
    r=load('remote_source_comparison.json');sel=[x for x in r if 'v2final' in x['path'] or x['path'] in [y['path'] for y in inv['provenance']]]
    write('remote_v2final_provenance.json',sel)
    print('REMOTE v2final/provenance',json.dumps(sel,ensure_ascii=False),flush=True)
    # Exact initial commit coverage, separate from dirty backup coverage.
    k=inv['keep'];print('KEEP before commit absent',[x['path'] for x in k if x['before_sha'] is None],flush=True)
    print('KEEP old git same',sum(x['before_sha']==x['after_sha'] for x in k if x['before_sha']), '/',sum(bool(x['before_sha']) for x in k),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['probes','md_compare']);a=p.parse_args();globals()[a.mode]()
