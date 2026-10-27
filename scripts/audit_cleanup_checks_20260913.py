"""Independent static analysis and isolated tests; never executes server jobs."""
from audit_cleanup_20260913 import *
import copy
import importlib.metadata
import shutil
import time

def snapshot(tag): return OUT/'snapshots'/tag/'outdoor_seld_e2e'
def read(p): return p.read_text(encoding='utf-8-sig')
def load(n): return json.loads(read(OUT/n))

class Normalize(ast.NodeTransformer):
    def __init__(self, after=False): self.after=after
    def visit_FunctionDef(self,node):
        if self.after and node.name=='_sc': return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if self.after and ast.unparse(node)=="sys.path.insert(0, str(Path(__file__).resolve().parent))": return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        if self.after and isinstance(node.func,ast.Name) and node.func.id=='_sc':
            return ast.Call(func=ast.Name(id='SCRIPT_PATH',ctx=ast.Load()),args=node.args,keywords=[])
        return self.generic_visit(node)
    def visit_BinOp(self,node):
        if isinstance(node.op,ast.Div):
            if ast.unparse(node.left)=="ROOT / 'scripts'":
                return ast.Call(func=ast.Name(id='SCRIPT_PATH',ctx=ast.Load()),args=[node.right],keywords=[])
            if isinstance(node.left,ast.Name) and node.left.id=='ROOT' and isinstance(node.right,ast.Constant) and isinstance(node.right.value,str) and node.right.value.startswith('scripts/'):
                return ast.Call(func=ast.Name(id='SCRIPT_PATH',ctx=ast.Load()),args=[ast.Constant(value=node.right.value[8:])],keywords=[])
        return self.generic_visit(node)

def analyze():
    old,new=snapshot('before'),snapshot('after')
    moves=load('moves.json'); records=[]; diffs=[]; helper_hashes={}; helpers=0; inserts=0
    for r in moves:
        a=read(old/r['old']);b=read(new/r['path']);t=ast.parse(b)
        for node in t.body:
            if isinstance(node,ast.FunctionDef) and node.name=='_sc':
                helpers+=1;helper_hashes.setdefault(sha(ast.dump(node).encode()),[]).append(r['path'])
        inserts+=b.count('sys.path.insert(0, str(Path(__file__).resolve().parent))  # 2026-09-12')
        na=Normalize().visit(ast.parse(a));nb=Normalize(True).visit(ast.parse(b))
        same=ast.dump(na)==ast.dump(nb)
        records.append(dict(path=r['path'],identical=r['identical'],normalized_ast_equal=same))
        if not same:
            diffs.extend(difflib.unified_diff(ast.unparse(ast.fix_missing_locations(na)).splitlines(True),ast.unparse(ast.fix_missing_locations(nb)).splitlines(True),fromfile=r['old'],tofile=r['path']))
    write('ast_comparison.json',dict(records=records,helper_count=helpers,helper_variants=helper_hashes,path_inserts=inserts))
    write('ast_residual.diff',''.join(diffs))
    refs=load('archive_references.json')
    active=[r for r in refs if r['source'].split('/')[0] in ['scripts','server_sde','src','tests','colab']]
    write('live_archive_candidates.json',active)
    imps=[r for r in load('imports.json') if r['archive'] and r['source'].split('/')[0] in ['scripts','server_sde','src','tests','colab']]
    write('live_archive_imports.json',imps)
    drift=[]
    for n in load('inventory.json')['source_current_drift']:
        same=(ROOT/n).exists() and read(ROOT/n)==read(new/n)
        drift.append(dict(path=n,newline_only=same))
    write('working_tree_drift.json',drift)
    print('AST',sum(r['normalized_ast_equal'] for r in records),'/',len(records),'helpers',helpers,'variants',len(helper_hashes),'insertions',inserts)
    print('AST residual',[r['path'] for r in records if not r['normalized_ast_equal']])
    print('live archived imports',json.dumps(imps,ensure_ascii=False))
    print('live archive candidates',len(active))
    for r in active:
        if r['source'].endswith(('.sh','.sbatch')) and not r['text'].lstrip().startswith('#'):
            print(r['source'],r['line'],r['text'])
    print('substantive current drift',[r for r in drift if not r['newline_only']])

def fixtures():
    names=['out/notify_v43_sweep/winner.json','out/v17b/C/infer_v17bft_e139_v12causal.csv','out/dataset_outdoor_siren_v12/plan/manifest_fold2_val.csv']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'out/dataset_outdoor_siren_v12/metadata_dist_h').glob('*.csv')]
    for c in ['fold1_room1_mix001','fold1_room1_mix002','fold1_room1_mix003']:
        base=ROOT/'out/dataset_outdoor_siren_v1/work'/c
        names.append((base/'scene.json').relative_to(ROOT).as_posix())
        for tag in ['direct','mirror']:
            p=base/f'mono_{tag}_48k_DSref.wav'
            if not p.exists():p=base/f'mono_{tag}_48k.wav'
            names.append(p.relative_to(ROOT).as_posix())
    manifest=[]
    for n in names:
        b=(ROOT/n).read_bytes()
        for tag in ['before','after']:
            p=snapshot(tag)/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        manifest.append(dict(path=n,sha256=sha(b),bytes=len(b)))
    write('test_fixture_manifest.json',manifest)
    print('fixtures',len(manifest),sum(r['bytes'] for r in manifest),'bytes',flush=True)

def normalize_log(s, root, temp):
    s=s.replace('\r\n','\n').replace(str(root),'<SNAPSHOT>').replace(root.as_posix(),'<SNAPSHOT>')
    s=s.replace(str(temp),'<TEMP>').replace(temp.as_posix(),'<TEMP>')
    s=re.sub(r'Ran (\d+) tests? in [0-9.]+s',r'Ran \1 tests in <TIME>',s)
    s=re.sub(r'render time \(direct\+mirror\): [0-9.]+s',r'render time (direct+mirror): <TIME>',s)
    # Only generated temporary directory names are normalized; no scientific numbers.
    s=re.sub(r'(?<=<TEMP>)[/\\](?:tmp|seld_integrity_|seld_pipeline_)[A-Za-z0-9_]+','/<GENERATED>',s)
    return s

def tests():
    fixtures()
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONHASHSEED']='0'
    packages={}
    for p in ['numpy','scipy','soundfile','librosa','dynamic-sound','torch','h5py','hydra-core','pytest']:
        try: packages[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: packages[p]=None
    write('test_environment.json',dict(python=sys.version,executable=sys.executable,packages=packages))
    ns=sorted([p.relative_to(snapshot('after')).as_posix() for p in (snapshot('after')/'scripts').glob('test_*.py')]+[p.relative_to(snapshot('after')).as_posix() for p in (snapshot('after')/'tests').glob('test_*.py')])
    results=[]
    for n in ns:
        pair={}
        for tag in ['before','after']:
            root=snapshot(tag);temp=OUT/'tmp'/tag/Path(n).stem;temp.mkdir(parents=True,exist_ok=True)
            env.update(TEMP=str(temp),TMP=str(temp),PYTHONPATH=os.pathsep.join([str(root/'scripts'),str(root/'src')]),NUMBA_CACHE_DIR=str(temp/'numba'))
            cmd=[sys.executable,'-B',str(root/n)]
            t=time.perf_counter()
            r=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
            sec=time.perf_counter()-t
            out=r.stdout.decode('utf-8',errors='replace');err=r.stderr.decode('utf-8',errors='replace')
            write(f'tests/{Path(n).stem}_{tag}.stdout.txt',out);write(f'tests/{Path(n).stem}_{tag}.stderr.txt',err)
            normalized=normalize_log(out+'\n<STDERR>\n'+err,root,temp)
            write(f'tests/{Path(n).stem}_{tag}.normalized.txt',normalized)
            pair[tag]=dict(command=cmd,rc=r.returncode,seconds=sec,raw_sha=sha(r.stdout+b'\n<STDERR>\n'+r.stderr),normalized_sha=sha(normalized.encode()),skip_lines=[l for l in (out+'\n'+err).splitlines() if 'SKIP' in l or 'skipped' in l],tail=out.splitlines()[-1:] or err.splitlines()[-1:])
            print(tag,n,'rc',r.returncode,'tail',pair[tag]['tail'],flush=True)
        results.append(dict(test=n,**pair,normalized_equal=pair['before']['normalized_sha']==pair['after']['normalized_sha']))
        write('test_results.json',results)
    # Direct script execution does not call these pytest-style functions.
    extra=[]
    for tag in ['before','after']:
        root=snapshot(tag)
        code="import runpy; d=runpy.run_path('tests/test_doppler_direction.py'); fs=[(k,v) for k,v in d.items() if k.startswith('test_') and callable(v)]; [(f(),print(k,'PASS')) for k,f in fs]; print(str(len(fs))+'/2 executed')"
        env['PYTHONPATH']=os.pathsep.join([str(root/'scripts'),str(root/'src')])
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
        extra.append(dict(tag=tag,rc=r.returncode,stdout=dec(r.stdout),stderr=dec(r.stderr)))
    write('doppler_explicit_calls.json',extra)
    print('PAIRS',sum(x['normalized_equal'] for x in results),'/',len(results),'DOPPLER',extra,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['analyze','tests']);a=p.parse_args()
    globals()[a.mode]()
