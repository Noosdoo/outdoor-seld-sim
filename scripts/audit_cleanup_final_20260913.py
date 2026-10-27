"""Follow-up evidence for independently identified gaps. All writes stay in OUT."""
from audit_cleanup_checks_20260913 import *
from collections import Counter, defaultdict, deque

class ReviewedNormalize(Normalize):
    def __init__(self, source, after=False, rel_loader=False):
        super().__init__(after);self.source=source;self.rel_loader=rel_loader
    def visit_Expr(self,node):
        if self.after and 'sys.path.insert(0, str(Path(__file__).resolve().parent))  # 2026-09-12' in self.source.splitlines()[node.lineno-1]:return None
        return self.generic_visit(node)
    def visit_BinOp(self,node):
        if isinstance(node.op,ast.Div) and (ast.unparse(node.left)=="R / 'scripts'" or self.rel_loader and ast.unparse(node)=='ROOT / rel'):
            return ast.Call(func=ast.Name(id='SCRIPT_PATH',ctx=ast.Load()),args=[node.right],keywords=[])
        return super().visit_BinOp(node)

def final_static():
    old,new=snapshot('before'),snapshot('after');records=[]
    for r in load('moves.json'):
        a=read(old/r['old']);b=read(new/r['path'])
        rel_loader=Path(r['path']).name in ['_sweep_link_deg.py','review_B_20260909_score_extremes.py','review_B_20260909_stats_from_cache.py']
        x=ReviewedNormalize(a,False,rel_loader).visit(ast.parse(a));y=ReviewedNormalize(b,True,rel_loader).visit(ast.parse(b))
        records.append(dict(path=r['path'],normalized_ast_equal=ast.dump(x)==ast.dump(y)))
    write('ast_reviewed.json',records);print('REVIEWED AST',sum(x['normalized_ast_equal'] for x in records),'/',len(records),flush=True)
    # Local graph uses Python import nodes and exact filename literals. It is a
    # conservative superset, not a proof for arbitrary computed import strings.
    files={p.relative_to(new).as_posix():p for d in ['scripts','scripts_archive','server_sde','src'] for p in (new/d).rglob('*.py')}
    bybase=defaultdict(list)
    for n in files:bybase[Path(n).name].append(n)
    edges=[]
    for n,p in files.items():
        t=ast.parse(read(p))
        for node in ast.walk(t):
            names=[];kind=''
            if isinstance(node,ast.Import):names=[a.name.replace('.','/')+'.py' for a in node.names];kind='import'
            if isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module.replace('.','/')+'.py'];kind='from-import'
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and re.fullmatch(r'(?:scripts/|server_sde/)?[\w.-]+\.py',node.value):names=[node.value];kind='filename literal'
            for v in names:
                direct='src/'+v
                candidates=[direct] if direct in files else bybase.get(Path(v).name,[])
                for dest in candidates:
                    if n!=dest:edges.append(dict(source=n,line=node.lineno,target=dest,kind=kind))
    shell=load('live_archive_shell_edges.json');edges+=shell
    graph=defaultdict(set)
    for e in edges:graph[e['source']].add(e['target'])
    def reachable(seeds):
        found=set(seeds);q=deque(seeds)
        while q:
            for v in graph[q.popleft()]:
                if v not in found:found.add(v);q.append(v)
        return sorted(found)
    broad=reachable({r['source'] for r in shell})
    starters=['scripts/step10_v2final_plan.py','scripts/_run_v2final_gen.py','scripts/step19_dist_labels_v2final.py','scripts/_v12_conf_notify_server.py','server_sde/_train_sde17.py','server_sde/_train_causal17.py','server_sde/_causal_infer_v17.py','server_sde/_preproc_emptyok12.py','server_sde/_preproc_sde.py','server_sde/hp_make_configs.py']
    final=reachable(starters)
    write('dependency_graph.json',dict(edges=edges,all_server_roots_archive_reachable=[n for n in broad if n.startswith('scripts_archive/')],v2final_roots=starters,v2final_reachable=final,v2final_archive=[n for n in final if n.startswith('scripts_archive/')]))
    print('BROAD ROOTS ARCHIVE',sum(n.startswith('scripts_archive/') for n in broad),'V2FINAL',len(final),'ARCHIVE',sum(n.startswith('scripts_archive/') for n in final),flush=True)
    # Verify exact local literals and retain every source line for the four scripts.
    rows=[];ledger=['# v2final 台本の全行照合（本体は実行しない）','', '行番号は e96d36417。直接指す E2E のコードは存在と旧コミット同一性を照合。外部 PSELDNets の配置・入力・GPU実行は別条件。','']
    for base in ['v2final_gen.sbatch','v2final_train.sbatch','v2final_causal.sbatch','v2final_conf.sbatch']:
        n='server_sde/'+base;s=read(new/n)
        for i,l in enumerate(s.splitlines(),1):
            for v in re.findall(r'(?:scripts|server_sde)/[\w.-]+\.(?:py|sh)',l):
                p=new/v
                rows.append(dict(source=n,line=i,target=v,exists=p.exists(),same_as_before=p.exists() and (old/v).exists() and p.read_bytes()==(old/v).read_bytes()))
        ledger+=['## '+n,'','旧コミットとバイト一致: '+str((old/n).read_bytes()==(new/n).read_bytes()),'','```text']
        ledger+=[f'{i}: {l}' for i,l in enumerate(s.splitlines(),1)];ledger+=['```','']
    write('v2final_literal_paths.json',rows);write('v2final_all_lines.md','\n'.join(ledger))
    print('V2FINAL literal references',sum(r['exists'] for r in rows),'/',len(rows),'missing',[r for r in rows if not r['exists']],flush=True)
    # Safe observed archive failure: both fail before data reads or writes at the
    # documented direct invocation when PYTHONPATH does not inject scripts/.
    res=[]
    for n in ['scripts_archive/_test_v11_peak_resample.py','scripts_archive/step19_dist_labels_eval.py']:
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1';env.pop('PYTHONPATH',None)
        r=subprocess.run([sys.executable,'-B',str(new/n),'--help'],cwd=new,env=env,capture_output=True,timeout=60)
        res.append(dict(path=n,rc=r.returncode,stdout=dec(r.stdout),stderr=dec(r.stderr)))
    write('archive_direct_launches.json',res);print('ARCHIVE LAUNCHES',res,flush=True)

def backup_test():
    # The exact Git baseline lacks this test and its audit module. Reconstruct
    # only these two immutable backup inputs, without altering the Git snapshot.
    wanted=['scripts/test_audit_v18_platoon_20260909.py','scripts/audit_v18_platoon_20260909.py']
    base=OUT/'backup_missing_test/outdoor_seld_e2e';records=[]
    with zipfile.ZipFile(next(BACKUP.glob('*.zip'))) as z:
        for n in wanted:
            key=n if n in z.namelist() else PREFIX+n;b=z.read(key)
            p=base/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
            records.append(dict(path=n,backup_sha=sha(b),current_sha=sha((ROOT/n).read_bytes())))
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(base/'scripts')
    r=subprocess.run([sys.executable,'-B',str(base/wanted[0])],cwd=base,env=env,capture_output=True,timeout=60)
    result=dict(records=records,rc=r.returncode,stdout=dec(r.stdout),stderr=dec(r.stderr))
    write('backup_missing_test.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

def summary():
    r=load('test_results.json')
    out=[]
    for x in r:
        n=Path(x['test']).stem;a=read(OUT/f'tests/{n}_before.normalized.txt');b=read(OUT/f'tests/{n}_after.normalized.txt')
        if a!=b:
            write(f'tests/{n}.diff',''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='before',tofile='after')))
        out.append(dict(test=x['test'],before=x['before']['rc'],after=x['after']['rc'],normalized_equal=a==b,raw_equal=x['before']['raw_sha']==x['after']['raw_sha'],before_tail=x['before']['tail'],after_tail=x['after']['tail']))
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
    print('DOCS',Counter(x['kind'] for x in load('md_full_comparison.json')),Counter(x['kind'] for x in load('md_backup_comparison.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['final_static','backup_test','summary']);a=p.parse_args();globals()[a.mode]()
