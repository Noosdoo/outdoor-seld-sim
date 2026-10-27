"""Compute final denominators and preserve review evidence; only writes audit OUT."""
from audit_cleanup_final_20260913 import *

def main():
    md=load('md_full_comparison.json');backup=load('md_backup_comparison.json')
    # The index moved before the cleanup commits but was uncommitted at 188121ae2.
    index='study/読解の索引_生きているコード_2026-09-09.md'
    corrected=[dict(r,kind='index_edited',old='md/'+index) if r['path']==index else r for r in md]
    write('md_final_classification.json',corrected)
    r=load('test_results.json');counts={}
    for x in r:
        name=Path(x['test']).stem;s=read(OUT/f'tests/{name}_after.stdout.txt')+'\n'+read(OUT/f'tests/{name}_after.stderr.txt')
        m=re.search(r'Ran (\d+) tests',s) or re.search(r'ALL PASS \((\d+) checks\)',s) or re.search(r'(\d+)/(\d+) ALL PASS',s) or re.search(r'(\d+) checks, 0 failed',s)
        counts[x['test']]=int(m[1]) if m else 0
    remote=load('remote_evidence.json');prov=load('provenance_manifest_keys.json')
    source=load('remote_source_comparison.json');v=load('inventory.json')
    before=snapshot('before');after=snapshot('after')
    direct=load('v2final_literal_paths.json')
    modified_server=[n.relative_to(after).as_posix() for n in (after/'server_sde').iterdir() if n.suffix in ['.py','.sh','.sbatch'] and (before/n.relative_to(after)).exists() and n.read_bytes()!=(before/n.relative_to(after)).read_bytes()]
    native=[]
    for n in modified_server:
        a=read(before/n);b=read(after/n)
        native.append(dict(path=n,only_archive_path=a==b.replace('scripts_archive/','scripts/')))
    write('server_changes.json',native)
    result=dict(
        markdown_counts=dict(Counter(x['kind'] for x in corrected)),
        markdown_existing=sum(x['kind']!='added' for x in corrected),
        markdown_result_records=[x for x in corrected if x['path'].startswith('md/results/')],
        backup_exceptions=[x for x in backup if x['kind']!='path_only'],
        test_counts=counts,total_checks=sum(counts.values()),extra_doppler_checks=2,
        strict_normalized_pairs=sum(x['normalized_equal'] for x in r),raw_pairs=sum(x['before']['raw_sha']==x['after']['raw_sha'] for x in r),
        provenance_key_shapes=dict(Counter((sum('/src/' in '/'+k for k in x['keys']),sum(k.endswith('causal_patch.py') for k in x['keys'])) for x in prov)),
        remote_code=dict(total=len(source),missing=sum(x['actual'] is None for x in source),different=sum(x['actual'] is not None and x['actual']!=x['expected'] for x in source),matching=sum(x['actual']==x['expected'] for x in source)),
        v2final_unique_paths=len({x['target'] for x in direct}),v2final_unique_existing=len({x['target'] for x in direct if x['exists']}),
        runtime_hashes=remote['runtime_hashes'],server_source_changes=native,
        provenance_five=v['provenance'],
    )
    result['provenance_key_shapes']={str(k):val for k,val in result['provenance_key_shapes'].items()}
    write('report_data.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    # Preserve a proposed historic-record restoration; do not apply it.
    patch=[];targets=[]
    for r in corrected:
        n=r['path']
        if r['kind']!='two_paths_only' or not n.startswith(('md/audit/','md/results/')) or r.get('old')!=n:continue
        a=read(after/n);b=read(before/n)
        if a==b:continue
        targets.append(n)
        patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+PREFIX+n,tofile='b/'+PREFIX+n))
    write('proposed_historic_paths_restore.patch',''.join(patch));write('proposed_historic_restore_targets.json',targets)
    print('HISTORIC RESTORATION PROPOSED ONLY',len(targets),flush=True)
    # Fresh SHA256 inventory of audit programs and all checked source files.
    audit_files=list((ROOT/'scripts').glob('audit_*_20260913*.py'))
    write('audit_script_hashes.json',[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p.read_bytes())) for p in audit_files])

if __name__=='__main__':main()
