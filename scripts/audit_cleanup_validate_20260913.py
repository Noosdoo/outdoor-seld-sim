"""Final artifact QA. May correct audit artifacts, never production inputs."""
from audit_cleanup_checks_20260913 import *

def main():
    report=ROOT/'md/audit/フォルダ整理の影響_独立監査_2026-09-13.md'
    s=read(report)
    corrections={
        '置き場所の点検_2026-09-12.md:29':'置き場所の点検_2026-09-12.md:28',
        'scripts/test_realsmoke_v2_units.py:908` 付近の stderr 末尾の表示':'scripts/test_realsmoke_v2_units.py:912` の stdout＋stderr 末尾 300 文字の表示',
        ':49 sde_patch、:53 height_patch、:57 `train.main`':':50 sde_patch、:54 height_patch、:58 `train.main`',
        'sde_patch :57、causal_patch :61、height_patch :65、train :69 へ至る経路（正確な定義位置は同ファイル全文参照）':'sde_patch :54、causal_patch :58、height_patch :62、train :66 へ至る経路',
        '`:319` 付近に予算内訳':'`:318` に予算内訳',
    }
    for a,b in corrections.items():s=s.replace(a,b)
    cmd='& $pyAudit -B outdoor_seld_e2e/scripts/audit_cleanup_remote_20260913.py'
    anchor='& $pyAudit -B outdoor_seld_e2e/scripts/audit_cleanup_probes_20260913.py probes'
    if s.index(cmd)>s.index(anchor):
        s=s.replace('# サーバは読み取りのみ。probes のサーバ比較部分より先に取得する場合もある。\n'+cmd+'\n','')
        s=s.replace(anchor,'# サーバは読み取りのみ。\n'+cmd+'\n'+anchor)
    s=s.replace('probes は既存の `remote_source_comparison.json` も読むため、新しい環境から全順序を再実行するときは remote を probes より先に実行する。','probes は remote の取得結果も読むため、上記の順序で実行する。最後に `audit_cleanup_validate_20260913.py` で証拠のハッシュ・リンク・復元パッチを検証できる。')
    report.write_text(s,encoding='utf-8',newline='\n')
    # Earlier text output used Windows newline translation. Restore exact
    # captured stdout/stderr bytes only when their recorded hash verifies it.
    verified=[]
    for r in load('test_results.json'):
        for tag in ['before','after']:
            stem=Path(r['test']).stem
            ps=[OUT/f'tests/{stem}_{tag}.{stream}.txt' for stream in ['stdout','stderr']]
            bs=[p.read_bytes() for p in ps];expected=r[tag]['raw_sha']
            if sha(bs[0]+b'\n<STDERR>\n'+bs[1])!=expected:
                restored=[b.replace(b'\r\n',b'\n') for b in bs]
                assert sha(restored[0]+b'\n<STDERR>\n'+restored[1])==expected,(stem,tag)
                for p,b in zip(ps,restored):p.write_bytes(b)
            verified.append(dict(test=r['test'],tag=tag,sha256=expected))
    # Make future audit runs write UTF-8 evidence without newline translation.
    utility=ROOT/'scripts/audit_cleanup_20260913.py'
    src=read(utility)
    original="p.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')"
    updated="p.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8', newline='\\n')"
    if original in src:
        utility.write_text(src.replace(original,updated),encoding='utf-8',newline='\n')
    inv=load('inventory.json')
    unchanged=[dict(path=r['path'],unchanged_since_inventory=sha((ROOT/r['path']).read_bytes())==r['current_sha']) for r in inv['keep']]
    assert all(r['unchanged_since_inventory'] for r in unchanged)
    provenance=[dict(path=r['path'],unchanged=sha((ROOT/r['path']).read_bytes())==r['current']) for r in inv['provenance']]
    assert all(r['unchanged'] for r in provenance)
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',s):
        if target.startswith(('http:','https:')):continue
        p=(report.parent/target).resolve();links.append(dict(target=target,exists=p.exists()))
    assert all(x['exists'] for x in links),links
    programs=list((ROOT/'scripts').glob('audit_*_20260913*.py'))
    for p in programs:ast.parse(read(p),filename=str(p))
    patch=OUT/'proposed_historic_paths_restore.patch'
    result=subprocess.run(['git','apply','--check','--',str(patch)],cwd=REPO,capture_output=True)
    assert result.returncode==0,dec(result.stderr)
    write('final_validation.json',dict(report=report.name,report_sha256=sha(report.read_bytes()),raw_test_logs_verified=len(verified),raw_test_logs=verified,keep_unchanged=unchanged,provenance_unchanged=provenance,links=links,audit_programs_parsed=len(programs),proposed_patch_check=dict(rc=result.returncode,stdout=dec(result.stdout),stderr=dec(result.stderr))))
    write('audit_script_hashes.json',[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p.read_bytes())) for p in programs])
    print('VALIDATED raw log pairs',len(verified),'keep',len(unchanged),'provenance',len(provenance),'links',len(links),'audit programs',len(programs),'patch rc',result.returncode,flush=True)

if __name__=='__main__':main()
