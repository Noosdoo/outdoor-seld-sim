"""Check delivered prose against numerical evidence and resolve local links."""
from audit_v20_reproduce_20260911 import *

def main():
    docs=[ROOT/'md/audit'/name for name in ['v20結果とv2仕様_独立監査_2026-09-11.md','v2_ckpt選択器_実装前仕様レビュー_2026-09-11.md','v2実測固定と規程照会_補足レビュー_2026-09-11.md']]+[OUT/'README.md']
    # Fully specify the previously illustrative tie key, in the new audit proposal only.
    p=docs[1];s=p.read_text(encoding='utf-8')
    s=s.replace('入力を`(frame,class,track,azimuth,elevation,distance)`等の宣言済みキーで正規化して並べ', '入力を各clip内の`(frame,class,track,azimuth,horizontal_distance)`の辞書順で並べ（整数3列、float64の方位・距離。距離NaNを先に対応候補から除く。elevationは検査するが水平距離の対応順には使わない）')
    p.write_text(s,encoding='utf-8')
    ci=json.loads((OUT/'ci_results.json').read_text(encoding='utf-8'));names={'b':'v20b','a':'v20a','v18':'v18'};metrics={'強到達':'strong','至近到達':'crit','至近検出率':'det_close','全帯距離誤差':'dist_err'};checked=0
    for line in docs[0].read_text(encoding='utf-8').splitlines():
        m=re.match(r'^\| (b|a)−(v18|a) \| (\d+) \| ([^|]+) \| ([^|]+) \| \[([^,]+), ([^\]]+)\] \|$',line)
        if not m:continue
        cand,base,e,metric,delta,lo,hi=m.groups();metric=metrics[metric.strip()]
        v=next(x for x in ci if x['dataset']=='platoon' and x['name']==names[cand] and x['base']==names[base] and x['epoch']==int(e) and x['metric']==metric)
        for written,actual in zip([delta,lo,hi],[v['delta']]+v['ci95']):assert abs(float(written.strip().replace('−','-'))-actual)<.00000051,(line,actual)
        checked+=1
    assert checked==7,checked
    links=0
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if target.startswith(('https://','http://','#')):continue
            assert (p.parent/target.split('#')[0]).exists(),(p,target);links+=1
    files=docs+sorted((ROOT/'scripts').glob('audit_*_20260911*.py'))+sorted(p for p in OUT.glob('*.json') if p.name!='delivery_validation.json')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    dump(OUT/'delivery_validation.json',dict(ci_rows_in_report_verified=checked,local_links_resolved=links,sha256=manifest,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    print('DELIVERY_VERIFIED',checked,links,len(manifest))

if __name__=='__main__':main()
