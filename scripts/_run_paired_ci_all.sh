#!/bin/bash
# v18 と v19（4 場面）の対応付き差の区間を全部まとめて出す（2026-09-09）。出力が既にあるものは飛ばす。
#   setsid nohup bash scripts/_run_paired_ci_all.sh > out/audit_rescore/_log_paired_ci_all.txt 2>&1 &
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/satos/research/outdoor_seld_e2e
PY=/c/Users/satos/research/DynamicSound/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
P18=out/dataset_outdoor_siren_v18/plan/assignment_v18.csv
G18=out/dataset_outdoor_siren_v18/metadata_dist
G19=out/dataset_outdoor_siren_v19/metadata_dist
M12=out/dataset_outdoor_siren_v12/plan/manifest_fold2_val.csv
GH=out/dataset_outdoor_siren_v12/metadata_dist_h
run() {   # run <出力md> <plan> <meta> <items...>
  local out=$1; shift
  if [ -s "$out" ] && grep -q "含" "$out"; then echo "skip $out"; return 0; fi
  echo "== $(date '+%H:%M:%S') $out"
  $PY scripts/_paired_ci.py "$out" --plan "$1" --meta "$2" "${@:3}"
}
sp=""; sc=""
for e in e099 e119 e139 e149; do
  sp="$sp v18_${e}=out/v17b/C/infer_v17bft_${e}_selfcausal.csv,out/v18/C/infer_v18ft_${e}_selfcausal.csv"
  sc="$sc v18_${e}=out/v17b/C/infer_v17bft_${e}_v12causal.csv,out/v18/C/infer_v18ft_${e}_v12causal.csv"
done
run out/audit_rescore/paired_ci_v18_self.md $P18 $G18 $sp
run out/audit_rescore/paired_ci_v18_cross.md $M12 $GH $sc
for s in platoon fast turn static; do
  a=""; c=""
  for e in e099 e119 e139 e149; do
    a="$a ${s}_${e}=out/v19/C/infer_v18ft_${e}_v19${s}causal.csv,out/v19/C/infer_v19${s}ft_${e}_selfcausal.csv"
    c="$c ${s}_${e}=out/v18/C/infer_v18ft_${e}_v12causal.csv,out/v19/C/infer_v19${s}ft_${e}_v12causal.csv"
  done
  run out/audit_rescore/paired_ci_v19_${s}_new.md out/dataset_outdoor_siren_v19/plan/assignment_v19_${s}.csv $G19 $a
  run out/audit_rescore/paired_ci_v19_${s}_self.md $P18 $G18 $a
  run out/audit_rescore/paired_ci_v19_${s}_cross.md $M12 $GH $c
done
echo "PAIRED_CI_ALL_DONE $(date '+%H:%M:%S')"
