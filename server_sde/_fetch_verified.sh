#!/bin/bash
# Shared v18/v19 transport. Every required run is verified, including cached runs.
# Callers set SSH (array), RUNS, REMOTE_VERIFY, PY, FETCH_DIR, and LOCAL_VERIFY.
fetch_one() {
  local n=$1 exp=$2 dest=${3:-$FETCH_DIR} v sha actual cmd
  [[ "$n" =~ ^infer_[A-Za-z0-9_]+$ ]] || { echo "NG run name: $n"; return 1; }
  printf -v cmd 'bash %q %q %q --repair' "$REMOTE_VERIFY" "$RUNS/$n" "$exp"
  if ! v=$("${SSH[@]}" "$cmd"); then echo "NG remote verification: $n"; return 1; fi
  [[ "$v" =~ ^(OK|REBUILT)[[:space:]]n_files=([0-9]+)[[:space:]]n_rows=([0-9]+)[[:space:]]sha256=([0-9a-f]{64})$ ]] || { echo "NG verification response: $n $v"; return 1; }
  [ "${BASH_REMATCH[2]}" = "$exp" ] || return 1
  sha=${BASH_REMATCH[4]}
  mkdir -p "$dest" || return 1
  actual=""
  [ ! -f "$dest/$n.csv" ] || actual=$(sha256sum "$dest/$n.csv" | cut -d' ' -f1) || return 1
  # Never trust a sidecar hash instead of hashing the actual cached CSV.
  if [ "$actual" != "$sha" ]; then
    scp -q "is-server:$RUNS/$n/val_all_causal.csv" "$dest/$n.csv.tmp" || return 1
    actual=$(sha256sum "$dest/$n.csv.tmp" | cut -d' ' -f1) || return 1
    [ "$actual" = "$sha" ] || { echo "NG download digest: $n"; return 1; }
    mv "$dest/$n.csv.tmp" "$dest/$n.csv" || return 1
  fi
  scp -q "is-server:$RUNS/$n/val_all_causal.done.json" "$dest/$n.done.json.tmp" || return 1
  scp -q "is-server:$RUNS/$n/run_meta.json" "$dest/$n.run_meta.json.tmp" || return 1
  "$PY" "$LOCAL_VERIFY" "$dest/$n.csv" "$exp" --download \
    --done "$dest/$n.done.json.tmp" --meta "$dest/$n.run_meta.json.tmp" || return 1
  mv "$dest/$n.done.json.tmp" "$dest/$n.done.json" || return 1
  mv "$dest/$n.run_meta.json.tmp" "$dest/$n.run_meta.json" || return 1
  # 実行記録（GPU・ノード・ドライバ・cuDNN・producer ごとの処理数・依頼 2-3）。identity には入らない。2026-09-14 より前の run には無い
  # 来歴の照合には使わない付随物なので、無ければ注記だけ（取得の成否は上の照合で決まる）
  scp -q "is-server:$RUNS/$n/run_exec.jsonl" "$dest/$n.run_exec.jsonl" 2>/dev/null || echo "note: $n に run_exec.jsonl が無い（2026-09-14 より前の推論）"
  printf '%s\n' "$sha" > "$dest/$n.sha256" || return 1
  echo "verified $n ($v)"
}
