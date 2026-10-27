#!/bin/bash
# 配備（本人の許可 15:08「慎重にお願いします」）: 一覧 → tar でサーバの同じ相対パスへ → サーバ側で sha の台帳 → 手元の sha と照合。
# 使い方: bash deploy_lists.sh <一覧> <台帳名>
# 注意（15:10 の事故の教訓）: リモートのコマンドに渡すファイル一覧は 1 行（空白区切り）にする。改行が入ると 2 行目以降が別のコマンドとして実行される。
set -u
cd /c/Users/satos/research/outdoor_seld_e2e || exit 1
LIST="$1"; LEDGER="$2"
S="/c/Users/satos/AppData/Local/Temp/claude/c--Users-satos-research/cec807fb-e518-4328-9066-69d11c9ff819/scratchpad"
FILES=$(grep -v '^#' "$LIST" | sed '/^[[:space:]]*$/d' | sort -u | tr '\n' ' ')
case "$FILES" in *$'\n'*) echo "一覧に改行が残っている。中止"; exit 9;; esac
n=0; miss=0
for f in $FILES; do [ -f "$f" ] || { echo "手元に無い: $f"; miss=$((miss+1)); }; n=$((n+1)); done
[ $miss -eq 0 ] || { echo "手元に無いファイルがある。中止"; exit 2; }
echo "配備する本数: $n（$LIST → $LEDGER）"
DIRS=$(for f in $FILES; do dirname "$f"; done | sort -u | tr '\n' ' ')
ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cd ~/research/outdoor_seld_e2e && mkdir -p $DIRS" || exit 3
tar -cf - $FILES | ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cd ~/research/outdoor_seld_e2e && tar -xf -" || exit 4
ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cd ~/research/outdoor_seld_e2e && sha256sum $FILES > server_sde/$LEDGER && sha256sum --quiet -c server_sde/$LEDGER && echo \"サーバ側の照合 OK（\$(wc -l < server_sde/$LEDGER) 本）\"" || { echo "サーバ側の照合 NG"; exit 5; }
# 手元の sha と突き合わせ（sha256sum の表記の違い〔手元は * 付き〕をそろえて比べる）
sha256sum $FILES | sed 's/ \*/  /' | sort > "$S/${LEDGER}.local"
ssh -o ConnectTimeout=10 -o BatchMode=yes is-server "cat ~/research/outdoor_seld_e2e/server_sde/$LEDGER" | sed 's/ \*/  /' | sort > "$S/${LEDGER}.server"
if diff "$S/${LEDGER}.local" "$S/${LEDGER}.server" > "$S/${LEDGER}.diff"; then
  echo "手元とサーバの sha は $n 本とも一致"
else
  echo "手元とサーバで sha が違う:"; head -20 "$S/${LEDGER}.diff"; exit 6
fi
