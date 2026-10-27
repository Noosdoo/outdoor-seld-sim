#!/bin/bash
# 初期重み（PSELDNets の事前学習 HTSAT checkpoint）を入手して sha256 で照合し、submodule の ckpts/ に置く（依頼 2-7・Astra R3・2026-09-14）。
# 第三者がクリーン環境で再現を始めるための入口。gitlink（8092a14）にはこの重みは含まれない。
#
# 使い方:  bash server_sde/_fetch_initial_weights.sh [--check-only]
#   入手元は PSELDNets の README が指す Hugging Face の **datasets** リポジトリ（匿名で取れる・2026-09-14 14:58 確認: 302 → 140,516,864 byte、
#   X-Linked-ETag = 813083ac…）。最初に試した models 名前空間の URL（huggingface.co/Jinbo-HU/PSELDNets）は存在せず 401/404 だった。
#   HF_TOKEN は不要だが、あれば Authorization に付ける。入手元の URL は md/design/初期重みの入手手順_2026-09-14.md §1 と同じ。変えるならそちらも直す。
# 出力: PSELDNet/PSELDNets/ckpts/mACCDOA-HTSAT-0.567.ckpt（140,516,864 byte・sha256 813083ac…）。DEST=<path> で置き場所を変えられる（試験用）
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PS=${PS:-$ROOT/../PSELDNet/PSELDNets}
DEST=${DEST:-$PS/ckpts/mACCDOA-HTSAT-0.567.ckpt}
EXPECT_SHA=813083ac938c5974a6f36ceca29ea66c0382091db5df1d6d47ece9572d5ac71b
EXPECT_SIZE=140516864
URL=${HTSAT_URL:-https://huggingface.co/datasets/Jinbo-HU/PSELDNets/resolve/main/model/mACCDOA-HTSAT-0.567.ckpt}
check() {   # 存在・大きさ・sha256 の 3 つを見る
  [ -f "$DEST" ] || { echo "NG: $DEST が無い"; return 1; }
  local size sha
  size=$(stat -c %s "$DEST" 2>/dev/null || stat -f %z "$DEST")
  sha=$(sha256sum "$DEST" | cut -d' ' -f1)
  [ "$size" = "$EXPECT_SIZE" ] || { echo "NG: 大きさ $size ≠ $EXPECT_SIZE"; return 1; }
  [ "$sha" = "$EXPECT_SHA" ] || { echo "NG: sha256 $sha ≠ $EXPECT_SHA"; return 1; }
  echo "OK: $DEST size=$size sha256=$sha"
}
if check 2>/dev/null; then exit 0; fi
[ "${1:-}" = "--check-only" ] && { check; exit 1; }
mkdir -p "$(dirname "$DEST")"
TMP=$DEST.part
hdr=()
[ -n "${HF_TOKEN:-}" ] && hdr=(-H "Authorization: Bearer $HF_TOKEN")
echo "取得: $URL"
curl -sS -L --fail --retry 3 "${hdr[@]}" -o "$TMP" "$URL" || { rm -f "$TMP"; echo "NG: 取得に失敗（URL が変わった可能性。手順書 §1 を見る）"; exit 2; }
mv "$TMP" "$DEST"
check
