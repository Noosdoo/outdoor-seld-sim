"""Field recording orientation (#44): independent of audio alignment --yaw."""
import math

ORIENTATION_COL = "三脚の向き"
CHECK_COL = "回転後点検原本"
HEADINGS = (0, 90, 180, 270)   # 0 = ロゴが +x（南）・90 = +y（東・上から見て反時計回り）・180 = −x（北）・270 = −y（西）。90／270 はクラクション E2 だけ（#69 r40・Astra H25-01・2026-10-25）


def tripod_heading(row):
    """Explicit 0/90/180/270 wins; missing/blank legacy fields use the note chip.
    方位の式（step19k: wrap(方位_現場 − heading)）はどの向きでも正しい一般形なので、ここは受け付ける値を決めるだけ（90／270 を 0／180 に書き換えない）。"""
    raw = str(row.get(ORIENTATION_COL, "") or "").strip()
    if not raw:
        return 180 if "三脚180°" in (row.get("特記", "") or "") else 0
    try:
        value = float(raw.removesuffix("°").strip())
    except ValueError:
        raise ValueError(f"{ORIENTATION_COL}: 0/90/180/270 が必要です: {raw!r}") from None
    if not math.isfinite(value) or value not in HEADINGS:
        raise ValueError(f"{ORIENTATION_COL}: 0/90/180/270 が必要です: {raw!r}")
    if value in (90, 270) and not script_id(row).startswith("E2"):
        # 2026-10-25（敵対レビュー 軽 1）: 90／270 はクラクション E2 だけ（#69 r40）。ほかの本（A2・C2・H2・歩く比較・台本ID の無い本）は 0／180
        raise ValueError(f"{row.get('session_id', '')}/take{row.get('take_id', '')}: {ORIENTATION_COL} {raw} はクラクション E2 だけ"
                         f"（台本ID {script_id(row) or '空'}・ほかの本は 0/180）")
    return int(value)


# ---- 歩く比較の置き方（変更票 #69 §00 の 4・Astra V88-02・R97-01・2026-10-20 B2） ----
WALKER_LINE_COL = "歩く人の線_m"        # 記録シート Version 88〜: 歩く人（本人）の線 [m]（既定 5.0・車 5 m の組は 7.0）
STATIC_X_COL = "歩く人の静止x_m"         # 記録シート Version 98〜: 静止の本の本人の x [m]（車の組 南向き +2・北向き −2・自転車の組 0・歩行は空）
SCRIPT_ID_COL = "台本ID"


def script_id(row) -> str:
    return str(row.get(SCRIPT_ID_COL, "") or "").strip().upper()


def _w_no(sid: str):
    return int(sid[1:3]) if sid.startswith("W") and sid[1:3].isdigit() else None


def walker_line_m(row) -> str:
    """本人（マイク）の線 [m]。記録シートの 歩く人の線_m があればそれ。空なら台本 ID から: W13〜W16（車 5 m）= 7.0・ほかの W = 5.0・歩く比較でない本 = 0（原点）。"""
    v = str(row.get(WALKER_LINE_COL, "") or "").strip()
    if v:
        return f"{float(v):g}"
    n = _w_no(script_id(row))
    if n is None:
        return "0"
    return "7" if 13 <= n <= 16 else "5"          # :g の表記にそろえる（7.0 → 7・5.0 → 5）


def static_mic_x(row, warns=None) -> str:
    """静止の本の本人の x [m]（現場の座標・南が +）。記録シートの 歩く人の静止x_m があればそれ。
    空なら台本 ID と向きから同じ決まりで補う: 車の組（W09〜W16）は 三脚の向き 0（南向き）= +2・180（北向き）= −2、自転車の組（W01〜W08）と歩く比較でない本は 0。
    補ったときは warns に残す（10/29 の記録シート V98 で記録するのが正）。"""
    v = str(row.get(STATIC_X_COL, "") or "").strip()
    if v:
        return f"{float(v):g}"
    n = _w_no(script_id(row))
    if n is None or n <= 8:
        return "0"
    h = tripod_heading(row)
    if h not in (0, 180):                       # 歩く比較の向きは 0／180 だけ（90／270 を黙って −2 にしない・地図 §8-3・2026-10-25）
        raise ValueError(f"{row.get('session_id', '')}/take{row.get('take_id', '')}: 歩く比較 {script_id(row)} の三脚の向き {h} は 0/180 のどちらか")
    x = "2" if h == 0 else "-2"
    if warns is not None:
        warns.append(f"{row.get('session_id', '')}/take{row.get('take_id', '')}: {STATIC_X_COL} が空 → 台本 {script_id(row)} と三脚の向き {h} から {x} m を補った（記録シート V98 で記録するのが正）")
    return x


# ---- マイクの現場の座標（記録シート Version 123・Astra H25-01・地図 §8-3 の候補 A・2026-10-25 Opus・Fable の確認待ち） ----
MIC_X_COL = "マイク_x_m"   # 三脚の本だけ: マイクの現場の座標（E2 クラクションは (0, +2.0)・ほかの三脚の本は (0, 0)）。ヘルメットの本は空
MIC_Y_COL = "マイク_y_m"


def mic_field_xy(row, state: str = "", warns=None) -> tuple[str, str]:
    """マイク（三脚・本人）の現場の座標 (x, y) [m] を文字列で返す（step19d の注釈の mic_x_m・mic_y_m）。
    - 記録シート Version 123〜 の三脚の本: 列 マイク_x_m・マイク_y_m（両方そろって有限の数）。x_m・y_m（音源）は現場の座標のまま渡し、
      差（音源 − マイク）は step19k の幾何の関数の中で 1 回だけ取る（ここでは引かない＝原点を二重に引かない）。
    - 列が無い・空（ヘルメットの本・Version 122 までの CSV）: 今までの歩く比較の決まり（歩行は x 空・y = 歩く人の線、静止は x = 歩く人の静止x_m）。歩く比較でない本は (0, 0)。
    - 歩行の本にマイクの列が書かれていたら使わない（人の位置は動画の mic の読みから）。warns に残す。"""
    mx = str(row.get(MIC_X_COL, "") or "").strip()
    my = str(row.get(MIC_Y_COL, "") or "").strip()
    if mx or my:
        if state == "歩行":
            if warns is not None:
                warns.append(f"{row.get('session_id', '')}/take{row.get('take_id', '')}: 歩行の本に {MIC_X_COL}／{MIC_Y_COL} がある → 使わない（歩く人の位置は動画の mic の読みから）")
        else:
            if not (mx and my):
                raise ValueError(f"{row.get('session_id', '')}/take{row.get('take_id', '')}: {MIC_X_COL}・{MIC_Y_COL} は両方書く（{mx!r}・{my!r}）")
            try:
                x, y = float(mx), float(my)
            except ValueError:
                raise ValueError(f"{row.get('session_id', '')}/take{row.get('take_id', '')}: {MIC_X_COL}・{MIC_Y_COL} が数でない（{mx!r}・{my!r}）") from None
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"{row.get('session_id', '')}/take{row.get('take_id', '')}: {MIC_X_COL}・{MIC_Y_COL} が有限でない（{mx!r}・{my!r}）")
            return f"{x:g}", f"{y:g}"
    return ("" if state == "歩行" else static_mic_x(row, warns)), walker_line_m(row)
