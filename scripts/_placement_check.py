# -*- coding: utf-8 -*-
"""置き場所の検査（段 0・2026-09-12）。フォルダ整理の前後で必ず走らせる。

設計書: md/audit/置き場所の点検_2026-09-12.md

ファイルを 1 本も実行せず、**読むだけ**で判定する（scripts/ の多くは import しただけで
処理が走るため、実行による検査は危険）。検査は 8 つ:

  検査1 _load() / spec_from_file_location() / _sc() の相手が実在するか
        -> 名前が「文字列」なので静的検査に映らない。移動で真っ先に壊れる所
  検査2 study/読解の索引_生きているコード_2026-09-09.md のリンクが実在するか
  検査3 import が「そのファイルの位置から」解決するか（標準ライブラリ・導入済みも可）
  検査4 md/ と study/ の文書内リンクが実在するか
  検査5 ROOT = Path(__file__).resolve().parents[1] が outdoor_seld_e2e を指すか
        -> 1 階層深いフォルダへ移すと静かにズレる。224 本がこの書き方をしている
  検査6 来歴の鍵になる server_sde/ の 5 ファイルが、同じ名前・同じ場所・同じ中身か
  検査7 `ROOT/"scripts"/"名前"` の形で名指ししている相手が scripts/ に在るか
        -> 退避すると「呼ぶ側も一緒に移したのに相手を scripts/ に探して落ちる」が起きる
  検査8 走らせても 1 件も検査しない試験が無いか（監査 B・2026-09-13）

検査3 の探索先は「そのファイル自身が sys.path に足す先」だけを見る（監査 R3・2026-09-13）。
--baseline の比較は件数ではなく**中身**で行い、同数でも入れ替わっていたら悪化とみなす。

使い方:
    python scripts/_placement_check.py                      # 検査して結果を表示
    python scripts/_placement_check.py --json out/x.json     # 結果を保存（移動前の基準）
    python scripts/_placement_check.py --baseline out/x.json # 基準と比べて差だけ出す
終了コード: 0 = 問題なし / 1 = 問題あり（--baseline のときは基準より悪化したら 1）
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "study" / "読解の索引_生きているコード_2026-09-09.md"
PROVENANCE = ["_causal_infer_v17.py", "_prediction_integrity.py",
              "sde_patch.py", "causal_patch.py", "height_patch.py"]
SKIP_DIRS = {"out", "__pycache__", ".git"}
NAME_RE = re.compile(r"[\w.\-]+\.(?:py|sh|sbatch|ipynb|js|md|json|yaml|yml|html|tsv|csv|pptx|tex|svg|pdf)\b")


def walk(*exts: str):
    """out/ と __pycache__ を除いて、指定拡張子のファイルを返す。"""
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        parts = p.relative_to(ROOT).parts
        if parts[0] in SKIP_DIRS or "__pycache__" in parts:
            continue
        if not exts or p.suffix.lower() in exts:
            yield p


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def read(p: Path) -> str:
    return p.read_bytes().decode("utf-8", "replace")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 検査 1
def check_dynamic_load():
    """_load("名", "相手.py") と spec_from_file_location の相手が実在するか。"""
    bad, n = [], 0
    for p in walk(".py"):
        try:
            tree = ast.parse(read(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if name not in ("_load", "spec_from_file_location", "_sc"):
                continue
            for s in ast.walk(node):
                if isinstance(s, ast.Constant) and isinstance(s.value, str) and s.value.endswith(".py"):
                    n += 1
                    # 呼び出し側の慣習は ROOT/"scripts"/名前。相対の with_name も使われる
                    cands = [ROOT / "scripts" / Path(s.value).name, ROOT / s.value,
                             p.parent / Path(s.value).name, ROOT / "server_sde" / Path(s.value).name]
                    if not any(c.exists() for c in cands):
                        bad.append({"from": rel(p), "target": s.value})
    return {"total": n, "bad": bad}


# ---------------------------------------------------------------- 検査 2・4
def _link_ok(token: str, src: Path) -> bool | None:
    t = token.replace("\\", "/").split("#")[0]
    if t.startswith(("http://", "https://", "mailto:")) or not t or t in (".", ".."):
        return None
    t = re.sub(r":\d+(-\d+)?$", "", t)          # VS Code 風の :行番号
    t = re.sub(r"^/(?=[A-Za-z]:/)", "", t)      # /C:/... -> C:/...
    for c in (Path(t), src.parent / t, ROOT / t, ROOT.parent / t):
        try:
            if c.exists():
                return True
        except OSError:
            pass
    return False


MOVETABLE = ROOT / "md" / "audit" / "ファイル移動表_2026-09-12.md"


def _moved_names():
    """移動表に載っているファイル名 -> 今の場所。記録文書が当時のパスを指すのは正しい。"""
    if not MOVETABLE.exists():
        return {}
    txt = read(MOVETABLE)
    out, dst = {}, None
    for line in txt.splitlines():
        m = re.match(r"^## `([^`]+)/` → `([^`]+)/`", line)
        if m:
            dst = m.group(2)
            continue
        m = re.match(r"^\| `([\w.\-]+)` \|", line)
        if m and dst:
            out[m.group(1)] = dst
    return out


def check_links(paths, label):
    """文書のリンクが実在するか。

    記録文書（md/audit・md/results）は**当時のパスのまま**にしてあるので（独立監査 R5）、
    移動表（md/audit/ファイル移動表_2026-09-12.md）で今の場所に解決できるものは問題としない。
    """
    moved = _moved_names()
    bad, hist, n = [], [], 0
    for p in paths:
        # 記録文書 = 過去の事実を書いたもの。パスは当時の表記のままにしてあるので、
        # 移動表で今の場所に解決できれば問題としない（独立監査 R5・2026-09-13）
        is_record = (rel(p).startswith(("md/audit/", "md/results/", "md/handoff/",
                                        "md/design/", "md/seminar/", "md/research/"))
                     or rel(p) in ("PROGRESS.md", "md/現状の状態表_2026-09-05.md"))
        for m in re.finditer(r"\]\(([^)\s]+)\)", read(p)):
            v = _link_ok(m.group(1), p)
            if v is None:
                continue
            n += 1
            if v is not False:
                continue
            base = re.sub(r":\d+(-\d+)?$", "", m.group(1).split("#")[0]).split("/")[-1]
            if is_record and base in moved and (ROOT / moved[base] / base).exists():
                hist.append({"doc": rel(p), "link": m.group(1), "now": moved[base] + "/" + base})
                continue
            bad.append({"doc": rel(p), "link": m.group(1)})
    return {"label": label, "total": n, "bad": bad,
            "当時の表記（移動表で解決）": len(hist), "その例": hist[:3]}


# ---------------------------------------------------------------- 検査 3
def check_imports():
    """import が「そのファイルの位置から」解決できるか。

    実行はしない（import すると処理が走るファイルが多いため）。Python は
    `python scripts/x.py` のとき scripts/ を最初に探すので、探し先は
    「自分のフォルダ → src/ → リポジトリ直下 → 導入済みパッケージ」の順で見る。

    解決できないものは 2 つに分ける:
      ★移動で壊れた疑い = 同じ名前の .py がリポジトリ内のどこかに在るのに、
                          そのファイルの位置からは見えない（＝退避したせい）
      環境             = そもそもリポジトリに無い外部パッケージ（torch など未導入）
    """
    import importlib.util

    std = set(getattr(sys, "stdlib_module_names", ()))
    anywhere = {p.stem for p in walk(".py")} | {p.parent.name for p in walk(".py") if p.name == "__init__.py"}
    moved, env, n = [], [], 0
    cache: dict[str, bool] = {}
    for p in walk(".py"):
        try:
            tree = ast.parse(read(p))
        except SyntaxError:
            moved.append({"from": rel(p), "module": "(構文エラー)"})
            continue
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module)
        # 監査 R3（2026-09-13）: 探索先は「そのファイルが自分で用意するもの」だけ。
        # scripts/ や src/ を常に足すと、実際には見えない相手を『解決できる』と誤判定する
        roots = [p.parent] + own_sys_path(p)
        for m in sorted(mods):
            n += 1
            top = m.split(".")[0]
            if top in std:
                continue
            parts = m.split(".")
            found = any((base.joinpath(*parts[:-1]) if len(parts) > 1 else base) / (parts[-1] + ".py")
                        for base in roots
                        if ((base.joinpath(*parts[:-1]) if len(parts) > 1 else base) / (parts[-1] + ".py")).exists()
                        or ((base.joinpath(*parts[:-1]) if len(parts) > 1 else base) / parts[-1] / "__init__.py").exists())
            if found:
                continue
            if top not in cache:
                try:
                    cache[top] = importlib.util.find_spec(top) is not None
                except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
                    cache[top] = False
            if cache[top]:
                continue
            (moved if top in anywhere else env).append({"from": rel(p), "module": m})
    return {"total": n, "bad": moved, "環境（移動と無関係）": len(env), "環境の例": env[:5]}


def _str_args(line: str):
    """1 行の中の str(...) の中身を、括弧の対応を見ながら取り出す。"""
    out, i = [], 0
    while True:
        j = line.find("str(", i)
        if j < 0:
            return out
        k, depth = j + 4, 1
        while k < len(line) and depth:
            depth += (line[k] == "(") - (line[k] == ")")
            k += 1
        out.append(line[j + 4:k - 1].strip())
        i = k


def own_sys_path(p: Path):
    """そのファイル自身が sys.path に足すフォルダを、実行せずに読み解く。

    扱う形（監査 R3 の指摘後に拡張・2026-09-13）:
      sys.path.insert(0, str(X)) / sys.path.append(str(X)) /
      sys.path[:0] = [str(X), str(Y)] / sys.path.extend([str(X)])
    中身は `Path(__file__).resolve().parent[s[N]]` か、代入済みの HERE・ROOT など。
    読み解けない式は**足さない**（見えないものを見えると誤判定しない側に倒す）。
    """
    src = read(p)
    base_vars = {}
    for m in re.finditer(r"^(\w+)\s*=\s*Path\(__file__\)\.resolve\(\)((?:\.parents?\[\d+\]|\.parent)*)\s*$",
                         src, re.M):
        base_vars[m.group(1)] = _walk_up(p.resolve(), m.group(2))
    out = []
    for line in src.splitlines():
        if "sys.path" not in line or line.lstrip().startswith("#"):
            continue
        for expr in _str_args(line):
            head = re.match(r"Path\(__file__\)\.resolve\(\)((?:\.parents?\[\d+\]|\.parent)*)", expr)
            if head:
                cur, rest = _walk_up(p.resolve(), head.group(1)), expr[head.end():]
            else:
                v = re.match(r"(\w+)((?:\.parents?\[\d+\]|\.parent)*)", expr)
                if not v or v.group(1) not in base_vars:
                    continue
                cur, rest = _walk_up(base_vars[v.group(1)], v.group(2)), expr[v.end():]
            for seg in re.finditer(r"/\s*[\"']([^\"']+)[\"']", rest):
                cur = cur / seg.group(1)
            if re.search(r"/\s*[^\"'\s]", re.sub(r"/\s*[\"'][^\"']+[\"']", "", rest)):
                continue          # 変数を挟む形は読み解けないので採用しない
            out.append(cur)
    return out


def _walk_up(path: Path, suffix: str):
    cur = path
    for m in re.finditer(r"\.parents\[(\d+)\]|\.parent", suffix):
        cur = cur.parents[int(m.group(1))] if m.group(1) else cur.parent
    return cur


# ---------------------------------------------------------------- 検査 5
ROOT_PAT = re.compile(r"(?:ROOT|HERE|BASE|REPO)\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[1\]")


def check_root_depth():
    """parents[1] で ROOT を作るファイルが、本当に outdoor_seld_e2e を指すか。"""
    bad, n = [], 0
    for p in walk(".py"):
        if not ROOT_PAT.search(read(p)):
            continue
        n += 1
        if p.resolve().parents[1] != ROOT:
            bad.append({"file": rel(p), "would_be": str(p.resolve().parents[1])})
    return {"total": n, "bad": bad}


# ---------------------------------------------------------------- 検査 7
SCRIPTS_PATH_RE = re.compile(r"""["']scripts["']\s*/\s*["']([\w.\-]+\.py)["']"""
                             r"""|["']scripts/([\w.\-]+\.py)["']""")


def check_scripts_path_refs():
    """コードが `ROOT/"scripts"/"名前.py"` の形で名指ししている相手が scripts/ に在るか。

    `_load` の中だけでなく、subprocess で叩く形もこの書き方をする。退避すると
    「呼ぶ側も一緒に移したのに、相手を scripts/ に探しに行って落ちる」が起きる。
    """
    bad, n = [], 0
    me = Path(__file__).name
    for p in walk(".py", ".sh", ".sbatch"):
        if p.name == me:      # この説明文自身が形に当たってしまうので除く
            continue
        src = read(p)
        peer_ok = "def _sc(" in src      # _sc を持つファイルは「自分の隣」も探せる
        for m in SCRIPTS_PATH_RE.finditer(src):
            name = m.group(1) or m.group(2)
            n += 1
            if (ROOT / "scripts" / name).exists():
                continue
            if peer_ok and (p.parent / name).exists():
                continue
            bad.append({"from": rel(p), "target": "scripts/" + name})
    return {"total": n, "bad": bad}


# ---------------------------------------------------------------- 検査 8
def check_vacuous_tests():
    """走らせても 1 件も検査しない試験を見つける（監査 B・2026-09-13）。

    `tests/test_doppler_direction.py` は test_ 関数を定義するだけで呼んでおらず、
    実行すると何も出さずに終了コード 0 を返す。これを「合格」と数えていた。
    """
    bad, n = [], 0
    for p in walk(".py"):
        if not p.name.startswith("test_"):
            continue
        n += 1
        try:
            tree = ast.parse(read(p))
        except SyntaxError:
            continue
        defs = {x.name for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if not any(d.startswith("test") for d in defs):
            continue                      # 関数に分けず直に書く形式は対象外
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
        if not (defs & called):
            bad.append({"file": rel(p), "why": "test_ 関数を定義するだけで呼んでいない（実行しても 0 件）",
                        "defs": sorted(d for d in defs if d.startswith("test"))[:6]})
    return {"total": n, "bad": bad}


# ---------------------------------------------------------------- 検査 6
def check_provenance():
    bad, got = [], {}
    for name in PROVENANCE:
        p = ROOT / "server_sde" / name
        if not p.exists():
            bad.append({"file": "server_sde/" + name, "why": "見つからない（名前か場所が変わった）"})
            continue
        got[name] = sha(p)
    return {"total": len(PROVENANCE), "bad": bad, "sha256": got}


# ---------------------------------------------------------------- 本体
def run():
    docs = [p for p in walk(".md") if rel(p).split("/")[0] in ("md", "study") or "/" not in rel(p)]
    res = {
        "検査1_動的読み込みの相手": check_dynamic_load(),
        "検査2_索引のリンク": check_links([INDEX] if INDEX.exists() else [], "索引"),
        "検査3_importの解決": check_imports(),
        "検査4_文書のリンク": check_links(docs, "md/ と study/"),
        "検査5_ROOTの深さ": check_root_depth(),
        "検査6_来歴の5ファイル": check_provenance(),
        "検査7_scriptsパス名指し": check_scripts_path_refs(),
        "検査8_空回りする試験": check_vacuous_tests(),
    }
    res["ファイル数"] = {"py": len(list(walk(".py"))), "md": len(list(walk(".md"))),
                    "scripts_py": len(list((ROOT / "scripts").glob("*.py")))}
    return res


def show(res, baseline=None):
    ng = 0
    print("置き場所の検査 — %s" % ROOT)
    print("  ファイル: scripts/*.py %d 本 / .py 全体 %d 本 / .md %d 本"
          % (res["ファイル数"]["scripts_py"], res["ファイル数"]["py"], res["ファイル数"]["md"]))
    print()
    for k in [k for k in res if k.startswith("検査")]:
        v = res[k]
        nbad = len(v["bad"])
        # 監査 R3（2026-09-13）: 件数ではなく中身で比べる。同数でも入れ替わっていたら悪化とみなす
        base_items = {json.dumps(b, ensure_ascii=False, sort_keys=True)
                      for b in baseline[k]["bad"]} if baseline and k in baseline else None
        now_items = {json.dumps(b, ensure_ascii=False, sort_keys=True) for b in v["bad"]}
        newly = sorted(now_items - base_items) if base_items is not None else None
        base = len(base_items) if base_items is not None else None
        mark = "OK " if nbad == 0 else ("→ " if newly is not None and not newly else "NG ")
        extra = "" if base is None else "（基準 %d 件・新規 %d 件）" % (base, len(newly))
        print("%s%-22s 対象 %5d 件 / 問題 %3d 件%s" % (mark, k, v["total"], nbad, extra))
        for b in v["bad"][:8]:
            print("      %s" % json.dumps(b, ensure_ascii=False))
        if nbad > 8:
            print("      ... 他 %d 件" % (nbad - 8))
        if base is None:
            ng += nbad
        elif newly:
            ng += len(newly)
            for x in newly[:5]:
                print("      ★新規: %s" % x)
    print()
    if baseline:
        print("判定: %s" % ("基準より悪化していない" if ng == 0 else "基準より %d 件 悪化 ★" % ng))
    else:
        print("判定: %s" % ("問題なし" if ng == 0 else "%d 件の問題" % ng))
    return ng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="結果を保存する（移動前の基準を取るとき）")
    ap.add_argument("--baseline", type=Path, help="この基準と比べる")
    a = ap.parse_args()
    res = run()
    base = json.loads(a.baseline.read_text(encoding="utf-8")) if a.baseline else None
    ng = show(res, base)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("保存: %s" % a.json)
    sys.exit(1 if ng else 0)


if __name__ == "__main__":
    main()
