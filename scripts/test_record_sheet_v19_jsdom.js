// 使い方: npm i jsdom@30.1.1（任意の場所・Astra CAR24-10: 22.1 などの古い版では [hidden] の試験 1 件が環境の違いで落ちる）→ node scripts/test_record_sheet_v19_jsdom.js <収録記録シート_v19.html> [旧ノート.html] [CSV の出力先]
// 収録記録シート v19（台本から選ぶ形・iPad の画面で拍手と通過の秒を取る形）の押下テスト（jsdom）。
// 旧ノート（v18）を渡すと、同じ記録から出る session.csv / events.csv が v18 と一字一句同じかも確かめる。
const fs = require("fs");
const { JSDOM } = require("jsdom");
const html = fs.readFileSync(process.argv[2], "utf-8");
const oldHtml = process.argv[3] && fs.existsSync(process.argv[3]) ? fs.readFileSync(process.argv[3], "utf-8") : null;
const outDir = process.argv[4] || require("fs").mkdtempSync(require("path").join(require("os").tmpdir(), "sheet_test_"));   // AR12-15: 既定はリポジトリの中に書かない
function boot(src, storage) {
  return new JSDOM("<!doctype html><html><head><meta charset='utf-8'></head><body>" + src + "</body></html>",
    { runScripts: "dangerously", url: "https://example.org/", pretendToBeVisual: true,
      beforeParse(win) { if (storage) win.localStorage.setItem("helmet_note_v1", storage); } });
}
// 変更票 #69（10/17）: 10/15 に録った本（旧の番号）を持つ記録。番号と原本名は本物の照合（raw/ledger の state_fixed）と同じ（名前は入れない）。
// asis = true なら、記録シートのそのままの書き出し（C-01 が 2 本・C-04 が 014）と同じ形にする
function fix1015(asis) {
  const C = asis ? [["C-01", "003", "0"], ["C-01", "006", "0"], ["C-03", "007", "0"], ["C-04", "014", "180"], ["C-02", "015", "180"], ["C-05", "086", "0"], ["C-07", "087", "0"], ["C-08", "089", "180"], ["C-06", "090", "180"]]
    : [["C-01", "006", "0"], ["C-03", "007", "0"], ["C-04", "012", "180"], ["C-02", "015", "180"], ["C-05", "086", "0"], ["C-07", "087", "0"], ["C-08", "089", "180"], ["C-06", "090", "180"]];
  const H = [["H-03", "049"], ["H-01", "050"], ["H-09", "051"], ["H-05", "053"], ["H-11", "054"], ["H-07", "055"], ["H-17", "056"], ["H-13", "057"], ["H-19", "059"], ["H-15", "060"], ["H-25", "063"], ["H-27", "064"],
    ["H-02", "066"], ["H-04", "068"], ["H-06", "069"], ["H-10", "070"], ["H-08", "071"], ["H-12", "072"], ["H-14", "073"], ["H-18", "074"], ["H-16", "075"], ["H-20", "076"], ["H-24", "077"], ["H-22", "079"], ["H-28", "080"], ["H-26", "081"], ["H-21", "083"], ["H-23", "084"]];
  // Astra MAP25-10（10/25）: 解析まで通す古い記録として、本物の 10/15 と同じ条件（線・台本の向き・ベルの位置）を持たせる（旧は全部 1.0 m・後→前・ベルなし）。
  // 向きは台本の向き（マイクから見た向き）で持つ＝三脚 180° の本は記録シートの書き出し（fieldDirection）で裏返る。対応表で使う 28 本は新しい台本 C2-13〜16・H2 の条件と合う
  const COND = { "C-01": ["1.0", "後→前", "-10"], "C-03": ["1.0", "前→後", "10"], "C-04": ["1.0", "前→後", "-10"], "C-02": ["1.0", "後→前", "10"],
    "C-05": ["3.0", "後→前", "-5"], "C-07": ["3.0", "前→後", "5"], "C-08": ["3.0", "前→後", "-5"], "C-06": ["3.0", "後→前", "5"],
    "H-03": ["1.0", "後→前"], "H-01": ["1.0", "前→後"], "H-09": ["1.0", "後→前"], "H-05": ["1.0", "前→後"], "H-11": ["1.0", "後→前"], "H-07": ["1.0", "前→後"],
    "H-17": ["2.0", "後→前"], "H-13": ["1.5", "前→後"], "H-19": ["2.0", "後→前"], "H-15": ["1.5", "前→後"], "H-25": ["5.0", "後→前"], "H-27": ["5.0", "前→後"],
    "H-02": ["1.0", "前→後"], "H-04": ["1.0", "後→前"], "H-06": ["1.0", "前→後"], "H-10": ["1.0", "後→前"], "H-08": ["1.0", "前→後"], "H-12": ["1.0", "後→前"],
    "H-14": ["1.5", "前→後"], "H-18": ["2.0", "後→前"], "H-16": ["1.5", "前→後"], "H-20": ["2.0", "後→前"], "H-24": ["3.0", "前→後"], "H-22": ["3.0", "後→前"],
    "H-28": ["5.0", "前→後"], "H-26": ["5.0", "後→前"], "H-21": ["3.0", "後→前"], "H-23": ["3.0", "前→後"] };
  const mk = (id, kubun, rows, cls) => ({ dev: "dIPAD", id, date: "2026-10-15", time: "10:00", place: "学内（森戸記念体育館西側）", kubun, script: "", yoto: "最終評価", prefix: "261015", nextFile: 91, videoNext: 5410,
    weather: "晴", wind: "弱", helper: "—", note: "", amedas: "", mount: "三脚", camera: "", marks: "", height: "160", level: "y", laeq: "52", win: "0-60", gain: "70", front: "",
    calibFile: "261015_001.WAV", checkFile: "261015_002.WAV", checkMode: "4", ritual: { calib: true, check: true }, bikeWb: "104", bellZ: "1.0",
    takes: rows.map(([sid, f, h], i) => { const hh = h || (Number(sid.slice(2)) % 2 ? "0" : "180"); return { take_id: i + 1, uid: id + "#" + (i + 1), file: "261015_" + f + ".WAV", running: false, scriptId: sid, script: sid,
      tripodHeading: hh, takeCheck: hh === "180" ? "261015_085.WAV" : "", checkOk: true, video: "IMG_" + (5336 + i) + ".MOV", clapA: "", clapV: "", clap2A: "", clap2V: "", clapF: "", videoOk: "", mount: "三脚", height: "160", sourceX: "", sourceY: "", sourceZ: cls === "bike_bell" ? "1.0" : "",
      events: [{ event_id: 1, cls, quad: "B", lap: "", group: "", power: "", vtype: "", confirm: "", speed: "", dist: cls === "bicycle" ? (COND[sid] || ["1.0"])[0] : "", stand: "静止", pair: "", note: "", scene: "", ppos: "",
        direction: (COND[sid] || [, "後→前"])[1], runSpeed: "", bellMark: (COND[sid] || [])[2] || "", line: (COND[sid] || ["1.0"])[0], equipmentNote: "" }] }; }),
    take: null, excluded: [], noVideo: false, stand: "静止", lastPair: "", equip: "", power: "", srcZ: "", heading: "0", checkHeading: "0", validCheck: "261015_002.WAV", rotCheck: "", pick: "",
    checks: { "261015_002.WAV": { h: "0", mode: "4" }, "261015_085.WAV": { h: "180", mode: "4" } }, ver: 54 });
  return { sessions: [mk("20261015_C1", "C", C, "bike_bell"), mk("20261015_H1", "H", H, "bicycle")], cur: null, prep: {}, tab: "start", check4Done: false };
}
let dom = boot(html, JSON.stringify(fix1015())), w = dom.window, d = w.document;   // #69: 10/15 の C・H がある iPad で 10/22 を録る形
const errs = []; w.addEventListener("error", e => errs.push(e.message));
const $ = id => d.getElementById(id);
const click = el => { if (!el) throw new Error("no element"); el.dispatchEvent(new w.Event("click", { bubbles: true })); };
// FR11-05（Astra 2026-10-11）: 試験は画面に出ているボタンだけ押す。隠れている（hidden）・無効（disabled）なボタンは押さない（台本の本では拍手・通過のボタンが出ない・#59）
const pv = (E, ck, id) => { const el = E(id); if (el && !el.hidden && !el.disabled) ck(el); };
const press = id => pv($, click, id);
const setv = (id, v) => { $(id).value = v; $(id).dispatchEvent(new w.Event("change", { bubbles: true })); };
const chip = (sel, re) => [...d.querySelectorAll(sel + " .chip")].find(x => re.test(x.textContent));
const cat = re => [...d.querySelectorAll("#catGrid .cat")].find(x => re.test(x.textContent));
const tab = t => click(d.querySelector("nav button[data-tab=" + t + "]"));
const state = () => JSON.parse(w.localStorage.getItem("helmet_note_v1"));
const curS = () => { const s = state(); return s.sessions.find(x => x.id === s.cur); };
const step = (name, fn) => { try { fn(); console.log("ok  ", name); } catch (e) { console.log("FAIL", name, "-", e.message); process.exitCode = 1; } };
const csvRows = id => $(id).value.trim().split("\n").map(l => l.split(","));
function take() { click($("btnStart")); press("btnClap"); press("btnEvent"); click($("btnStop")); if (!$("running").hidden) click($("btnStop")); }   // FR11-05: 出ているボタンだけ。台本の本は 1 回目の停止で止まる

step("区分のカードに中身と本数が出る（#69: A 車の通過 32 本・10/15 の本は対応表で C「4 / 16」・H「24 / 60」・X 踏切は保留）", () => {
  tab("start"); const a = cat(/車の通過/), x = cat(/踏切/);
  if (!a || !/32 本/.test(a.textContent) || !/時速 10 km と 15 km/.test(a.textContent)) throw new Error(a && a.textContent);
  if (!/4 \/ 16 本 済み/.test(cat(/自転車のベル/).textContent) || !/24 \/ 60 本 済み/.test(cat(/自転車（ベルなし）/).textContent) || !/0 \/ 16 本 済み/.test(cat(/歩く比較（自転車）/).textContent)) throw new Error("done counts: " + cat(/自転車のベル/).textContent + " / " + cat(/自転車（ベルなし）/).textContent);
  if (!x || !/保留/.test(x.textContent)) throw new Error(x && x.textContent);
  if (d.querySelectorAll("#catGrid .cat").length !== 13) throw new Error("cats=" + d.querySelectorAll("#catGrid .cat").length);   // 10/13: 歩く比較を自転車と車に分けて 12（#69 r47 の開発用 3 つは r48 で外した）・10/27 走りながらのクラクション E3 を足して 13
});
step("自転車 C を本番で始める（#69: 10/22・録音 1・動画 100）→ 校正 261022_001 / 点検 261022_002・台本は C2", () => {
  click(cat(/自転車のベル/)); setv("inDate", "2026-10-22"); $("inFileNo").value = "1"; $("inVideoNo").value = "100"; $("inGain").value = "80";
  click($("btnNewSession")); const s = curS();
  if (s.id !== "20261022_C1" || s.script !== "C2" || s.kubun !== "C" || s.calibFile !== "261022_001.WAV" || s.checkFile !== "261022_002.WAV" || s.place !== "学内（森戸記念体育館西側）" || s.mount !== "三脚") throw new Error(JSON.stringify([s.id, s.script, s.calibFile, s.checkFile, s.place, s.mount]));
});
step("最初にやること（LAeq・点検・高さ・水準器）→ 次の本は C2-01（10/15 の C2-13〜16 は済み・#69 の優先: 録り直しの 4 本から）", () => {
  setv("inLaeq", "52.3"); click($("btnCheckDone")); setv("inHeight", "160"); click(chip("#chipLevel", /真ん中/));
  if (!/最初にやること: ④ ベルの前後の位置/.test($("nextTxt").textContent)) throw new Error("bell offset reminder: " + $("nextTxt").textContent);   // 文書の点検 1-4（10/18）: ベルの前後の位置は必ず測る
  setv("inBellOff", "25");
  if (!/C2-01/.test($("nextTxt").textContent)) throw new Error($("nextTxt").textContent);
  tab("rec"); if ($("itId").textContent !== "C2-01") throw new Error($("itId").textContent);
  if (!/前輪が現場の x = -10（北）の目印を越えてから 1 回/.test($("itFacts").textContent) || !/出発は現場の x = -14/.test($("itFacts").textContent) || !/助走/.test($("itFacts").textContent)) throw new Error($("itFacts").textContent);
});
step("C2-01 を録る（開始 → 停止・拍手と通過のボタンは出ない）→ 台本どおりの値が入り、秒は空（帰って動画と波形で読む・#59・FR11-05）", () => {
  take(); const s = curS(); const t = s.takes[0], e = t.events[0];
  if (t.scriptId !== "C2-01" || t.file !== "261022_003.WAV" || t.video !== "IMG_0100.MOV" || t.clapA !== "" || t.tripodHeading !== "0") throw new Error(JSON.stringify(t));
  if (e.cls !== "bike_bell" || e.bellMark !== "-10" || e.line !== "1.0" || e.quad !== "B" || e.direction !== "後→前" || e.lap !== "") throw new Error(JSON.stringify(e));
  // 10/9 Version 54: 走る向きが交互になる順（#69: C2-01 → C2-03 →（180°）C2-04 → C2-02）
  if ($("itId").textContent !== "C2-03") throw new Error("next=" + $("itId").textContent);
});
step("左の 2 本（C2-01・03）のあと、C2-04 の前に向き 180° の案内が出て、録音開始は止まる（#69: まとまりごとに左右をそろえる）", () => {
  take();
  if ($("itId").textContent !== "C2-04") throw new Error($("itId").textContent);
  if ($("rotBox").hidden) throw new Error("rotBox hidden"); if (!$("runCard").hidden) throw new Error("runCard visible");
  if (!/261022_005\.WAV/.test($("rotSteps").textContent)) throw new Error($("rotSteps").textContent);
});
step("点検を録った → C2-04 は向き 180°・回転後点検 261022_005、ファイル 261022_006", () => {
  click($("btnRotDone")); if (curS().rotCheck) throw new Error("rotation recorded at first press (S1)"); click($("btnRotDone")); if (!$("rotBox").hidden) throw new Error("rotBox still");
  if (!/出発は現場の x = -14/.test($("itFacts").textContent) || !/前輪が現場の x = -10（北）の目印を越えてから 1 回（マイクから見て 前 10 m）/.test($("itFacts").textContent)) throw new Error($("itFacts").textContent);   // C2-04 = 右 1.0 m・前→後・ベル 10 m
  take(); const t = curS().takes[2];
  if (t.scriptId !== "C2-04" || t.tripodHeading !== "180" || t.takeCheck !== "261022_005.WAV" || t.file !== "261022_006.WAV" || t.video !== "IMG_0102.MOV") throw new Error(JSON.stringify(t));
});
step("やり直す →『録音した→使わない』で番号と動画番号が進む（C2-02 は未録のまま）", () => {
  click($("btnStart")); click($("btnAbort"));
  for (const want of ["人の声", "工事音"]) if (!chip("#abReason", new RegExp("^" + want + "$"))) throw new Error("ハプニングの理由に " + want + " が無い（本人 10/26 23:17）");
  click(chip("#abReason", /車が通った/)); click($("abExclude"));
  const s = curS(); if (s.excluded.length !== 1 || s.excluded[0].file !== "261022_007.WAV" || s.nextFile !== 8 || s.videoNext !== 104 || !/車が通った/.test(s.excluded[0].reason)) throw new Error(JSON.stringify([s.excluded, s.nextFile, s.videoNext]));
  if ($("itId").textContent !== "C2-02") throw new Error($("itId").textContent);
});
step("監査 10/8 V50-01〜04: ③で向きを変えたら点検を録るまで録れない・使用済みの番号に直せない・録音していなかった（動画は回した）", () => {
  const bx = boot(html, null), W = bx.window, D = W.document; const E = id => D.getElementById(id);
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
  const st = () => { const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); return x.sessions.find(y => y.id === x.cur); };
  ck(D.querySelector("nav button[data-tab=start]")); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent)));
  sv("inDate", "2026-10-15"); E("inFileNo").value = "1"; E("inVideoNo").value = "200"; ck(E("btnNewSession"));
  sv("inLaeq", "50"); ck(E("btnCheckDone")); if (st().checkHeading !== "0") throw new Error("check heading " + st().checkHeading);
  // V50-01: ③で 180° にすると、点検を録るまで録音できない
  ck([...D.querySelectorAll("#chipHeading .chip")].find(x => /180/.test(x.textContent)));
  ck(D.querySelector("nav button[data-tab=rec]"));
  if (E("rotBox").hidden || !E("runCard").hidden) throw new Error("no check prompt after manual heading change");
  ck(E("btnStart")); if (st().take) throw new Error("started without a check for 180");
  ck(E("btnRotDone")); ck(E("btnRotDone")); { const z = st(); if (z.checkHeading !== "180" || z.rotCheck !== "261015_003.WAV" || z.nextFile !== 4) throw new Error(JSON.stringify([z.checkHeading, z.rotCheck, z.nextFile])); }
  if (!E("rotBox").hidden) throw new Error("prompt still shown");
  // V50-02: 録音中に使用済みの番号（校正 001）へは直せない
  ck(E("btnStart")); const before = st().take.file; sv("inRunFile", "1"); if (st().take.file !== before) throw new Error("clash accepted: " + st().take.file);
  pv(E, ck, "btnClap"); pv(E, ck, "btnEvent"); ck(E("btnStop")); if (!E("running").hidden) ck(E("btnStop"));
  { const t = st().takes[0]; if (t.takeCheck !== "261015_003.WAV" || t.checkOk !== true) throw new Error(JSON.stringify([t.takeCheck, t.checkOk])); }
  sv("inNextFile", "2"); if (st().nextFile === 2) throw new Error("next file clash accepted");
  // V50-04: 録音していなかった（動画は回した）= 動画の番号だけ進む
  const v0 = st().videoNext, f0 = st().nextFile; ck(E("btnStart")); ck(E("btnAbort")); ck(E("abNotRecV"));
  if (st().take || st().videoNext !== v0 + 1 || st().nextFile !== f0) throw new Error(JSON.stringify([st().videoNext, v0, st().nextFile, f0]));
  // 衝突は保存前の検査でも止まる（一覧で原本名を校正と同じにした場合）
  const raw = JSON.parse(W.localStorage.getItem("helmet_note_v1")); raw.sessions[0].takes[0].file = "261015_001.WAV";
  const b2 = boot(html, JSON.stringify(raw));
  b2.window.document.querySelector("nav button[data-tab=save]").dispatchEvent(new b2.window.Event("click", { bubbles: true }));
  if (!/261015_001\.WAV が 2 つの記録で重なっている/.test(b2.window.document.getElementById("issues").textContent)) throw new Error("clash not reported");
});
step("自己監査 S2: 押し間違えた見本の行を消すと、見本の印が外れ、番号を戻せる", () => {
  const bx = boot(html, null), W = bx.window, D = W.document; const E = id => D.getElementById(id);
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
  const st = () => { const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); return x.sessions.find(y => y.id === x.cur); };
  ck(D.querySelector("nav button[data-tab=start]")); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent)));
  sv("inDate", "2026-10-15"); E("inFileNo").value = "1"; ck(E("btnNewSession"));
  ck(E("btnBellRef")); ck(E("btnBellRef")); if (st().bellRef !== "261015_003.WAV" || st().nextFile !== 4) throw new Error("sample");
  ck(D.querySelector("nav button[data-tab=list]")); const btn = E("exdel_0"); ck(btn); if (!st().bellRef) throw new Error("deleted at first press");
  ck(E("exdel_0")); if (st().bellRef || st().excluded.length) throw new Error(JSON.stringify([st().bellRef, st().excluded]));
  sv("inNextFile", "3"); if (st().nextFile !== 3) throw new Error("could not set back to 003");
  ck(D.querySelector("nav button[data-tab=start]")); if (!E("rBellRef").hidden) throw new Error("bell card shown (#66)");
});
step("ハプニング →「続ける」はメモに残り、止めたときの記録の特記に入る（10/8・別の画面で）", () => {
  const bx = boot(html, null), W = bx.window, D = W.document; const E = id => D.getElementById(id);
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
  const st = () => { const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); return x.sessions.find(y => y.id === x.cur); };
  ck(D.querySelector("nav button[data-tab=start]")); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent)));
  sv("inDate", "2026-10-15"); E("inFileNo").value = "1"; ck(E("btnNewSession"));
  ck(D.querySelector("nav button[data-tab=rec]")); ck(E("btnStart")); pv(E, ck, "btnClap"); ck(E("btnAbort"));
  ck([...D.querySelectorAll("#abReason .chip")].find(x => /人・自転車/.test(x.textContent))); ck(E("abKeep"));
  if (E("running").hidden) throw new Error("stopped by keep");
  pv(E, ck, "btnEvent"); ck(E("btnStop")); if (!E("running").hidden) ck(E("btnStop"));
  const t = st().takes[0]; if (!t || !/ハプニング（続行）: 人・自転車が通った/.test(t.events[0].note || "")) throw new Error(JSON.stringify(t && t.events));
  if (!/ハプニング（続行）/.test(W.__note.eventsCsv())) throw new Error("not in csv");
});
step("#59: 台本の本は通過ボタンが出ず、1 回目の停止で秒なしの記録（帰って動画・波形で読む）として残る", () => {
  click($("btnStart")); press("btnClap");
  if (!$("btnEvent").hidden) throw new Error("event button visible in a scripted take");
  click($("btnStop"));
  if (!$("running").hidden) throw new Error("not stopped at first press");
  const t = curS().takes[3]; if (!t || t.scriptId !== "C2-02" || t.events[0].lap !== "") throw new Error(JSON.stringify(t));
});
step("#59（23:58）: 台本の本は ① 拍手のボタンも出ず、拍手の確認なしで 1 回目で止まる", () => {
  click($("btnStart"));
  if (!$("btnClap").hidden || !$("btnClap2").hidden) throw new Error("clap buttons visible in a scripted take");
  click($("btnStop")); if (!$("running").hidden) throw new Error("not stopped: " + $("btnStop").textContent);
});
step("C2 の残り（180° の 3 本 → 0° に戻して 4 本）を録ると全部済み・録音開始は押せない", () => {
  take(); take(); take(); if ($("itId").textContent !== "C2-05" || $("rotBox").hidden) throw new Error("rotation before C2-05: " + $("itId").textContent);
  click($("btnRotDone")); click($("btnRotDone")); take(); take(); take(); take();
  if ($("itId").textContent !== "全部済み") throw new Error($("itId").textContent); if (!$("btnStart").disabled) throw new Error("start enabled");
  if (curS().takes.map(t => t.scriptId).join() !== "C2-01,C2-03,C2-04,C2-02,C2-08,C2-06,C2-12,C2-10,C2-05,C2-07,C2-09,C2-11") throw new Error(curS().takes.map(t => t.scriptId).join());
});
step("events.csv: 180° の本は direction とベル目印が現場の向きに直る・除外の原本は行に無い・#69 の 3 列（台本ID・条件ID）", () => {
  tab("save"); const r = csvRows("outE"); const H = r[0]; const col = n => H.indexOf(n);
  const c04 = r.find(x => x[col("台本")] === "自転車1.0m・前→後・右・ベル+10" && x[col("原本")] === "261022_006.WAV");   // 180° の最初の本は C2-04
  if (!c04 || c04[col("direction")] !== "後→前" || c04[col("ベル目印_m")] !== "-10" || c04[col("三脚の向き")] !== "180" || c04[col("回転後点検原本")] !== "261022_005.WAV" || c04[col("象限")] !== "F" || c04[col("line_m")] !== "1.0" || c04[col("台本ID")] !== "C2-04" || c04[col("条件ID")] !== "C2-04") throw new Error(JSON.stringify(c04));
  if (/261022_007/.test($("outE").value)) throw new Error("excluded present");
  if (r.length !== 1 + 36 + 12) throw new Error("rows=" + (r.length - 1));   // 10/15 の 36 本 + 10/22 の C2 12 本
  if (H.length !== 47 || H.slice(-8).join() !== "歩く人の線_m,台本ID,条件ID,歩く人の静止x_m,合図x_m,終わり秒,マイク_x_m,マイク_y_m") throw new Error("cols=" + H.length);   // Astra H25-01（10/25）: マイクの現場の座標の 2 列   // 本人 10/18 21:28: 終わり秒   // Astra R97-01・02: 末尾に 2 列
  const byOld = id => r.find(x => x[col("台本ID")] === id && x[col("session_id")].startsWith("20261015"));
  if (byOld("C-05")[col("条件ID")] !== "C2-13" || byOld("C-06")[col("条件ID")] !== "C2-14" || byOld("H-03")[col("条件ID")] !== "H2-01" || byOld("H-28")[col("条件ID")] !== "H2-56") throw new Error("mapped ids");
  if (byOld("C-01")[col("条件ID")] !== "" || byOld("H-07")[col("条件ID")] !== "" || byOld("H-12")[col("条件ID")] !== "") throw new Error("dropped takes got a condition id");
});
step("session.csv: 場所は学内・カメラ位置あり・除外が備考に残る・23 列", () => {
  const r = csvRows("outS"); const H = r[0]; if (H.length !== 23) throw new Error("cols=" + H.length);
  const row = $("outS").value.trim().split("\n").find(l => l.startsWith("20261022_C1,"));
  if (!/学内（森戸記念体育館西側）/.test(row) || !/除外: 261022_007.WAV/.test(row) || !/,80$/.test(row) || !/52\.3/.test(row)) throw new Error(row);
});
step("次の区分へ → 自転車（ベルなし）H2 は校正と点検を引き継ぐ（新しい番号を使わない）", () => {
  tab("start"); click($("btnNextKubun")); if (cat(/キックボード/)) throw new Error("kick card still there (#50)");
  if (!cat(/自転車（ベルなし）/).classList.contains("on")) throw new Error("next kubun after C is not H (S3)"); click(cat(/自転車（ベルなし）/));
  if ($("fInherit").hidden || !$("inInherit").checked) throw new Error("inherit not offered");
  click($("btnNewSession")); const s = curS();
  if (s.id !== "20261022_H1" || s.script !== "H2" || s.calibFile !== "261022_001.WAV" || s.checkFile !== "261022_002.WAV" || s.nextFile !== 18 || !s.ritual.calib || s.inheritFrom !== "20261022_C1" || s.heading !== "0") throw new Error(JSON.stringify([s.id, s.calibFile, s.nextFile, s.ritual, s.inheritFrom, s.heading]));
});
step("C2 の最後が 0° なので、H2 は回さずに左の本から（#69 の H2 の 1 つ目のまとまり: H2-45 = 3.0 m・速い・後→前）→ 次も左の H2-47（走る向きが交互）", () => {
  tab("rec"); if (!$("rotBox").hidden) throw new Error("rotation asked although H2-45 can be recorded at 0°");
  if (!/速い 約 16 km\/h/.test($("itFacts").textContent)) throw new Error($("itFacts").textContent);
  take(); const e = curS().takes[0].events[0];
  if (curS().takes[0].scriptId !== "H2-45" || e.cls !== "bicycle" || e.dist !== "3.0" || e.runSpeed !== "速い" || e.quad !== "B" || e.direction !== "後→前") throw new Error(JSON.stringify(e));
  if ($("itId").textContent !== "H2-47" || !$("rotBox").hidden) throw new Error($("itId").textContent);
});
step("H2-13（1.5 m）は横距離が幅 1.4-1.6 で入る・歩く比較 W05〜W08 は自転車（ベルなし）2.0 m・ふつう・#69 の本数", () => {
  const it = w.__note.scriptOf("H2").find(x => x.id === "H2-13"); if (it.dist !== "1.4-1.6") throw new Error(it.dist);
  const p = w.__note.scriptOf("W2_BIKE").filter(x => /^W0[5-8]/.test(x.id)); if (p.length !== 8 || !p.every(x => x.cls === "bicycle" && x.dist === "2.0" && x.speed === "ふつう")) throw new Error(JSON.stringify(p[0]));
  if (w.__note.scriptOf("W2").some(x => x.cls === "kick") || w.__note.scriptOf("F").length !== 0) throw new Error("kick left in scripts");
  const n = k => w.__note.scriptOf(k).length;
  if (n("H2") !== 60 || n("A2") !== 32 || n("B2") !== 4 || n("C2") !== 16 || n("E2") !== 8 || n("W2_BIKE") + n("W2_CAR") !== 32 || n("G") !== 20) throw new Error("counts");
});
step("コマンド: 共通の校正は 1 回だけ変換する", () => {
  // Version 129（Astra MAP25-04）: 照らし合わせと変換は 1 行（終了コードの分かれ道ごとに --calib が出る）→ 校正の行が 1 行だけかを数える
  const c = w.__note.cmdText(), calLines = c.split("\n").filter(l => /--calib raw\/261022_001\.WAV/.test(l));
  if (calLines.length !== 1 || !/field_calib_check\.py --wav raw\/261022_001\.WAV/.test(calLines[0]) || !/--laeq 52\.3/.test(c) || !/261022_005\.WAV/.test(c)) throw new Error(c);
  if (!/--laeq-window 0-60/.test(c)) throw new Error("no --laeq-window (Fable 10/7: step19b が止まる)");
});
step("通過の秒が拍手の秒より前なら「確かめること」に出る（10/7 家デモ 55.58 < 88.06）", () => {
  // 10/9: 実物のセッションと同じく点検（② c.WAV・0°）を持たせる（点検の向きの確かめが出ないように）
  const s = { id: "X", kubun: "C", gain: "70", ritual: { calib: true, check: true }, checkFile: "c.WAV", checks: { "c.WAV": { h: "0" } }, takes: [{ take_id: 1, file: "a.WAV", clapA: 88.06, tripodHeading: "0", events: [{ lap: 55.58, cls: "bike_bell" }] }] };
  const iss = w.__note.issuesOf(s); if (!iss.some(x => x.type === "warn" && /通過 55\.58 秒が拍手 88\.06 秒より前/.test(x.text))) throw new Error(JSON.stringify(iss));
  s.takes[0].events[0].lap = 95.2; if (w.__note.issuesOf(s).length) throw new Error("false warning");
});
step("何も来ない G: 60 秒前の停止は 1 回目で止まり、2 回目で none（静か）として残る", () => {
  tab("start"); click($("btnNextKubun")); click(cat(/何も来ない/)); click($("btnNewSession"));
  tab("rec"); if (!$("btnEvent").hidden === false) {}
  click($("btnStart")); press("btnClap"); if (!$("btnEvent").hidden) throw new Error("event button visible in G");
  click($("btnStop")); if ($("running").hidden) throw new Error("stopped before 60 s");
  click($("btnStop")); const t = curS().takes[0]; if (t.events[0].cls !== "none" || t.events[0].note !== "静か" || t.scriptId !== "G-01") throw new Error(JSON.stringify(t));
});
step("歩く比較: ヘルメットなので引き継がず新しい校正。#69: 車は後ろから・あなたはカメラ側の線（既定 5.0 m・車 2.0 m）・静止 → 歩行", () => {
  tab("start"); click($("btnCloseSession")); click($("btnCloseSession")); click(cat(/歩く比較（車）/));   // 10/13: 自転車と車に分けた
  if (!$("fInherit").hidden) throw new Error("inherit offered across mount");
  click($("btnNewSession")); const s = curS(); if (s.mount !== "ヘルメット" || s.calibFile === "261015_001.WAV") throw new Error(JSON.stringify([s.mount, s.calibFile]));
  setv("inLaeq", "50.1"); click($("btnCheckDone")); setv("inHeight", "175"); click(chip("#chipLevel", /真ん中/));
  tab("rec"); if ($("itId").textContent !== "W10-静止") throw new Error($("itId").textContent);   // 0° のままなので右の組（相手があなたの右）から
  if ($("btnStart").disabled || !$("fItDir").hidden || !$("fItPlace").hidden) throw new Error("car direction not preset / place chips shown (only one place since 10/17 21:21)");
  if (!/向き：後ろから来て、歩く人を追い越す/.test($("itFacts").textContent) || !/歩く人は y = 5\.0 m の線、相手は y = 2\.0 m の線（歩く人から 3\.0 m・歩く人の右）/.test($("itFacts").textContent)) throw new Error($("itFacts").textContent);
  click($("btnWPractice")); click($("btnWPractice")); if (!curS().wPractice) throw new Error("practice not recorded");   // 敵対レビュー S5: 当日と同じく練習を先に
  take(); take(); const t = curS().takes; const e0 = t[0].events[0], e1 = t[1].events[0];
  if (e0.pair !== "W10" || e0.stand !== "静止" || e1.stand !== "歩行" || e0.quad !== "B" || e0.dist !== "3.0" || e0.line !== "2.0" || e0.walkLine !== "5.0" || t[0].tripodHeading !== "0" || t[0].mount !== "ヘルメット" || t[0].height !== "175") throw new Error(JSON.stringify([e0, e1, t[0].mount]));
  if ($("itId").textContent !== "W12-静止") throw new Error($("itId").textContent);
  if (e0.direction !== "後→前" || e1.direction !== "後→前") throw new Error("dir " + JSON.stringify([e0.direction, e1.direction]));
  // #69: 歩く比較は全部 後ろから（自転車のベルも）・いつも 静止 → 歩行
  const P = w.__note.scriptOf("W2"); if (!P.every(x => x.dir === "後→前" && x.quad === "B") || P.length !== 32) throw new Error("pair dirs");
  const W2 = w.__note.scriptOf("W2_CAR"); if (!W2.every((x, i) => x.stand === (i % 2 ? "歩行" : "静止"))) throw new Error("static first");
});
step("台本の無い長回し: 相手が来たら種類を選べる・来なければ none", () => {
  tab("start"); click($("btnCloseSession")); click($("btnCloseSession")); click(cat(/長回し/)); click($("btnNewSession"));
  setv("inLaeq", "49"); click($("btnCheckDone")); setv("inHeight", "160"); click(chip("#chipLevel", /真ん中/));
  tab("rec"); click($("btnStart")); if ($("btnClap").hidden || $("btnEvent").hidden) throw new Error("long run: clap/event buttons must be visible"); press("btnClap"); press("btnEvent");
  if ($("evCard").hidden) throw new Error("evCard hidden"); click(chip("#evBox", /キックボード/)); click(chip("#evBox", /^右$/));
  click($("btnStop")); if (!$("running").hidden) throw new Error("long run stop asked twice although only class and time are needed");
  // 監査 10/8 J1: 相手が来た長回しの本にも none の行（担当区間）があり、event_id が重ならない
  { const ev = curS().takes[0].events; const ids = ev.map(x => x.event_id); if (!ev.some(x => x.cls === "none") || new Set(ids).size !== ids.length) throw new Error("J1: " + JSON.stringify(ev.map(x => [x.event_id, x.cls]))); }
  const e = curS().takes[0].events[0]; if (e.cls !== "kick" || e.quad !== "R") throw new Error(JSON.stringify(e));
  // 10/8: 長回しは既定で動画を撮らない（番号を進めない）・通った相手に向きと距離を求めない
  if (!curS().noVideo || curS().takes[0].video !== "" || !$("videoSteps").hidden || $("noVideoBox").hidden) throw new Error("long run video: " + JSON.stringify([curS().noVideo, curS().takes[0].video]));
  // 10/8: 長回しは 1 本 25 分まで（H3-VR が約 30 分でファイルを分けるため）
  if (!/25 分たったら止めて/.test($("stepNow") ? $("stepNow").textContent : d.body.textContent)) throw new Error("no 25-min hint");
});
step("長回しで通った車に「大型」を 1 回で付けられる（メモに残る）", () => {
  tab("rec"); click($("btnStart")); press("btnClap"); press("btnEvent"); click(chip("#evBox", /^車$/));
  const big = chip("#evBox", /大型/); if (!big) throw new Error("no 大型 chip"); click(big);
  click($("btnStop")); if (!$("running").hidden) click($("btnStop"));
  const t = curS().takes[curS().takes.length - 1]; const e = t.events.find(x => x.cls === "car_drive");
  if (!e || !/大型/.test(e.note || "")) throw new Error(JSON.stringify(t.events));
});
step("FR11-05: 長回しで拍手を押さずに止めると 1 回目は止まらず（拍手の確認）、2 回目で確定する", () => {
  tab("rec"); click($("btnStart")); if ($("btnClap").hidden) throw new Error("long run: clap button hidden");
  press("btnEvent"); click(chip("#evBox", /^車$/));
  click($("btnStop")); if ($("running").hidden) throw new Error("stopped at first press without a clap");
  click($("btnStop")); if (!$("running").hidden) throw new Error("not stopped at second press");
  const t = curS().takes[curS().takes.length - 1]; if (t.clapA !== "") throw new Error("clap recorded although not pressed: " + JSON.stringify(t.clapA));
});
step("一覧で直す: 台本なしの本（ボタンで拍手を押す本）は拍手の秒を書き換えられる", () => {
  tab("list"); click(d.querySelector("#listTable button")); if ($("editCard").hidden) throw new Error("edit hidden");
  if (curS().takes[0].scriptId) throw new Error("take 1 should be a free take here");
  const inp = [...d.querySelectorAll("#editCard .field")].find(f => /拍手の秒（録音開始から）/.test(f.textContent)).querySelector("input");
  inp.value = "3.21"; inp.dispatchEvent(new w.Event("change", { bubbles: true })); if (curS().takes[0].clapA !== "3.21") throw new Error(curS().takes[0].clapA);
});
step("一覧で直す（台本の本）: 拍手の秒・録音の波形（合成を固めたあとに読む）を書ける。ボタンの拍手・動画側・高さ・回転後の点検ファイルの欄は出さず、向きと記録は畳む（10/13 本人「いらないもの整理」）", () => {
  const a = fresh(); a.begin(); a.tk(); a.edit(1);
  const labels = [...a.D.querySelectorAll("#editCard .field > label")].map(l => l.textContent);
  const inp = [...a.D.querySelectorAll("#editCard .field")].find(f => /拍手の秒・録音の波形/.test(f.textContent)).querySelector("input");
  inp.value = "3.21"; inp.dispatchEvent(new a.W.Event("change", { bubbles: true })); if (a.st().takes[0].clapF !== "3.21") throw new Error(a.st().takes[0].clapF);
  if (labels.some(l => /拍手の秒（録音開始から）|最後の拍手|動画側|回転後の点検ファイル|^マイクの高さ/.test(l))) throw new Error("unneeded fields shown: " + labels.join(" / "));
  const fx = a.E("editFixBox"); if (!fx || fx.open || !fx.querySelector(".chips") || !/種類/.test(fx.textContent) || /秒（録音開始からの秒）/.test(fx.textContent)) throw new Error("fix box");
  if (!a.E("btnExTake") || fx.contains(a.E("btnExTake"))) throw new Error("使わない button must stay outside the folded box");
  if (a.evRows()[0]["拍手_録音側秒"] !== "3.21") throw new Error("events.csv 拍手_録音側秒");
});
step("はじめる: 今のマイクの向きは畳んで、今の値だけ見せる（ふだんは自動）", () => {
  tab("start"); const hb = $("headingBox"); if (!hb || hb.open || !/0°|180°/.test($("headingNow").textContent)) throw new Error("heading box");
});
step("一覧で外す: 録り終わった C2-08 を『使わない』にすると、使わない録音に回り、次の本が C2-08 に戻る", () => {
  tab("list"); const sr = [...d.querySelectorAll("#sessTable tr")].find(tr => /20261022_C1/.test(tr.textContent)); click(sr.querySelector("button"));
  tab("list"); const row = [...d.querySelectorAll("#listTable tr")].find(tr => /C2-08/.test(tr.textContent)); click(row.querySelector("button"));
  click($("btnExTake")); if (!curS().takes.some(t => t.scriptId === "C2-08")) throw new Error("removed at first press");
  click($("btnExTake")); const s = curS();
  if (s.takes.some(t => t.scriptId === "C2-08") || !s.excluded.some(x => /C2-08/.test(x.reason))) throw new Error(JSON.stringify(s.excluded));
  if (s.takes.map(t => t.take_id).join() !== s.takes.map((t, i) => i + 1).join()) throw new Error("take_id not renumbered");
  tab("rec"); if ($("itId").textContent !== "C2-08") throw new Error($("itId").textContent);
});
step("番号がずれたとき: 次の録音番号と動画番号を「録る」画面で直せる", () => {
  setv("inNextFile", "40"); setv("inNextVideo", "300"); const s = curS();
  if (s.nextFile !== 40 || s.videoNext !== 300) throw new Error(JSON.stringify([s.nextFile, s.videoNext]));
  if ($("runFile").textContent !== "261022_040.WAV" || $("runVideo").textContent !== "IMG_0300.MOV") throw new Error($("runFile").textContent + " " + $("runVideo").textContent);
  setv("inNextVideo", ""); if (curS().videoNext !== "") throw new Error("video not cleared");
});
step("バックアップ json → 空の別ブラウザに読み込むと同じ CSV", () => {
  const bk = w.__note.backupJson(); const d2 = boot(html, null); const n = d2.window.__note.importBackup(bk);
  if (n !== 7) throw new Error("sessions=" + n);   // 10/15 の C・H ＋ C2・H2・G・歩く比較（車）・長回し
  if (d2.window.__note.eventsCsv() !== w.__note.eventsCsv() || d2.window.__note.sessionCsv() !== w.__note.sessionCsv()) throw new Error("csv differs");
});
step("再起動しても残る（保存文字列から起動・録音中でなければ同じ CSV）", () => {
  const raw = w.localStorage.getItem("helmet_note_v1"); const d2 = boot(html, raw);
  if (d2.window.__note.eventsCsv() !== w.__note.eventsCsv()) throw new Error("csv differs after reboot");
});
if (oldHtml) step("同じ記録を v18 に読ませると session.csv / events.csv が一字一句同じ", () => {
  const raw = w.localStorage.getItem("helmet_note_v1"); const o = boot(oldHtml, raw);
  const noBicycleDist = csv => { const L = csv.trim().split("\n"); const H = L[0].split(","); const ci = H.indexOf("class"), di = H.indexOf("横距離m");
    return L.map((l, i) => { if (!i) return l; const c = l.split(","); if (c[ci] === "bicycle") c[di] = ""; return c.join(","); }).join("\n"); };
  // Astra P25（10/25）: 末尾の個数で落とすと列が増えるたびにずれる → v18 の見出しの列だけを名前で取り出し、v19 で増えてよい列は名前で明示する
  const V19_ONLY = ["歩く人の線_m", "台本ID", "条件ID", "歩く人の静止x_m", "合図x_m", "終わり秒", "マイク_x_m", "マイク_y_m"];
  const oldHead = o.window.__note.eventsCsv().trim().split("\n")[0].split(",");
  const no69 = csv => { const L = csv.trim().split("\n"); const H = L[0].split(",");
    const extra = H.filter(h => !oldHead.includes(h)), bad = extra.filter(h => !V19_ONLY.includes(h)), lost = oldHead.filter(h => !H.includes(h));
    if (bad.length || lost.length) throw new Error("見出しの差が想定外: 増えた " + bad.join("・") + " / 無くなった " + lost.join("・"));
    const idx = oldHead.map(h => H.indexOf(h));
    return L.map(l => { const c = l.split(","); return idx.map(i => c[i]).join(","); }).join("\n"); };
  // （v18 に無い列: #69 の 歩く人の線_m・台本ID・条件ID／Astra R97 の 歩く人の静止x_m・合図x_m／R108 の 終わり秒／H25-01 の マイク_x_m・マイク_y_m）
  if (!/,bicycle,/.test(w.__note.eventsCsv())) throw new Error("no bicycle row to compare");
  if (noBicycleDist(o.window.__note.eventsCsv()) !== noBicycleDist(no69(w.__note.eventsCsv()))) throw new Error("events differ (bicycle の横距離と #69 の 3 列以外)");
  const noDims = csv => csv.replace(/／自転車の長さ: 軸の間 [\d.]+ cm/g, "").replace(/／ベルの位置: 前輪の軸から後ろへ -?[\d.]+ cm/g, "");   // #59 r4 で v19 の備考に足した長さ（10/15 の記録には入っている）と #69 r6 のベルの前後の位置は v18 に無い
  if (o.window.__note.sessionCsv() !== noDims(w.__note.sessionCsv())) throw new Error("session differ");
});
step("#66: ④ ベルの見本の欄は出ない（押す道と、番号・除外・備考・コマンドの扱いは前のまま）", () => {
  const bx = boot(html, null), W = bx.window, D = W.document; const E = id => D.getElementById(id);
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
  const st = () => { const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); return x.sessions.find(y => y.id === x.cur); };
  ck(D.querySelector("nav button[data-tab=start]")); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent)));
  sv("inDate", "2026-10-15"); E("inFileNo").value = "1"; ck(E("btnNewSession"));
  if (!E("rBellRef").hidden) throw new Error("bell card shown (#66)");
  ck(E("btnBellRef")); if (st().bellRef) throw new Error("sample at first press (S1)"); ck(E("btnBellRef")); const s = st();
  if (s.bellRef !== "261015_003.WAV" || s.nextFile !== 4 || !s.excluded.some(x => x.file === "261015_003.WAV" && /ベルの見本/.test(x.reason))) throw new Error(JSON.stringify([s.bellRef, s.nextFile, s.excluded]));
  if (!E("rBellRef").hidden) throw new Error("card still shown");
  if (!/ベル見本: 261015_003\.WAV/.test(W.__note.sessionCsv()) || !/# ベルの見本.*261015_003\.WAV/.test(W.__note.cmdText())) throw new Error("not in csv/cmd");
  // チチチの測定（10/8 12:36）: 同じ形で番号と動画の番号を 1 つずつ使う
  sv("inNextVideo", "500"); if (!E("rCoastRef").hidden) throw new Error("coast card shown (#66)");
  ck(E("btnCoastRef")); ck(E("btnCoastRef")); const c = st();
  if (c.coastRef !== "261015_004.WAV" || c.nextFile !== 5 || c.videoNext !== 501 || !c.excluded.some(x => x.file === "261015_004.WAV" && /チチチの測定/.test(x.reason))) throw new Error(JSON.stringify([c.coastRef, c.nextFile, c.videoNext]));
  // 監査 10/8 J4: 録音中は見本の「録った」を押せない ／ J5: 録音中に今の本のファイル名を直せる
  ck(E("btnStart")); ck(E("btnBellRef")); if (st().excluded.length !== 2 || st().nextFile !== 5) throw new Error("J4: sample taken while recording");
  sv("inRunFile", "9"); if (st().take.file !== "261015_009.WAV" || st().nextFile !== 9) throw new Error("J5: " + JSON.stringify([st().take.file, st().nextFile]));
  ck(E("btnAbort")); ck(E("abNotRec")); if (st().take || st().nextFile !== 9) throw new Error("abort after fix");
  if (!E("rCoastRef").hidden || !/チチチ測定: 261015_004\.WAV/.test(W.__note.sessionCsv()) || !/# チチチの測定.*261015_004\.WAV/.test(W.__note.cmdText())) throw new Error("coast not in csv/cmd");
  // 監査 10/8 M3: ① ② の録り直し（2 回押し）で原本名が差し替わり、失敗した録音は使わない録音に
  // 再監査 10/9 RV52-03: ② の「録れた」が未押しなので、① の録り直しは「② をもう録ったか」を聞いてから（ここは「まだ」）
  ck(E("btnCalibRedo")); if (st().calibFile !== "261015_001.WAV" || E("calibRedoAsk").hidden) throw new Error("redo at first press / not asked");
  ck(E("crNotYet")); { const z = st(); if (z.calibFile !== "261015_002.WAV" || z.checkFile !== "261015_009.WAV" || z.nextFile !== 10 || !z.excluded.some(x => x.file === "261015_001.WAV" && /校正の録り直し/.test(x.reason))) throw new Error("M3 calib: " + JSON.stringify([z.calibFile, z.checkFile, z.nextFile])); }
  ck(E("btnCheckRedo")); ck(E("btnCheckRedo")); { const z = st(); if (z.checkFile !== "261015_010.WAV" || z.nextFile !== 11 || !z.excluded.some(x => x.file === "261015_009.WAV" && /点検の録り直し/.test(x.reason))) throw new Error("M3 check"); }
  sv("inCalibFile", "261015_020.WAV"); if (st().calibFile !== "261015_020.WAV") throw new Error("M3 edit calib");
  // 監査 10/8 M2: ③ で 180° にしたあと、校正を引き継がずに次の区分を始めても向きは 180° のまま
  ck([...D.querySelectorAll("#chipHeading .chip")].find(x => /180/.test(x.textContent))); if (st().heading !== "180") throw new Error("M2 chip");
  ck(E("btnCloseSession")); ck(E("btnCloseSession")); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /何も来ない/.test(x.textContent)));
  if (E("inInherit")) E("inInherit").checked = false; ck(E("btnNewSession"));
  if (st().kubun !== "G" || st().inheritFrom || st().heading !== "180") throw new Error("M2 carry: " + JSON.stringify([st().kubun, st().inheritFrom, st().heading]));
});
tab("save"); fs.writeFileSync(outDir + "/v19_session.csv", $("outS").value); fs.writeFileSync(outDir + "/v19_events.csv", $("outE").value);
// 家のデモ（自転車 3 本）は別の画面で: CSV の上では区分 C・用途 調整用、台本は C-01・C-07・C-02 と同じ中身（#68 r3 の番号。旧 C-01・C-06・C-03）
step("家のデモ: カードを選ぶと練習になり、DEMO-1・2 → 向き 180° → DEMO-3。CSV は区分 C・調整用・台本は C と同じ書き方", () => {
  const dm = boot(html, null), W = dm.window, D = W.document; const E = id => D.getElementById(id);
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
  const tb = t => ck(D.querySelector("nav button[data-tab=" + t + "]"));
  const st = () => { const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); return x.sessions.find(y => y.id === x.cur); };
  const tk = () => { ck(E("btnStart")); pv(E, ck, "btnClap"); pv(E, ck, "btnEvent"); ck(E("btnStop")); if (!E("running").hidden) ck(E("btnStop")); };
  tb("start"); const card = [...D.querySelectorAll("#catGrid .cat")].find(x => /家のデモ/.test(x.textContent)); ck(card);
  if (!/練習/.test([...D.querySelectorAll("#chipYoto .chip.on")].map(x => x.textContent).join())) throw new Error("yoto not practice");
  sv("inDate", "2026-10-07"); E("inFileNo").value = "1"; ck(E("btnNewSession"));
  const s = st(); if (s.kubun !== "C" || s.script !== "DEMO" || s.yoto !== "調整用" || s.id !== "20261007_C1") throw new Error(JSON.stringify([s.kubun, s.script, s.yoto, s.id]));
  sv("inLaeq", "45"); ck(E("btnCheckDone")); sv("inHeight", "150"); ck([...D.querySelectorAll("#chipLevel .chip")][0]);
  tb("rec"); if (E("itId").textContent !== "DEMO-1") throw new Error(E("itId").textContent);
  tk(); if (E("itId").textContent !== "DEMO-2") throw new Error(E("itId").textContent);
  tk(); if (E("rotBox").hidden) throw new Error("no rotation before DEMO-3"); ck(E("btnRotDone")); ck(E("btnRotDone"));
  tk(); const t = st().takes; if (t.length !== 3 || t[2].scriptId !== "DEMO-3" || t[2].tripodHeading !== "180" || !t[2].takeCheck) throw new Error(JSON.stringify(t.map(x => [x.scriptId, x.tripodHeading, x.takeCheck])));
  if (E("itId").textContent !== "全部済み") throw new Error(E("itId").textContent);
  const ev = W.__note.eventsCsv().trim().split("\n"); const H = ev[0].split(",");
  if (ev.length !== 4 || ev[1].split(",")[H.indexOf("台本")] !== "自転車1.0m・後→前・左・ベル−4") throw new Error(ev.join("\n"));
  const ss = W.__note.sessionCsv().trim().split("\n")[1]; if (!/,C,調整用,/.test(ss)) throw new Error(ss);
  if (!/raw_demo\//.test(W.__note.cmdText())) throw new Error("no demo commands");
  // Astra 監査 10/7 V02: 家デモの step19c にも今の計画（G を含む）を明示する
  if (!/out\/field_demo\/ann_all\.csv --cut --plan A=32,B=4,C=16,D=0,E=12,H=60,G=20,歩行=32 --opportunity X=8/.test(W.__note.cmdText())) throw new Error("demo step19c without --plan");   // #69 の本数   // 10/10 #56: D は 0（書かないと既定の D:8 が残る・Astra AS10-01）
  // Astra 監査 10/7 L01: 動くベルの高さ（台本 1.0 m）が 高さ_m に入る
  if (!ev.slice(1).every(l => l.split(",")[H.indexOf("高さ_m")] === "1.0")) throw new Error("bell height: " + ev.join("\n"));
});
step("Astra V01: 台本のない長回しで「② 相手が来た」を押した時刻が残る", () => {
  const dm = boot(html, null), W = dm.window, D = W.document; const E = id => D.getElementById(id);
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  const tb = t => ck(D.querySelector("nav button[data-tab=" + t + "]"));
  tb("start"); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /長回し/.test(x.textContent))); ck(E("btnNewSession"));
  tb("rec"); ck(E("btnStart")); if (E("btnClap").hidden || E("btnEvent").hidden) throw new Error("long run: buttons hidden"); pv(E, ck, "btnClap"); pv(E, ck, "btnEvent");
  const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); const s = x.sessions.find(y => y.id === x.cur);
  if (!s.take || s.take.events.length !== 1 || s.take.events[0].lap === "" || s.take.events[0].lap == null) throw new Error(JSON.stringify(s.take && s.take.events));
});
step("マイクゲイン 70 固定（本人 10/7）: 最初から 70 が入り、引き継ぎは 1 回で始まる（ゲインの確かめは外した）", () => {
  const dm = boot(html, null), W = dm.window, D = W.document; const E = id => D.getElementById(id);
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
  const tb = t => ck(D.querySelector("nav button[data-tab=" + t + "]"));
  const st = () => { const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); return x; };
  tb("start"); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent) && !/家のデモ/.test(x.textContent)));
  if (E("inGain").value !== "70") throw new Error("default gain " + E("inGain").value);
  sv("inDate", "2026-10-15"); E("inFileNo").value = "1"; ck(E("btnNewSession"));
  if (st().sessions[0].gain !== "70") throw new Error("session gain " + st().sessions[0].gain);
  sv("inLaeq", "50"); ck(E("btnCheckDone")); sv("inHeight", "150"); ck([...D.querySelectorAll("#chipLevel .chip")][0]);
  ck(E("btnNextKubun")); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /車の通過/.test(x.textContent)));
  if (E("fInherit").hidden || !E("inInherit").checked) throw new Error("inherit not offered");
  ck(E("btnNewSession")); const x = st(); const s = x.sessions.find(y => y.id === x.cur);
  if (x.sessions.length !== 2 || s.inheritFrom !== x.sessions[0].id || s.gain !== "70") throw new Error(JSON.stringify(x.sessions.map(y => [y.id, y.inheritFrom, y.gain])));
});
// v19.3「測る」タブ（自転車の音の測定・Fable 10/7）: 別の画面で。本番の記録（helmet_note_v1）には入らない
step("測るタブ: 開けて、1 回分を足し、秒から km/h が出て CSV に入る。消すのは 2 回押し。本番の CSV は変わらない", () => {
  const dm = boot(html, null), W = dm.window, D = W.document; const E = id => D.getElementById(id);
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  const typ = (el, v) => { el.value = v; el.dispatchEvent(new W.Event("input", { bubbles: true })); };
  const before = W.__note.eventsCsv();
  ck(D.querySelector("nav button[data-tab=measure]")); if (!E("tab-measure").classList.contains("on")) throw new Error("measure tab not shown");
  if (D.querySelectorAll("nav button").length !== 5) throw new Error("nav buttons");
  ck(E("bmAdd")); const row = E("bmRows").querySelector(".card"); if (!row) throw new Error("no row");
  typ(row.querySelector('input[data-k="sec"]'), "2.4"); typ(row.querySelector('input[data-k="db"]'), "62.5");
  if (!/15\.0 km\/h/.test(row.textContent)) throw new Error(row.textContent);
  // 10/8: 既定は惰性のチチチの測定（最初の回は「止めて回す」・CSV の頭に 測定 の行）
  if (!/"1","1 m","止めて回す","2.4","15.0","62.5"/.test(E("bmOut").value) || !/"測定","惰性のチチチ"/.test(E("bmOut").value)) throw new Error(E("bmOut").value);
  if (E("bmCoastGuide").hidden) throw new Error("coast guide hidden");
  // 10/7 の走行音の記録に切り替えると別の記録（ここでは空）・戻すと惰性の 1 回分が残っている
  const setChip = s => D.querySelector('#bmSet .chip[data-set="' + s + '"]');
  ck(setChip("roll")); if (E("bmRows").querySelector(".card") || !E("bmCoastGuide").hidden || /惰性のチチチ/.test(E("bmOut").value)) throw new Error("roll not separate");
  ck(E("bmAdd")); if (!/"1","1 m","漕ぎ"/.test(E("bmOut").value)) throw new Error("roll default mode: " + E("bmOut").value);
  if (!W.localStorage.getItem("bike_measure_v1") || !W.localStorage.getItem("bike_measure_coast_v1")) throw new Error("keys");
  ck(E("bmRows").querySelector("[data-del]")); ck(E("bmRows").querySelector("[data-del]"));
  ck(setChip("coast")); if (E("bmRows").querySelectorAll(".card").length !== 1) throw new Error("coast row lost");
  const del = E("bmRows").querySelector("[data-del]"); ck(del); if (!E("bmRows").querySelector(".card")) throw new Error("deleted at first press");
  ck(E("bmRows").querySelector("[data-del]")); if (E("bmRows").querySelector(".card")) throw new Error("not deleted at second press");
  if (W.__note.eventsCsv() !== before || W.localStorage.getItem("helmet_note_v1") && /bike_measure/.test(W.localStorage.getItem("helmet_note_v1"))) throw new Error("main record touched");
});
step("変更票 #56（10/10）: バックブザー D はやめた — カードに「やめた」と出て始められない・本番のコマンドの --plan に D が無い", () => {
  const dm = boot(html, null), W = dm.window, D = W.document; const E = id => D.getElementById(id);
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  ck(D.querySelector("nav button[data-tab=start]")); const card = [...D.querySelectorAll("#catGrid .cat")].find(x => /バックブザー/.test(x.textContent));
  if (!card || !/やめた/.test(card.textContent)) throw new Error("D card: " + (card && card.textContent));
  ck(card); ck(E("btnNewSession")); if (W.localStorage.getItem("helmet_note_v1") && JSON.parse(W.localStorage.getItem("helmet_note_v1")).sessions.length) throw new Error("D session started");
  // Astra AS10-01: step19c の --plan は既定の辞書を上書きするだけなので、D を書かないと既定の D:8 が残る → D=0 を明示（実際に step19c に通す試験は scripts/test_record_sheet_plan_step19c.py）
  if (/D=8/.test(w.__note.cmdText()) || !/--plan A=32,B=4,C=16,D=0,E=12,H=60,G=20,歩行=32 --opportunity X=8/.test(w.__note.cmdText())) throw new Error("plan: " + w.__note.cmdText().match(/--plan [^ ]+/g));   // #69 の本数
});
// ---------- 10/9 Version 54: 走る向きが交互になる順（骨組み §0b）。中身は台本のまま・全部の本がそろっている ----------
step("#69 録る順: 自転車の日（C2 → H2）と車の日（A2 → B2）を記録シートの出す順でたどると、向きを変えるのはまとまりごとに 1 回で、走る向きは最後まで交互・どの台本も全部の本が 1 回ずつ", () => {
  const fieldDir = x => { const back = x.dir === "後→前"; return (x.heading === "180") !== back ? "→" : "←"; };
  // nextItem と同じ選び方（まとまりの中で今の向きの本を先に）。skip0 = 最初のまとまり（10/15 に録った本）は済み
  const day = (keys, skip0) => { let h = "0"; const seq = []; let rot = 0;
    for (const k of keys) { const L = w.__note.scriptOf(k); const nb = Math.max(...L.map(x => x.blk));
      if (new Set(L.map(x => x.id)).size !== L.length || L.some(x => !(x.blk >= 0))) throw new Error(k + " ids/blk");
      for (let b = skip0 ? 1 : 0; b <= nb; b++) { const g = L.filter(x => x.blk === b); const first = g.filter(x => x.heading === h), rest = g.filter(x => x.heading !== h);
        for (const x of [...first, ...rest]) { if (x.heading !== h) { rot++; h = x.heading; } seq.push(x); } } }
    return { seq, rot }; };
  const bike = day(["C2", "H2"], true), car = day(["A2", "B2"], false);
  for (const [nm, r] of [["自転車の日", bike], ["車の日", car]]) { const f = r.seq.map(fieldDir).join(""); if (!/^(→←)+→?$/.test(f)) throw new Error(nm + " not alternating: " + f); }
  if (bike.seq.length !== 12 + 36 || bike.rot !== 2 + 5) throw new Error("bike day " + bike.seq.length + " rot " + bike.rot);   // C2 2 まとまり・H2 5 まとまり
  if (car.seq.length !== 36 || car.rot !== 4 + 1) throw new Error("car day " + car.seq.length + " rot " + car.rot);
  if (bike.seq.slice(0, 4).map(x => x.id).join() !== "C2-01,C2-03,C2-04,C2-02" || bike.seq[12].id !== "H2-45") throw new Error("bike order " + bike.seq.slice(0, 13).map(x => x.id).join());
  // 速い自転車は遠い線（3.0 m）で試してから 1.0 m（2 つ目のまとまり）
  const fast10 = bike.seq.findIndex(x => x.id === "H2-09"), fast30 = bike.seq.findIndex(x => x.id === "H2-45"); if (!(fast30 < fast10 && fast10 < 12 + 16)) throw new Error("fast order");
  // H2 のまとまりは どれも 左右が同数（途中で止めても左右がそろう）
  const H = w.__note.scriptOf("H2"); for (let b = 1; b <= 5; b++) { const g = H.filter(x => x.blk === b); if (g.filter(x => x.side === "左").length !== g.filter(x => x.side === "右").length) throw new Error("H2 blk " + b); }
  const D = w.__note.scriptOf("DEMO"); if (D.map(x => x.id).join() !== "DEMO-1,DEMO-2,DEMO-3" || D[1].dir !== "前→後") throw new Error("demo");
});
// ---------- 変更票 #68（10/14）: A・B・H・C は左右が同じ条件・同じ本数。A・B・H は 奇数 = 左（0°）・偶数 = 右（180°） ----------
step("#69 条件ごとに 1 本ずつ: A2・B2・C2・H2 は 線・向き・速さ・ベルの組ごとに左右 1 本ずつ・奇数が左・どの線も同じ本数・前後も同じ。歩く比較は条件ごとに左右 2 組", () => {
  const key = x => [x.line, x.dir, x.speed || "", x.bell != null && x.bell !== "" ? Math.abs(Number(x.bell)) : ""].join("|");
  for (const k of ["A2", "B2", "H2", "C2"]) {
    const L = w.__note.scriptOf(k), cnt = {};
    for (const x of L) { const c = cnt[key(x)] = cnt[key(x)] || { 左: 0, 右: 0 }; c[x.side]++; if ((x.side === "右") !== (x.heading === "180")) throw new Error(k + " heading " + x.id); }
    for (const [kk, c] of Object.entries(cnt)) if (c.左 !== 1 || c.右 !== 1) throw new Error(k + " not one per side " + kk + " " + JSON.stringify(c));
    if (!L.every(x => (Number(x.id.slice(3)) % 2 === 1) === (x.side === "左"))) throw new Error(k + " odd/even");
    for (const sd of ["左", "右"]) { const b = L.filter(x => x.side === sd && x.dir === "後→前").length, f = L.filter(x => x.side === sd && x.dir === "前→後").length; if (b !== f) throw new Error(k + " F/B " + sd + " " + b + "/" + f); }
    const byLine = {}; L.forEach(x => byLine[x.line] = (byLine[x.line] || 0) + 1); if (new Set(Object.values(byLine)).size !== 1) throw new Error(k + " lines " + JSON.stringify(byLine));
  }
  const n = k => w.__note.scriptOf(k).length; if (n("A2") !== 32 || n("B2") !== 4 || n("H2") !== 60 || n("C2") !== 16 || n("E2") !== 8) throw new Error("counts");
  const A = w.__note.scriptOf("A2"); if (A.filter(x => x.kmh === "10").length !== 16 || A.find(x => x.id === "A2-17").kmh !== "10" || A.find(x => x.id === "A2-16").line !== "5.0") throw new Error("A2 speeds");
  if (!/右/.test(A.find(x => x.id === "A2-02").title) || !/時速 15 km/.test(A.find(x => x.id === "A2-02").title)) throw new Error("A2 title");
  const H = w.__note.scriptOf("H2"); for (const sp of ["ゆっくり", "ふつう", "速い"]) if (H.filter(x => x.speed === sp).length !== 20) throw new Error("H2 speed " + sp);
  const C = w.__note.scriptOf("C2"); if (C.filter(x => Math.abs(Number(x.bell)) === 10).length !== 8 || C.find(x => x.id === "C2-13").bell !== "-5" || C.find(x => x.id === "C2-13").line !== "3.0") throw new Error("C2 bells");
  if (w.__note.scriptOf("E2").map(x => x.side).join() !== "前,前,後,後,左,左,右,右" || w.__note.scriptOf("E2").map(x => x.hornR).join() !== "3,7,7,3,3,7,7,3") throw new Error("E2 sides");   // 本人 10/18 19:00: 3 m と 7 m・同じ側の 2 本を続ける   // 敵対レビュー L6: 左と右を続けて（車を動かさない）
  const W = w.__note.scriptOf("W2"); const pairs = {}; W.forEach(x => (pairs[x.pair] = pairs[x.pair] || []).push(x.stand));
  if (Object.keys(pairs).length !== 16 || !Object.values(pairs).every(p => p.join() === "静止,歩行")) throw new Error("W2 pairs");
  const g = {}; W.filter(x => x.stand === "静止").forEach(x => { const c = g[x.grp] = g[x.grp] || { 左: 0, 右: 0 }; c[x.side]++; });
  if (Object.keys(g).length !== 4 || !Object.values(g).every(c => c.左 === 2 && c.右 === 2)) throw new Error("W2 sides " + JSON.stringify(g));
  if (!W.every(x => (Number(x.pair.slice(1)) % 2 === 1) === (x.side === "左"))) throw new Error("W2 odd/even");
});
// ---------- 10/17 本人: 「ほかの本を選ぶ」で本を押しても閉じない ----------
step("10/17 本人: 「ほかの本を選ぶ」を開いて本を押しても、開いたまま（選んだ本が次の本になる）", () => {
  const a = fresh(); a.begin(); a.tb("rec"); a.E("pickBox").open = true; a.ck(a.chipIn("#itPicker", /C2-05$/));
  if (!a.E("pickBox").open) throw new Error("pickBox closed after a pick");
  if (a.E("itId").textContent !== "C2-05") throw new Error("pick not applied: " + a.E("itId").textContent);
});
// ---------- 10/15 の実録の照合（本人と Opus）: 開き直したら番号を追いつかせる・もう使った動画番号は止める ----------
step("10/15 の照合: 前のセッション（C）を開き直すと、録音と動画の番号がこの日のいちばん新しい続きになる。もう使った動画番号では録り始めの 1 回目で止まる", () => {
  const a = fresh(); a.begin(); a.tk(); a.tk();   // C: 003・004 / 動画 100・101
  a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck(a.E("btnNewSession"));   // H（同じ日）
  if (a.st().kubun !== "H") throw new Error("not H");
  a.tk(); a.tk();   // H: 005・006 / 動画 102・103
  if (a.st().nextFile !== 7 || Number(a.st().videoNext) !== 104) throw new Error("H counters " + JSON.stringify([a.st().nextFile, a.st().videoNext]));
  a.tb("list"); const sr = [...a.D.querySelectorAll("#sessTable tr")].find(tr => /20261015_C1/.test(tr.textContent)); a.ck(sr.querySelector("button"));
  if (a.st().id !== "20261015_C1" || a.st().nextFile !== 7 || Number(a.st().videoNext) !== 104) throw new Error("C not caught up: " + JSON.stringify([a.st().id, a.st().nextFile, a.st().videoNext]));
  a.sv("inNextVideo", "101"); a.tb("rec"); a.ck(a.chipIn("#itPicker", /C2-01$/)); a.ck(a.E("btnStart"));   // #69: 0° の本を選んで（C2 の 0° の 2 本は録ったので、次は 180°）
  if (a.st().take) throw new Error("started with a used video number");
  if (!/IMG_0101\.MOV はもう/.test(a.E("btnStart").textContent)) throw new Error("no warning: " + a.E("btnStart").textContent);
  a.ck(a.E("btnStart")); if (!a.st().take || a.st().take.video !== "IMG_0101.MOV") throw new Error("second press did not start");
});
// ---------- 再監査 10/9（Astra・Version 52 の RV52-01〜03）: 再現手順が止まる／正しい番号になる ----------
function fresh() {
  const dm = boot(html, null), W = dm.window, D = W.document; const E = id => D.getElementById(id);
  W.addEventListener("error", e => errs.push(e.message));
  const ck = el => { if (!el) throw new Error("no element"); el.dispatchEvent(new W.Event("click", { bubbles: true })); };
  const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
  const tb = t => ck(D.querySelector("nav button[data-tab=" + t + "]"));
  const st = () => { const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); return x.sessions.find(y => y.id === x.cur); };
  const chipIn = (sel, re) => [...D.querySelectorAll(sel + " .chip")].find(x => re.test(x.textContent));
  const tk = () => { tb("rec"); ck(E("btnStart")); pv(E, ck, "btnClap"); pv(E, ck, "btnEvent"); ck(E("btnStop")); if (st().take) ck(E("btnStop")); };
  const twice = id => { ck(E(id)); ck(E(id)); };
  const begin = check => { tb("start"); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent) && !/家のデモ/.test(x.textContent)));
    sv("inDate", "2026-10-15"); E("inFileNo").value = "1"; E("inVideoNo").value = "100"; ck(E("btnNewSession"));
    sv("inLaeq", "52.3"); if (check !== false) ck(E("btnCheckDone")); sv("inHeight", "160"); ck(chipIn("#chipLevel", /真ん中/)); };
  const exportOk = () => { tb("save"); return !E("btnFileE").disabled; };
  const edit = n => { tb("list"); const c = E("editCard"); const b = c.querySelector("b");
    if (c.hidden || !b || !b.textContent.startsWith(n + ". ")) ck(D.querySelectorAll("#listTable tr")[n].querySelector("button")); };
  const field = (label, v) => { const f = [...D.querySelectorAll("#editCard .field")].find(e => e.querySelector("label") && e.querySelector("label").textContent === label);
    if (!f) throw new Error("no field " + label); const el = f.querySelector("input"); el.value = v; el.dispatchEvent(new W.Event("change", { bubbles: true })); };
  const evRows = () => { const ev = W.__note.eventsCsv().trim().split("\n"); const H = ev[0].split(","); return ev.slice(1).map(l => { const c = l.split(","); const o = {}; H.forEach((h, i) => o[h] = c[i]); return o; }); };
  return { W, D, E, ck, sv, tb, st, chipIn, tk, twice, begin, exportOk, edit, field, evRows };
}
step("再監査 RV52-01: 同じ台本を選び直した別の本は、録音中も一覧でも、前の本と同じ原本名にできない（保存が止まる）", () => {
  const a = fresh(); a.begin(); a.tk();   // C2-13 = 003（10/15 の本が無い記録なので、最初のまとまりから）
  a.tb("rec"); a.ck(a.chipIn("#itPicker", /C2-13$/)); a.ck(a.E("btnStart"));
  if (a.st().take) throw new Error("retake started without confirmation (10/15)");
  a.ck(a.E("btnStart"));   // 10/15 の照合: もう一度押すと前の C-01 を使わない録音にして録り直す
  if (!a.st().excluded.some(x => x.file === "261015_003.WAV" && /録り直し（C2-13）/.test(x.reason))) throw new Error("previous take not excluded: " + JSON.stringify(a.st().excluded));
  a.sv("inRunFile", "3"); if (a.st().take.file !== "261015_004.WAV") throw new Error("live clash accepted: " + a.st().take.file);
  pv(a.E, a.ck, "btnClap"); pv(a.E, a.ck, "btnEvent"); a.ck(a.E("btnStop")); if (a.st().take) a.ck(a.E("btnStop"));
  const ts = a.st().takes; if (ts.length !== 1 || ts[0].scriptId !== "C2-13" || ts[0].file !== "261015_004.WAV" || !ts[0].uid) throw new Error(JSON.stringify(ts.map(t => [t.scriptId, t.file, t.uid])));
  if (!a.exportOk()) throw new Error("normal retake blocked: " + a.E("issues").textContent);
  a.edit(1); a.field("録音ファイル", "261015_003.WAV");
  if (a.exportOk()) throw new Error("list clash not blocked");
  if (!/261015_003\.WAV が 2 つの記録で重なっている.*使わない録音/.test(a.E("issues").textContent)) throw new Error(a.E("issues").textContent);
  a.edit(1); a.field("録音ファイル", "261015_004.WAV"); if (!a.exportOk()) throw new Error("still blocked after fix");   // 10/15: 前の本は使わない録音に回ったので、本は 1 本
});
step("再監査 RV52-01: 長回しの同じ本の中の複数の記録は重なりにしない", () => {
  const a = fresh(); a.begin(); a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /長回し/.test(x.textContent))); a.ck(a.E("btnNewSession"));
  a.tb("rec"); a.ck(a.E("btnStart")); if (a.E("btnClap").hidden) throw new Error("long run: clap hidden"); pv(a.E, a.ck, "btnClap"); pv(a.E, a.ck, "btnEvent"); pv(a.E, a.ck, "btnEvent"); a.ck(a.E("btnStop")); if (a.st().take) a.ck(a.E("btnStop"));
  if (a.st().takes[0].events.length !== 3 || !a.exportOk()) throw new Error(JSON.stringify([a.st().takes[0].events.length, a.E("issues").textContent]));
});
step("再監査 RV52-02 再現A: 本を録ったあとは ② を差し替えず、「点検を録り直す」で新しい番号に。前の本の点検は変わらない", () => {
  const a = fresh(); a.begin(); for (let i = 0; i < 2; i++) a.tk();   // 003・004（② 002・0°）（#69: C2 の最初のまとまりは 0° が 2 本）
  a.tb("rec"); a.twice("btnRotDone");                                   // 005・180°
  a.tb("start"); a.twice("btnCheckRedo");
  let s = a.st(); if (s.checkFile !== "261015_002.WAV" || s.nextFile !== 6 || s.excluded.length) throw new Error("initial check replaced: " + JSON.stringify([s.checkFile, s.nextFile, s.excluded]));
  a.tb("rec"); if (a.E("checkAgainBox").hidden) throw new Error("no redo box");
  a.ck(a.E("btnCheckAgain")); if (a.st().validCheck !== "261015_005.WAV") throw new Error("redo at first press");
  a.ck(a.E("btnCheckAgain")); s = a.st();
  if (!s.excluded.some(x => x.file === "261015_005.WAV" && /点検の録り直し/.test(x.reason)) || s.rotCheck) throw new Error("unused 005 not excluded: " + JSON.stringify([s.excluded, s.rotCheck]));
  if (a.E("rotBox").hidden || !/点検を録り直す/.test(a.E("rotTitle").textContent) || !a.E("runCard").hidden) throw new Error("rotBox " + a.E("rotTitle").textContent);
  a.ck(a.E("btnStart")); if (a.st().take) throw new Error("started without a check");
  a.twice("btnRotDone"); a.tk(); s = a.st();
  const t = s.takes; if (t[2].takeCheck !== "261015_006.WAV" || t[2].file !== "261015_007.WAV" || t.slice(0, 2).some(x => x.takeCheck)) throw new Error(JSON.stringify(t.map(x => [x.file, x.takeCheck, x.tripodHeading])));
  if (s.checks["261015_002.WAV"].h !== "0" || s.checks["261015_006.WAV"].h !== "180" || s.checks["261015_005.WAV"]) throw new Error(JSON.stringify(s.checks));
  if (!a.exportOk()) throw new Error("export blocked: " + a.E("issues").textContent);
  const ev = a.evRows(); if (ev.slice(0, 2).some(r => r["回転後点検原本"] !== "" || r["三脚の向き"] !== "0") || ev[2]["回転後点検原本"] !== "261015_006.WAV" || ev[2]["三脚の向き"] !== "180") throw new Error(JSON.stringify(ev));
  if (!/,261015_002\.WAV,/.test(a.W.__note.sessionCsv()) || !/除外: .*261015_005\.WAV/.test(a.W.__note.sessionCsv())) throw new Error(a.W.__note.sessionCsv());
});
step("再監査 RV52-02: 点検を使った本があれば、その点検は「点検を録り直す」でも使わない録音に回さない", () => {
  const a = fresh(); a.begin(); a.tk(); a.tb("rec"); a.twice("btnCheckAgain"); const s = a.st();
  if (s.excluded.length || s.checkFile !== "261015_002.WAV" || s.checkHeading !== "" || a.E("rotBox").hidden) throw new Error(JSON.stringify([s.excluded, s.checkFile, s.checkHeading]));
  a.twice("btnRotDone"); a.tk(); const t = a.st().takes; if (t[0].takeCheck !== "" || t[1].takeCheck !== "261015_004.WAV" || !a.exportOk()) throw new Error(JSON.stringify(t.map(x => [x.file, x.takeCheck])));
});
step("再監査 RV52-02: 本を録る前に「点検を録り直す」を押すと ② の録り直しになる（次の番号・前の ② は使わない録音）", () => {
  const a = fresh(); a.begin(); a.tb("rec"); a.twice("btnCheckAgain"); const s = a.st();
  if (s.checkFile !== "261015_003.WAV" || s.ritual.check || s.nextFile !== 4 || !s.excluded.some(x => x.file === "261015_002.WAV" && /点検の録り直し/.test(x.reason))) throw new Error(JSON.stringify([s.checkFile, s.ritual, s.nextFile]));
  if (!a.E("tab-start").classList.contains("on")) throw new Error("not moved to start");
});
step("再監査 RV52-02 再現B: 済んだ ② の「録れた」をもう一度押しても、別の向きの点検にならない", () => {
  const a = fresh(); a.begin(); a.tb("start"); a.ck(a.chipIn("#chipHeading", /^180°/)); a.ck(a.E("btnCheckDone"));
  const s = a.st(); if (s.checks["261015_002.WAV"].h !== "0" || s.checkHeading !== "0" || s.nextFile !== 3) throw new Error(JSON.stringify([s.checks, s.checkHeading, s.nextFile]));
  a.tb("rec"); if (a.E("rotBox").hidden) throw new Error("no rotBox"); a.ck(a.E("btnStart")); if (a.st().take) throw new Error("started");
});
step("再監査 RV52-02 再現C: 一覧で本の向きを 180° に直すと、180° の点検を選ぶまで保存が止まる", () => {
  const a = fresh(); a.begin(); a.tk();   // 003・0°・点検 002
  a.edit(1); a.ck(a.chipIn("#editCard", /^180°$/));
  if (a.st().takes[0].checkOk !== false || a.exportOk()) throw new Error("heading edit not blocked");
  if (!/180° なのに、点検 261015_002\.WAV は 0° で録ったもの/.test(a.E("issues").textContent)) throw new Error(a.E("issues").textContent);
  a.tb("rec"); a.ck(a.chipIn("#itPicker", /C2-04$/)); if (a.E("rotBox").hidden) throw new Error("no rotation"); a.twice("btnRotDone");   // 004・180°（右は偶数）
  a.edit(1); a.ck(a.chipIn("#editCheckChips", /261015_004（180°）/));
  const t = a.st().takes[0]; if (t.takeCheck !== "261015_004.WAV" || t.checkOk !== true || !a.exportOk()) throw new Error(JSON.stringify([t.takeCheck, t.checkOk, a.E("issues").textContent]));
  const ev = a.evRows(); if (ev[0]["三脚の向き"] !== "180" || ev[0]["回転後点検原本"] !== "261015_004.WAV") throw new Error(JSON.stringify(ev));
});
step("再監査 RV52-03: ② の「録れた」を押し忘れて ① を録り直すときは、② を録ったかを聞く（録った ② の番号を ① に回さない）", () => {
  const a = fresh(); a.begin(false);   // 校正 001・点検 002（「録れた」は押していない）
  a.tb("start"); a.ck(a.E("btnCalibRedo")); let s = a.st();
  if (a.E("calibRedoAsk").hidden || s.calibFile !== "261015_001.WAV" || s.nextFile !== 3) throw new Error("asked? " + JSON.stringify([s.calibFile, s.nextFile]));
  if (!/もう録った → ① は 261015_003\.WAV/.test(a.E("crHint").textContent)) throw new Error(a.E("crHint").textContent);
  a.ck(a.E("crDone")); s = a.st();
  if (s.calibFile !== "261015_003.WAV" || s.checkFile !== "261015_002.WAV" || s.nextFile !== 4 || !s.excluded.some(x => x.file === "261015_001.WAV") || !a.E("calibRedoAsk").hidden) throw new Error(JSON.stringify([s.calibFile, s.checkFile, s.nextFile]));
  a.sv("inLaeq", "52.0"); a.ck(a.E("btnCheckDone")); a.tk();
  s = a.st(); if (s.takes[0].file !== "261015_004.WAV" || !a.exportOk()) throw new Error(JSON.stringify([s.takes[0].file, a.E("issues").textContent]));
});
step("再監査 10/9: 10/9 より前の記録（本の固有の印・点検の向きが無い）を読んでも推して埋め、食い違いは確かめるだけ（保存は止めない）", () => {
  const a = fresh(); a.begin(); for (let i = 0; i < 2; i++) a.tk(); a.tb("rec"); a.twice("btnRotDone"); a.tk();   // #69: 0° の 2 本 → 005・180° → 006
  const raw = JSON.parse(a.W.localStorage.getItem("helmet_note_v1"));
  raw.sessions.forEach(s => { delete s.checks; delete s.ver; s.takes.forEach(t => delete t.uid); });
  raw.sessions[0].takes[1].tripodHeading = "180";   // 旧版で一覧から向きだけ直した本
  const dm = boot(html, JSON.stringify(raw)), W = dm.window, D = W.document;
  const s = JSON.parse(W.localStorage.getItem("helmet_note_v1")).sessions[0];
  if (s.checks["261015_002.WAV"].h !== "0" || s.checks["261015_005.WAV"].h !== "180" || new Set(s.takes.map(t => t.uid)).size !== 3) throw new Error(JSON.stringify([s.checks, s.takes.map(t => t.uid)]));
  D.querySelector("nav button[data-tab=save]").dispatchEvent(new W.Event("click", { bubbles: true }));
  if (D.getElementById("btnFileE").disabled || !/2 本目.*点検 261015_002\.WAV は 0° で録ったもの/.test(D.getElementById("issues").textContent)) throw new Error(D.getElementById("issues").textContent);
});
// ---------- 変更票 #69（10/17）: 10/15 の本の対応表・歩く比較の置き方・後回し ----------
function fixed(asis) {   // 10/15 の記録を持った画面（10/22 に始める用）
  const dm = boot(html, JSON.stringify(fix1015(asis))), W = dm.window, D = W.document; const E = id => D.getElementById(id);
  W.addEventListener("error", e => errs.push(e.message));
  const ck = el => { if (!el) throw new Error("no element"); el.dispatchEvent(new W.Event("click", { bubbles: true })); };
  const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
  const tb = t => ck(D.querySelector("nav button[data-tab=" + t + "]"));
  const st = () => { const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); return x.sessions.find(y => y.id === x.cur); };
  const chipIn = (sel, re) => [...D.querySelectorAll(sel + " .chip")].find(x => re.test(x.textContent));
  const card = re => [...D.querySelectorAll("#catGrid .cat")].find(x => re.test(x.textContent) && !/家のデモ/.test(x.textContent));
  const tk = () => { tb("rec"); ck(E("btnStart")); pv(E, ck, "btnClap"); pv(E, ck, "btnEvent"); ck(E("btnStop")); if (st().take) ck(E("btnStop")); };
  const prac = () => { tb("rec"); ck(E("btnWPractice")); ck(E("btnWPractice")); };   // 歩く比較の練習の 1 回（使わない録音）
  const start = (re, mount, noPrac) => { tb("start"); ck(card(re)); sv("inDate", "2026-10-22"); E("inFileNo").value = "1"; E("inVideoNo").value = "100"; if (E("inInherit")) E("inInherit").checked = false; ck(E("btnNewSession"));
    sv("inLaeq", "52"); ck(E("btnCheckDone")); sv("inHeight", mount === "ヘルメット" ? "175" : "160"); ck(chipIn("#chipLevel", /真ん中/)); if (mount === "ヘルメット" && !noPrac) prac(); };
  return { W, D, E, ck, sv, tb, st, chipIn, card, tk, start, prac };
}
step("Astra R142-01・02（10/27）: E2 の途中で振り切った → ハプニング「振り切った（割れた）→ やり直す」で、録った本は 70 の校正のまま・60 の新しいセッション（校正が済むまで始めない）・録り直し → E3 は 60 のまま → B2 は 70 に戻して校正と点検をやり直す", () => {
  const a = fixed(); a.start(/^Eクラクション/);
  const s0 = a.st(); if (s0.script !== "E2" || String(s0.gain) !== "70" || !s0.ritual.calib) throw new Error("E2 start " + JSON.stringify([s0.script, s0.gain, s0.ritual]));
  a.tk();                                                                                  // E2-03（70・採用）
  a.tb("rec"); if (!a.E("rotBox").hidden) { a.ck(a.E("btnRotDone")); a.ck(a.E("btnRotDone")); }
  const idClip = a.E("itId").textContent; a.ck(a.E("btnStart")); if (!a.st().take) throw new Error("clip take not started");
  const fClip = a.st().take.file;
  a.ck(a.E("btnAbort")); a.ck(a.chipIn("#abReason", /振り切った（割れた）/)); a.ck(a.E("abExclude"));
  const all = () => JSON.parse(a.W.localStorage.getItem("helmet_note_v1")).sessions;
  const old = all().find(x => x.id === s0.id), n = a.st();
  if (String(old.gain) !== "70" || old.takes.length !== 1 || !old.excluded.some(x => x.file === fClip && /振り切った（割れた）/.test(x.reason))) throw new Error("old session " + JSON.stringify([old.gain, old.takes.length, old.excluded]));
  if (n.id === s0.id || String(n.gain) !== "60" || n.ritual.calib || !n.ritual.check || !n.needCalib || !/ゲイン 70 → 60/.test(n.note || "") || n.calibFile === old.calibFile) throw new Error("new session " + JSON.stringify([n.id, n.gain, n.ritual, n.needCalib, n.note, n.calibFile]));
  a.tb("rec"); a.ck(a.E("btnStart")); if (a.st().take) throw new Error("started before calibration");
  a.tb("start"); a.sv("inGain2", "50"); if (String(a.st().gain) !== "60") { /* 録った本が無いセッションなので変えられる → 戻す */ a.sv("inGain2", "60"); }
  a.sv("inLaeq", "53"); a.tk(); const n2 = a.st();
  if (n2.takes.length !== 1 || n2.takes[0].scriptId !== idClip) throw new Error("re-record " + JSON.stringify(n2.takes.map(t => t.scriptId)) + " want " + idClip);
  a.tb("start"); a.sv("inGain2", "50"); if (String(a.st().gain) !== "60") throw new Error("gain overwritten with takes: " + a.st().gain);
  a.ck(a.E("btnNextKubun")); if (a.E("inGain").value !== "60") throw new Error("E3 gain " + a.E("inGain").value);
  a.E("inInherit").checked = true; a.ck(a.E("btnNewSession")); const e3 = a.st();
  if (e3.script !== "E3" || String(e3.gain) !== "60" || !e3.ritual.calib || e3.needCalib) throw new Error("E3 " + JSON.stringify([e3.script, e3.gain, e3.ritual, e3.needCalib]));
  a.ck(a.E("btnNextKubun")); if (a.E("inGain").value !== "70" || a.E("fInherit").hidden || !a.E("inInherit").disabled || a.E("inInherit").checked || !/ゲインを 70 に戻した/.test(a.E("inheritTxt").textContent)) throw new Error("B2 form " + [a.E("inGain").value, a.E("inheritTxt").textContent]);
  a.ck(a.E("btnNewSession")); const b2 = a.st();
  if (b2.script !== "B2" || String(b2.gain) !== "70" || b2.ritual.calib || b2.ritual.check || !b2.needCalib || !/ゲインを 70 に戻した/.test(b2.note || "")) throw new Error("B2 " + JSON.stringify([b2.script, b2.gain, b2.ritual, b2.needCalib, b2.note]));
  a.tb("rec"); a.ck(a.E("btnStart")); if (a.st().take) throw new Error("B2 started before calibration");
  // 録った本が無いうちに振り切った（朝の確かめ）→ 同じセッションのゲインを 60 にして校正をやり直す（新しいセッションは作らない）
  const f = fixed(); f.start(/^Eクラクション/); const f0 = f.st(); f.tb("rec"); f.ck(f.E("btnStart")); f.ck(f.E("btnAbort")); f.ck(f.chipIn("#abReason", /振り切った（割れた）/)); f.ck(f.E("abExclude"));
  const f1 = f.st(); if (f1.id !== f0.id || String(f1.gain) !== "60" || f1.ritual.calib || !f1.needCalib || f1.calibFile === f0.calibFile) throw new Error("morning clip " + JSON.stringify([f1.id, f1.gain, f1.ritual, f1.calibFile]));
  // ゲインを変えない日は、B2 で校正を引き継ぐ（不要な校正を増やさない）
  const g = fixed(); g.start(/^Eクラクション/); g.tk(); g.tb("start"); g.ck(g.E("btnNextKubun")); g.E("inInherit").checked = true; g.ck(g.E("btnNewSession"));
  g.ck(g.E("btnNextKubun")); if (g.E("inGain").value !== "70" || g.E("inInherit").disabled) throw new Error("normal day B2 form"); g.E("inInherit").checked = true; g.ck(g.E("btnNewSession"));
  if (g.st().script !== "B2" || !g.st().ritual.calib || g.st().needCalib) throw new Error("normal day B2 " + JSON.stringify([g.st().script, g.st().ritual]));
});
step("Astra P25（10/25）: マイクの高さは cm。1.6 や 2.5 のような m の値は止め、175 は通す", () => {
  const a = fixed(); a.start(/自転車のベル/);
  if (a.st().height !== "160") throw new Error("start height " + a.st().height);
  a.sv("inHeight", "1.6"); if (a.st().height !== "160") throw new Error("1.6 accepted: " + a.st().height);
  a.sv("inHeight", "2.5"); if (a.st().height !== "160") throw new Error("2.5 accepted: " + a.st().height);
  if (a.E("inHeight").value !== "160") throw new Error("input not reset: " + a.E("inHeight").value);
  a.sv("inHeight", "175"); if (a.st().height !== "175") throw new Error("175 rejected: " + a.st().height);
});
for (const asis of [false, true]) step("#69 対応表（Astra R17-03）: 10/15 の" + (asis ? "そのままの書き出し（C-01 が 2 本）" : "照合で直した記録") + "でも、済みは C2-13〜16 と H の 24 本だけ。C2-01〜04 と足す本は未収録・H-07・08・11・12 は数えない", () => {
  const a = fixed(asis); a.tb("start");
  if (!/4 \/ 16 本 済み/.test(a.card(/自転車のベル/).textContent) || !/24 \/ 60 本 済み/.test(a.card(/自転車（ベルなし）/).textContent)) throw new Error("cards " + a.card(/自転車のベル/).textContent);
  a.start(/自転車のベル/); a.tb("list");
  const done = [...a.D.querySelectorAll("#progChips .chip.done")].map(x => x.textContent.replace("✓ ", "")).sort().join();
  if (done !== "C2-13,C2-14,C2-15,C2-16") throw new Error("C2 done: " + done);
  a.tb("rec"); if (a.E("itId").textContent !== "C2-01") throw new Error("next " + a.E("itId").textContent);
  a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck(a.E("btnNewSession")); a.tb("list");
  const hd = [...a.D.querySelectorAll("#progChips .chip.done")].map(x => x.textContent.replace("✓ ", "")).sort();
  const want = ["01","02","03","04","05","06","07","08","15","16","19","20","25","26","29","30","41","42","43","44","53","54","55","56"].map(n => "H2-" + n);
  if (hd.join() !== want.join()) throw new Error("H2 done: " + hd.join());
  // 一覧で 10/15 のセッションを開くと、新しい番号への対応と「使わない」の印が見える。そのセッションでは録らない
  const sr = [...a.D.querySelectorAll("#sessTable tr")].find(tr => /20261015_C1/.test(tr.textContent)); a.ck(sr.querySelector("button")); a.tb("list");
  const rows = [...a.D.querySelectorAll("#listTable tr")].map(tr => tr.textContent);
  if (!rows.some(r => /C-05 → C2-13/.test(r)) || !rows.some(r => /C-01.*録り直す（#69/.test(r))) throw new Error("list marks: " + rows.join(" | "));
  if (!/#69 の前の台本/.test(a.E("sessProg").textContent)) throw new Error("old script not marked: " + a.E("sessProg").textContent);
  const n0 = a.st().takes.length; a.tb("rec"); a.ck(a.E("btnStart")); if (a.st().take || a.st().takes.length !== n0) throw new Error("recorded in an old-script session");
  const hr = [...a.D.querySelectorAll("#sessTable tr")].find(tr => /20261015_H1/.test(tr.textContent)); a.ck(hr.querySelector("button")); a.tb("list");
  if (![...a.D.querySelectorAll("#listTable tr")].some(tr => /H-07.*同じ条件の 2 本目/.test(tr.textContent))) throw new Error("H-07 not marked");
});
step("#69 歩く比較の置き方（本人 10/17 21:21・敵対レビュー S2）: 前の形は無く、置き方を選ぶ欄は出ない。車 5 m はあなた 7.0 m・車 2.0 m（左の組は 180°）で、駐車スペースが埋まっていたら録らない、と出る", () => {
  const a = fixed(); a.start(/歩く比較（車）/, "ヘルメット"); a.tb("rec");
  a.E("pickBox").open = true; a.ck(a.chipIn("#itPicker", /W13-静止$/));
  if (!a.E("fItPlace").hidden) throw new Error("place chips shown although only one place");
  if (a.W.__note.scriptOf("W2").some(x => x.places.length !== 1)) throw new Error("fallback place left");
  if (a.E("rotBox").hidden) throw new Error("W13 (左) should be 180°");
  if (!/歩く人は y = 7\.0 m の線、相手は y = 2\.0 m の線（歩く人から 5\.0 m・歩く人の左）/.test(a.E("itFacts").textContent) || !/駐車スペースが埋まっていたら、この組はその日は録らない/.test(a.E("itFacts").textContent)) throw new Error(a.E("itFacts").textContent);
  a.ck(a.E("btnRotDone")); a.ck(a.E("btnRotDone")); a.tk(); const t = a.st().takes[0], e = t.events[0];
  if (t.scriptId !== "W13-静止" || t.tripodHeading !== "180" || e.line !== "2.0" || e.walkLine !== "7.0" || e.dist !== "5.0") throw new Error(JSON.stringify([t.scriptId, t.tripodHeading, e.line, e.walkLine]));
  if (a.E("itId").textContent !== "W13-歩行") throw new Error("walk of the same pair should come next: " + a.E("itId").textContent);
  const ev = a.W.__note.eventsCsv().trim().split("\n"); const H = ev[0].split(","); const row = ev.find(l => /W13/.test(l)).split(",");
  if (row[H.indexOf("歩く人の線_m")] !== "7.0" || row[H.indexOf("line_m")] !== "2.0" || row[H.indexOf("条件ID")] !== "W13-静止") throw new Error(row.join(","));
});
step("#69 後回し（Astra R17-08）: 「この条件を後回し」で同じ条件の残りが区分の最後に回り、戻すと元の順に", () => {
  const a = fixed(); a.start(/自転車（ベルなし）/); a.tb("rec");
  if (a.E("itId").textContent !== "H2-45") throw new Error("first " + a.E("itId").textContent);
  a.ck(a.E("btnLater")); if ((a.st().later || []).length) throw new Error("deferred at first press"); a.ck(a.E("btnLater"));
  if ((a.st().later || []).sort().join() !== "H2-45,H2-46,H2-47,H2-48") throw new Error("later " + JSON.stringify(a.st().later));
  if (a.E("itId").textContent !== "H2-13" || !/後回し.*H2-45/.test(a.E("laterTxt").textContent)) throw new Error("next after defer " + a.E("itId").textContent);
  a.ck(a.E("btnLaterClear")); if (a.E("itId").textContent !== "H2-45") throw new Error("not restored " + a.E("itId").textContent);
});
step("#69 文言: A2 の時速 10 km・E2 は目印に合わせる・C2 の 5 m は目印 ±4 と ±6 の間", () => {
  const a = fixed(); a.start(/車の通過/); a.tb("rec"); a.E("pickBox").open = true;
  a.ck(a.chipIn("#itPicker", /A2-17$/)); if (!/車（時速 10 km・車体の中心で線をまたぐ/.test(a.E("itFacts").textContent)) throw new Error(a.E("itFacts").textContent);
  a.tb("start"); a.ck(a.E("btnNextKubun")); if (!/歩く比較（車）/.test(a.E("btnNewSession").textContent)) throw new Error("A2 → " + a.E("btnNewSession").textContent);   // 本人 10/24 16:33: 車の日は E2 → B2 → A2 → 歩く比較（車）
  const e = fixed(); e.start(/クラクション/); e.tb("rec"); const ef = e.E("itFacts").textContent;   // 本人 10/25: 三脚は y = +2.0・車は北 3 m
  if (!/三脚（マイク）の位置：クラクションの間だけ、x = 0 の線の上で東へ 2 m（現場の y = \+2\.0）/.test(ef) || !/前バンパーの中央を目印に合わせる（マイクから 3 m/.test(ef) || !/騒音計は校正のときの場所（原点の横）のまま/.test(ef)) throw new Error(ef);
  e.tb("start"); e.ck(e.E("btnNextKubun")); e.ck(e.E("btnNewSession")); if (e.st().script !== "E3") throw new Error("E2 → " + e.st().script);   // 本人 10/27: E2 → 走りながらのクラクション E3 → B2
  e.tb("start"); e.ck(e.E("btnNextKubun")); e.ck(e.E("btnNewSession")); if (e.st().script !== "B2") throw new Error("E3 → " + e.st().script);
  e.tb("start"); e.ck(e.E("btnNextKubun")); e.ck(e.E("btnNewSession")); if (e.st().script !== "A2") throw new Error("B2 → " + e.st().script);
  const b = fixed(); b.start(/自転車のベル/); b.tb("rec"); b.E("pickBox").open = true; b.ck(b.chipIn("#itPicker", /C2-05$/));
  if (!/現場の x = -5（北）あたり（目印 −4 と −6 の間）で 1 回/.test(b.E("itFacts").textContent)) throw new Error(b.E("itFacts").textContent);
});
// ---------- Astra 監査 V88（10/17）: 歩く比較の組は 同じセッション・同じ置き方・静止 → 歩行 ----------
for (const [re, pick, place] of [[/歩く比較（自転車）/, "W02-静止", null], [/歩く比較（車）/, "W13-静止", null]]) step("Astra V88-01: " + pick.slice(0, 3) + "（" + (place ? "前の形" : "既定の置き方") + "）の静止だけ録って閉じ、新しいセッションを始めても、残りの歩行はそこでは出ず・録れない。元のセッションを開けば同じ置き方で続けられる", () => {
  const a = fixed(); a.start(re, "ヘルメット"); a.tb("rec"); a.E("pickBox").open = true; a.ck(a.chipIn("#itPicker", new RegExp(pick + "$")));
  if (place) a.ck(a.chipIn("#chipItPlace", place));
  if (!a.E("rotBox").hidden) { a.ck(a.E("btnRotDone")); a.ck(a.E("btnRotDone")); }
  a.tk(); const sid1 = a.st().id, pid = pick.slice(0, 3);
  a.tb("start"); a.ck(a.E("btnCloseSession")); a.ck(a.E("btnCloseSession"));
  a.ck(a.card(re)); a.ck(a.E("btnNewSession")); a.sv("inLaeq", "52"); a.ck(a.E("btnCheckDone")); a.sv("inHeight", "175"); a.ck(a.chipIn("#chipLevel", /真ん中/)); a.prac();
  if (a.st().id === sid1) throw new Error("no new session");
  a.tb("rec"); if (a.E("itId").textContent.startsWith(pid)) throw new Error("the other session offers " + a.E("itId").textContent);
  a.E("pickBox").open = true; a.ck(a.chipIn("#itPicker", new RegExp(pid + "-歩行$"))); const n0 = a.st().takes.length; a.ck(a.E("btnStart")); a.ck(a.E("btnStart"));
  if (a.st().take || a.st().takes.length !== n0) throw new Error("walk recorded in another session");
  a.tb("list"); const sr = [...a.D.querySelectorAll("#sessTable tr")].find(tr => tr.textContent.includes(sid1)); a.ck(sr.querySelector("button"));
  a.tb("rec"); if (a.E("itId").textContent !== pid + "-歩行") throw new Error("original session next " + a.E("itId").textContent);
  a.tk(); const t = a.st().takes.filter(x => x.scriptId.startsWith(pid));
  if (t.map(x => x.scriptId).join() !== pid + "-静止," + pid + "-歩行" || t[0].events[0].walkLine !== t[1].events[0].walkLine || t[0].events[0].line !== t[1].events[0].line || t[0].tripodHeading !== t[1].tripodHeading) throw new Error(JSON.stringify(t.map(x => [x.scriptId, x.events[0].walkLine, x.events[0].line, x.tripodHeading])));
});
step("Astra V88-04: 静止の前に歩行を選んでも録れない。両方録ったあとで静止を録り直すと歩行も外れて 静止 → 歩行 で録り直す。一覧で静止を外すと歩行も一緒に外れる", () => {
  const a = fixed(); a.start(/歩く比較（自転車）/, "ヘルメット"); a.tb("rec"); a.E("pickBox").open = true;
  a.ck(a.chipIn("#itPicker", /W02-歩行$/)); a.ck(a.E("btnStart")); a.ck(a.E("btnStart")); if (a.st().take) throw new Error("walk started before static");
  a.ck(a.chipIn("#itPicker", /W02-静止$/)); a.tk(); a.tk();
  if (a.st().takes.map(t => t.scriptId).join() !== "W02-静止,W02-歩行") throw new Error(a.st().takes.map(t => t.scriptId).join());
  a.tb("rec"); a.E("pickBox").open = true; a.ck(a.chipIn("#itPicker", /W02-静止$/)); a.ck(a.E("btnStart")); if (a.st().take) throw new Error("retake at first press");
  if (!/組の静止と歩行の両方/.test(a.E("btnStart").textContent)) throw new Error("retake message: " + a.E("btnStart").textContent);
  a.ck(a.E("btnStart")); if (!a.st().take) throw new Error("retake not started");
  pv(a.E, a.ck, "btnClap"); pv(a.E, a.ck, "btnEvent"); a.ck(a.E("btnStop")); if (a.st().take) a.ck(a.E("btnStop"));
  const ex = a.st().excluded.map(x => x.reason).join(" / ");
  if (!/録り直し（W02-静止）/.test(ex) || !/録り直し（W02-歩行）・組の静止を録り直したため/.test(ex)) throw new Error(ex);
  if (a.st().takes.map(t => t.scriptId).join() !== "W02-静止" || a.E("itId").textContent !== "W02-歩行") throw new Error("after retake " + a.st().takes.map(t => t.scriptId).join() + " next " + a.E("itId").textContent);
  a.tk();
  a.tb("list"); const row = [...a.D.querySelectorAll("#listTable tr")].find(tr => /W02-静止/.test(tr.textContent)); a.ck(row.querySelector("button"));
  a.ck(a.E("btnExTake")); if (!/同じ組の歩行/.test(a.E("btnExTake").textContent)) throw new Error("exclude message: " + a.E("btnExTake").textContent); a.ck(a.E("btnExTake"));
  if (a.st().takes.some(t => /^W02/.test(t.scriptId))) throw new Error("pair not removed together: " + a.st().takes.map(t => t.scriptId).join());
  a.tb("rec"); if (a.E("itId").textContent !== "W02-静止") throw new Error("next after removing " + a.E("itId").textContent);
});
// ---------- 本人 10/17 19:59〜20:01: 案内は全部 現場の座標（南が +）・歩く比較は場所で合図・練習の 1 回 ----------
step("10/17 現場の座標: 歩く比較（自転車）は 南向き／北向きの組ごとに、立つ位置・出発の印・合図の目印・成功の範囲・ベルを現場の数字と方角で出す", () => {
  const a = fixed(); a.start(/歩く比較（自転車）/, "ヘルメット"); a.tb("rec");
  const facts = () => a.E("itFacts").textContent; const need = (re, what) => { if (!re.test(facts())) throw new Error(what + ": " + facts()); };
  if (a.E("itId").textContent !== "W02-静止") throw new Error(a.E("itId").textContent);
  need(/南向きの組/, "face"); need(/自分の線の現場の x = 0 に立ち、南を向く/, "static bike at 0"); need(/相手は現場の x = -14（北）の印から出る/, "bike start");   // 本人 10/17 21:50: 出発は ±14
  need(/ベル：前輪が現場の x = -6（北）の目印を越えたら 1 回/, "bell south");
  if (!/南/.test(a.E("headingNow").textContent)) throw new Error("heading label " + a.E("headingNow").textContent);
  a.tk(); if (a.E("itId").textContent !== "W02-歩行") throw new Error(a.E("itId").textContent);
  need(/自分の線を現場の x = -10（北） → \+10（南）へ/, "walk south"); need(/歩く人の足が現場の x = -6（北）を越えたら、あなたが片手を上げて合図/, "cue south"); need(/成功：追い越されたとき、歩く人が現場の x = 0〜\+6（南）にいる/, "success south");
  a.E("pickBox").open = true; a.ck(a.chipIn("#itPicker", /W01-静止$/));
  need(/北向きの組/, "face north"); need(/現場の x = 0 に立ち、北を向く/, "static north"); need(/相手は現場の x = \+14（南）の印から出る/, "bike start north"); need(/前輪が現場の x = \+6（南）の目印を越えたら/, "bell north");
  a.ck(a.chipIn("#itPicker", /W01-歩行$/)); need(/現場の x = \+10（南） → -10（北）へ/, "walk north"); need(/足が現場の x = \+6（南）を越えたら/, "cue north"); need(/現場の x = 0〜-6（北）にいる/, "success north");
  const b = fixed(); b.start(/歩く比較（車）/, "ヘルメット"); b.tb("rec");
  if (!/自分の線の現場の x = \+2（南）に立ち、南を向く/.test(b.E("itFacts").textContent) || !/現場の x = -14（北）の印から出る/.test(b.E("itFacts").textContent)) throw new Error("car south: " + b.E("itFacts").textContent);
  b.E("pickBox").open = true; b.ck(b.chipIn("#itPicker", /W09-静止$/));
  if (!/現場の x = -2（北）に立ち、北を向く/.test(b.E("itFacts").textContent) || !/現場の x = \+14（南）の印から出る/.test(b.E("itFacts").textContent)) throw new Error("car north: " + b.E("itFacts").textContent);
  b.ck(b.chipIn("#itPicker", /W10-歩行$/)); if (!/歩く人の足が現場の x = -4（北）を越えたら/.test(b.E("itFacts").textContent)) throw new Error("car cue default -4: " + b.E("itFacts").textContent);   // 車の合図の目印の既定は 4
});
step("10/17 練習の 1 回と合図の目印（10/21 #69 r28: 練習は録音なし）: 歩く比較のセッションで「練習した」を押しても番号は録音・動画とも進まず、使わない録音も増えない。合図の目印を変えると案内と session.csv の備考に入り、録った組があると 2 回押しで変わる", () => {
  const a = fixed(); a.start(/歩く比較（自転車）/, "ヘルメット", true); a.tb("rec");
  if (a.E("fWPrac").hidden) throw new Error("practice box hidden in a W2 session");
  // 敵対レビュー S5: 練習の前に「録音開始」を押すと 1 回目は止まり、練習のことを知らせる
  if (!/練習は録音なし: H3-VR も iPhone も回さず/.test(a.E("wPracTxt").textContent)) throw new Error("practice text: " + a.E("wPracTxt").textContent);
  a.ck(a.E("btnStart")); if (a.st().take || !/練習がまだ。練習は録音なしで/.test(a.E("btnStart").textContent)) throw new Error("no practice warning: " + a.E("btnStart").textContent);
  const f0 = a.st().nextFile, v0 = Number(a.st().videoNext);
  a.ck(a.E("btnWPractice")); if (a.st().wPractice) throw new Error("practice at first press"); a.ck(a.E("btnWPractice"));
  const s = a.st(); if (s.wPractice !== "録音なし" || s.nextFile !== f0 || Number(s.videoNext) !== v0 || s.excluded.some(x => /練習/.test(x.reason))) throw new Error(JSON.stringify([s.wPractice, s.nextFile, s.videoNext, s.excluded]));
  if (!/練習した（録音なし）/.test(a.E("wPracTxt").textContent)) throw new Error("practice done text: " + a.E("wPracTxt").textContent);
  a.ck(a.chipIn("#chipWTrig", /合図 −8/)); if (a.st().wTrig !== "8") throw new Error("trigger " + a.st().wTrig);
  a.E("pickBox").open = true; a.ck(a.chipIn("#itPicker", /W02-歩行$/));
  if (!/歩く人の足が現場の x = -8（北）を越えたら/.test(a.E("itFacts").textContent)) throw new Error(a.E("itFacts").textContent);
  if (!/合図の目印: 南向き −8・北向き \+8（歩く比較）/.test(a.W.__note.sessionCsv())) throw new Error("not in session.csv");
  a.ck(a.chipIn("#itPicker", /W02-静止$/)); a.tk();
  a.ck(a.chipIn("#chipWTrig", /合図 −6/)); if (a.st().wTrig !== "8") throw new Error("changed at first press after a take");
  a.ck(a.chipIn("#chipWTrig", /合図 −6/)); if (a.st().wTrig !== "6") throw new Error("not changed at second press");
  const t = fixed(); t.start(/自転車のベル/); t.tb("rec"); if (!t.E("fWPrac").hidden) throw new Error("practice box shown outside W2");
});
// ---------- 敵対レビュー（10/17・Opus）の直し: M8・L2・L3・S1・L4・L9 ----------
step("本人 10/24: 歩く比較の歩く人の文（静止・歩行）とヘルメットの案内に、マイクのロゴと赤い REC ランプの面は顔と同じ向き", () => {
  const a = fixed(); a.start(/歩く比較（車）/, "ヘルメット");   // 練習の 1 回のあと（歩行の本が出る） a.tb("rec");
  if (!/を向く（マイクのロゴと赤い REC ランプの面も顔と同じ(南|北)）/.test(a.E("itFacts").textContent)) throw new Error("static facts: " + a.E("itFacts").textContent);
  if (a.E("helmetHelpBox").hidden || !/マイクの前（ロゴと赤い REC ランプの面）は、歩く人の顔と同じ向き/.test(a.E("helmetHelpBox").textContent)) throw new Error("helmet box: " + a.E("helmetHelpBox").textContent);
  if (!a.E("rotBox").hidden) { a.ck(a.E("btnRotDone")); a.ck(a.E("btnRotDone")); } a.tk(); if (!/-歩行$/.test(a.E("itId").textContent)) throw new Error("next: " + a.E("itId").textContent);
  if (!/歩く（(南|北)へ・マイクのロゴと赤い REC ランプの面も顔と同じ(南|北)）/.test(a.E("itFacts").textContent)) throw new Error("walk facts: " + a.E("itFacts").textContent);
});
step("敵対レビュー M8: マイクゲインを前のセッションと違う値にすると、校正と点検を引き継がない（知らせて、印を外す）", () => {
  const a = fixed(); a.start(/クラクション/);   // 本人 10/24: 車の日は E2 → B2（同じ三脚）
  a.tb("start"); a.ck(a.E("btnNextKubun"));
  if (a.E("fInherit").hidden || a.E("inInherit").disabled) throw new Error("inherit not offered with the same gain");
  a.sv("inGain", "50");
  if (a.E("fInherit").hidden || !a.E("inInherit").disabled || a.E("inInherit").checked || !/ゲイン（50）が前のセッション/.test(a.E("inheritTxt").textContent)) throw new Error("gain change not handled: " + a.E("inheritTxt").textContent);
  a.E("inInherit").checked = true; a.ck(a.E("btnNewSession"));   // 印を付け直しても、ゲインが違えば引き継がない
  const s = a.st(); if (s.inheritFrom || s.ritual.calib || s.gain !== "50") throw new Error(JSON.stringify([s.inheritFrom, s.ritual, s.gain]));
});
step("敵対レビュー L2・S1・L4: 合図の目印は同じ日の作り直したセッションへ（L3 の練習の行は 10/21 #69 r28 で録音なしになったので、番号が進まないことを見る）。ヘルメットの案内（声で伝える・顔の前 → 頭の後ろ）", () => {
  const a = fixed(); a.start(/歩く比較（自転車）/, "ヘルメット", true); a.tb("rec");
  if (a.E("helmetHelpBox").hidden || !/赤いランプが点いた/.test(a.E("helmetHelpBox").textContent)) throw new Error("helmet help box");
  const v = a.st().videoNext, f = a.st().nextFile; a.ck(a.E("btnWPractice")); a.ck(a.E("btnWPractice"));
  if (a.st().wPractice !== "録音なし" || a.st().videoNext !== v || a.st().nextFile !== f || a.st().excluded.some(x => /練習/.test(x.reason))) throw new Error("practice consumed numbers " + JSON.stringify([a.st().videoNext, v, a.st().nextFile, f]));
  a.ck(a.chipIn("#chipWTrig", /合図 −8/));
  a.tb("start"); a.ck(a.E("btnCloseSession")); a.ck(a.E("btnCloseSession")); a.ck(a.card(/歩く比較（自転車）/)); a.ck(a.E("btnNewSession"));
  if (a.st().wTrig !== "8") throw new Error("trigger not inherited: " + a.st().wTrig);
  a.tb("rec"); a.E("pickBox").open = true; a.ck(a.chipIn("#itPicker", /W01-静止$/));   // 北向き（180°）の組 → 向きを変える案内
  if (a.E("rotBox").hidden || !/歩く人が反対を向いて立つ/.test(a.E("rotSteps").textContent) || !/顔の正面 1 m → 歩く人の右 → 頭の後ろ → 歩く人の左/.test(a.E("rotSteps").textContent)) throw new Error("helmet check text: " + a.E("rotSteps").textContent);
});
step("敵対レビュー L9: 次の本のカードに「まとまりの残り」が出る（H2 の 1 つ目のまとまりは 8 本）", () => {
  const a = fixed(); a.start(/自転車（ベルなし）/); a.tb("rec");
  if (!/まとまりの残り 8 本/.test(a.E("itBadge").textContent)) throw new Error(a.E("itBadge").textContent);
  a.tk(); if (!/まとまりの残り 7 本/.test(a.E("itBadge").textContent)) throw new Error(a.E("itBadge").textContent);
});
step("本人 10/17 21:21〜21:23: 歩く比較の歩行の本は、あなたが −6 で片手を上げて合図（上の案内と手順の文）。練習で外れたときのずらし方。自転車の本は ±12 の目印の前で一定の速さ", () => {
  const a = fixed(); a.start(/歩く比較（自転車）/, "ヘルメット"); a.tb("rec");
  if (!/成功の範囲（次の組なら現場の x = 0〜\+6（南））/.test(a.E("wPracTxt").textContent) || !/「追い越しが \+6（南）を越えた（遅い）」なら 1 つ手前（-8（北））へ、「追い越しが 0 より手前（早い）」なら 1 つ先（-4（北））へ/.test(a.E("wPracTxt").textContent)) throw new Error(a.E("wPracTxt").textContent);   // Astra R97-06: 次の組の向きの現場の x
  a.tk(); if (a.E("itId").textContent !== "W02-歩行") throw new Error(a.E("itId").textContent);
  if (!/5 つ数えてから現場の x = -10（北）から歩き出す。あなたは録音を始めたら合図の目印 x = -6（北）の真横・y = 0.*足が目印を越えたら片手を上げて合図.*追い越して 5 秒たつまで動かない/.test(a.E("clapStep").textContent)) throw new Error("clap step: " + a.E("clapStep").textContent);
  a.ck(a.E("btnStart")); if (!a.st().take) throw new Error("walk not started");
  if (!/足が x = -6（北）を越えたら、その目印の真横（y = 0・相手の線の外のカメラと反対側）に立ったあなたが片手を上げる/.test(a.E("nextTxt").textContent)) throw new Error("top text: " + a.E("nextTxt").textContent);
  a.ck(a.E("btnStop")); if (a.st().take) a.ck(a.E("btnStop"));
  const b = fixed(); b.start(/自転車のベル/); b.tb("rec"); if (!/x = ±12 の目印の前で一定の速さ/.test(b.E("itFacts").textContent)) throw new Error("bike start: " + b.E("itFacts").textContent);
  if (!/マイクの正面（ロゴ側）1 m で手を 1 回叩く/.test(b.E("clapStep").textContent)) throw new Error("tripod clap step: " + b.E("clapStep").textContent);
  const c = fixed(); c.start(/車の通過/); c.tb("rec"); if (!/x = ±12 の目印の前で一定の速さ/.test(c.E("itFacts").textContent)) throw new Error("car start: " + c.E("itFacts").textContent);
});
step("本人 10/17 21:39: 歩く比較の歩行の本のハプニングに「+6（南）を越えた（遅い）」「0 より手前（早い）」（北向きの組は -6（北））。やり直すと合図の目印が 1 つずれる（遅い → −8・早い → −6 に戻る）。ほかの本には出ない", () => {
  const a = fixed(); a.start(/歩く比較（自転車）/, "ヘルメット"); a.tb("rec"); a.tk();   // W02-静止
  if (a.E("itId").textContent !== "W02-歩行") throw new Error(a.E("itId").textContent);
  a.ck(a.E("btnStart")); a.ck(a.E("btnAbort"));
  if (!a.chipIn("#abReason", /\+6（南）を越えた（遅い）/) || !a.chipIn("#abReason", /0 より手前（早い）/)) throw new Error("walk reasons missing");
  a.ck(a.chipIn("#abReason", /\+6（南）を越えた（遅い）/)); a.ck(a.E("abExclude"));
  if (a.st().wTrig !== "8" || !a.st().excluded.some(x => /遅い/.test(x.reason))) throw new Error("late shift: " + JSON.stringify([a.st().wTrig, a.st().excluded.map(x => x.reason)]));
  if (a.E("itId").textContent !== "W02-歩行" || !/足が現場の x = -8（北）を越えたら/.test(a.E("itFacts").textContent)) throw new Error("after late: " + a.E("itFacts").textContent);
  a.ck(a.E("btnStart")); a.ck(a.E("btnAbort")); a.ck(a.chipIn("#abReason", /0 より手前（早い）/)); a.ck(a.E("abExclude"));
  if (a.st().wTrig !== "6") throw new Error("early shift: " + a.st().wTrig);
  const b = fixed(); b.start(/自転車のベル/); b.tb("rec"); b.ck(b.E("btnStart")); b.ck(b.E("btnAbort"));
  if (b.chipIn("#abReason", /遅い|早い/)) throw new Error("walk reasons shown outside the walking comparison");
  // 北向きの組（W01）: 文字は現場の x の -6（北）。遅いでやり直すと合図は +8（南）
  const n = fixed(); n.start(/歩く比較（自転車）/, "ヘルメット"); n.tb("rec"); n.E("pickBox").open = true; n.ck(n.chipIn("#itPicker", /W01-静止$/));
  if (!n.E("rotBox").hidden) { n.ck(n.E("btnRotDone")); n.ck(n.E("btnRotDone")); } n.tk();
  if (n.E("itId").textContent !== "W01-歩行") throw new Error("north next " + n.E("itId").textContent);
  n.ck(n.E("btnStart")); n.ck(n.E("btnAbort"));
  if (!n.chipIn("#abReason", /-6（北）を越えた（遅い）/) || n.chipIn("#abReason", /\+6（南）を越えた/)) throw new Error("north label: " + n.E("abReason").textContent);
  n.ck(n.chipIn("#abReason", /-6（北）を越えた（遅い）/)); n.ck(n.E("abExclude"));
  if (n.st().wTrig !== "8" || !/足が現場の x = \+8（南）を越えたら/.test(n.E("itFacts").textContent)) throw new Error("north late: " + n.st().wTrig + " " + n.E("itFacts").textContent);
});
step("本人 10/17 21:39: ベルの前後の位置（前輪の軸から後ろへ cm）を入れる欄。session.csv の備考に入り、同じ日の次のセッションへ。発進は東側の前輪を ∓10 の長い横テープにのせて止まる（#69 r45）", () => {
  const a = fixed(); a.start(/自転車のベル/); a.tb("start");
  if (a.E("fBellOff").hidden) throw new Error("bell offset field hidden in C2");
  a.sv("inBellOff", "25"); if (a.st().bellOff !== "25") throw new Error("bellOff " + a.st().bellOff);
  if (!/ベルの位置: 前輪の軸から後ろへ 25 cm/.test(a.W.__note.sessionCsv())) throw new Error("not in session.csv");
  a.ck(a.E("btnNextKubun")); a.ck(a.E("btnNewSession")); if (a.st().bellOff !== "25") throw new Error("not inherited: " + a.st().bellOff);
  const c = fixed(); c.start(/発進/); c.tb("rec");
  if (!/東側（カメラ側）の前輪を現場の x = -10（北）の長い横テープにのせて止まり/.test(c.E("itFacts").textContent)) throw new Error("launch: " + c.E("itFacts").textContent);
});
step("本人 10/17 22:03: 車を使う区分（A2・B2・E2・歩く比較の車）の はじめる に「車の設定」。設定したを押すと機材のメモに入り、済になる。自転車の区分には出ない", () => {
  const a = fixed(); a.start(/車の通過/); a.tb("start");
  if (a.E("carSetBox").hidden || !/アイドリングストップを切る/.test(a.E("carSetBox").textContent) || a.E("bdCarSet").textContent !== "未") throw new Error("car settings box");
  a.ck(a.E("btnCarSet")); if (!/車の設定: アイドリングストップ切/.test(a.st().equip || "") || a.E("bdCarSet").textContent !== "済") throw new Error("equip: " + a.st().equip);
  a.ck(a.E("btnCarSet")); if ((a.st().equip.match(/車の設定: /g) || []).length !== 1) throw new Error("written twice");
  const b = fixed(); b.start(/自転車のベル/); b.tb("start"); if (!b.E("carSetBox").hidden) throw new Error("car settings shown for bikes");
  const c = fixed(); c.start(/歩く比較（車）/, "ヘルメット"); c.tb("start"); if (c.E("carSetBox").hidden) throw new Error("car settings hidden for W2_CAR");
});
step("本人 10/17: 10/15 の H を開いたまま保存したバックアップを空の iPad に読み込む → 前の日のセッションは閉じて、カードが C「4 / 16」・H「24 / 60」。同じ日のセッションは開いたまま", () => {
  const f = fix1015(true); f.cur = "20261015_H1";
  const e = boot(html, null), EW = e.window, ED = e.window.document; const n = EW.__note.importBackup(JSON.stringify({ app: "helmet_note", version: 1, state: f }));
  const sx = JSON.parse(EW.localStorage.getItem("helmet_note_v1"));
  if (n !== 2 || sx.cur !== null || ED.getElementById("startNew").hidden) throw new Error("old session left open: " + sx.cur);
  const cd = re => [...ED.querySelectorAll("#catGrid .cat")].find(x => re.test(x.textContent));
  if (!/4 \/ 16 本 済み/.test(cd(/自転車のベル/).textContent) || !/24 \/ 60 本 済み/.test(cd(/自転車（ベルなし）/).textContent)) throw new Error("cards " + cd(/自転車のベル/).textContent + " / " + cd(/自転車（ベルなし）/).textContent);
  const g = JSON.parse(JSON.stringify(f)); const td = EW.eval("(()=>{const d=new Date();return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');})()");
  g.sessions[1].date = td; g.sessions[1].id = td.replace(/-/g, "") + "_H1"; g.cur = g.sessions[1].id;
  const e2 = boot(html, null); e2.window.__note.importBackup(JSON.stringify({ state: g }));
  if (JSON.parse(e2.window.localStorage.getItem("helmet_note_v1")).cur !== g.cur) throw new Error("same-day session closed");
});
step("Astra R97-01（10/17）: 歩く比較の静止で立った現場の x が events.csv の「歩く人の静止x_m」に出る（車の南向き +2・北向き −2・自転車 0・歩行は空）。案内と同じ値", () => {
  const col = (W, name) => { const L = W.__note.eventsCsv().trim().split("\n").map(l => l.split(",")); const i = L[0].indexOf(name), k = L[0].indexOf("台本ID"); if (i < 0) throw new Error("no column " + name); return Object.fromEntries(L.slice(1).map(r => [r[k], r[i]])); };
  const c = fixed(); c.start(/歩く比較（車）/, "ヘルメット"); c.tb("rec");
  if (c.E("itId").textContent !== "W10-静止" || !/現場の x = \+2（南）に立ち/.test(c.E("itFacts").textContent)) throw new Error("W10 facts: " + c.E("itId").textContent + " " + c.E("itFacts").textContent);
  c.tk(); c.tk(); { const x = col(c.W, "歩く人の静止x_m"); if (x["W10-静止"] !== "2" || x["W10-歩行"] !== "") throw new Error("W10 static x: " + JSON.stringify(x)); }
  const n = fixed(); n.start(/歩く比較（車）/, "ヘルメット"); n.tb("rec"); n.E("pickBox").open = true; n.ck(n.chipIn("#itPicker", /W09-静止$/));
  if (!n.E("rotBox").hidden) { n.ck(n.E("btnRotDone")); n.ck(n.E("btnRotDone")); }
  if (!/現場の x = -2（北）に立ち/.test(n.E("itFacts").textContent)) throw new Error("W09 facts: " + n.E("itFacts").textContent);
  n.tk(); { const x = col(n.W, "歩く人の静止x_m"); if (x["W09-静止"] !== "-2") throw new Error("W09 static x: " + JSON.stringify(x)); }
  const b = fixed(); b.start(/歩く比較（自転車）/, "ヘルメット"); b.tb("rec"); b.tk(); { const x = col(b.W, "歩く人の静止x_m"); if (x["W02-静止"] !== "0") throw new Error("bike static x: " + JSON.stringify(x)); }
});
step("Astra R97-02（10/17）: 本を始めたときの合図の目印が events.csv の「合図x_m」に出る。手で変えた・遅いでずらした・上限で変わらない・同じ日のセッションへ引き継いだ、が session.csv の「合図の変更」と使わない録音の理由に残る", () => {
  const a = fixed(); a.start(/歩く比較（自転車）/, "ヘルメット"); a.tb("rec");
  a.ck(a.chipIn("#chipWTrig", /合図 −8/)); if (a.st().wTrig !== "8") throw new Error("manual chip " + a.st().wTrig);
  a.tk();   // W02-静止
  for (let i = 0; i < 2; i++) { a.ck(a.E("btnStart")); a.ck(a.E("btnAbort")); a.ck(a.chipIn("#abReason", /（遅い）/)); a.ck(a.E("abExclude")); }
  const rs = a.st().excluded.map(x => x.reason).join(" | ");
  if (a.st().wTrig !== "10" || !/W02-歩行）・追い越しが \+6（南）を越えた（遅い）・合図 8 → 10/.test(rs) || !/合図は 10 のまま（上限）/.test(rs)) throw new Error("reasons: " + a.st().wTrig + " " + rs);
  a.tk();   // W02-歩行（合図 10）
  const L = a.W.__note.eventsCsv().trim().split("\n").map(l => l.split(",")); const iw = L[0].indexOf("合図x_m"), ik = L[0].indexOf("台本ID");
  const w = Object.fromEntries(L.slice(1).map(r => [r[ik], r[iw]])); if (w["W02-歩行"] !== "-10" || w["W02-静止"] !== "") throw new Error("cue column: " + JSON.stringify(w));
  const sc = a.W.__note.sessionCsv();
  if (!/合図の変更: \d\d:\d\d 6 → 8（手で・練習）；\d\d:\d\d 8 → 10（遅い・W02-歩行・261022_\d{3}\.WAV を使わない）；\d\d:\d\d 10 のまま（遅い・W02-歩行・261022_\d{3}\.WAV を使わない）/.test(sc)) throw new Error("session log: " + sc.split("\n").find(l => /合図/.test(l)));
  a.ck(a.E("btnCloseSession")); a.ck(a.E("btnCloseSession")); a.start(/歩く比較（自転車）/, "ヘルメット");
  if (a.st().wTrig !== "10" || !/10 を引き継ぎ（20261022_/.test(a.W.__note.sessionCsv())) throw new Error("inherit log: " + JSON.stringify(a.st().wTrigLog));
});
step("本人 10/17 23:46 案 2（Astra R97-04・#69 r9）: 発進 B2 は止まったらエンジンを切って待ち、合図でかけてすぐ出る（案内・手順の文・録音中の上の文・車の設定）。通過の本には出ない", () => {
  const a = fixed(); a.start(/発進/); a.tb("rec");
  if (!/B2-/.test(a.E("itId").textContent) || !/エンジンを切って待つ（切ってから録音開始）。合図でエンジンをかけ、かかったらすぐ動き出して/.test(a.E("itFacts").textContent)) throw new Error("B2 facts: " + a.E("itFacts").textContent);
  if (!/運転する人に合図（エンジンをかけ、かかったらすぐ出る/.test(a.E("clapStep").textContent)) throw new Error("B2 clap step: " + a.E("clapStep").textContent);
  a.ck(a.E("btnStart")); if (!/5 つ数えて合図（エンジンをかけ、かかったらすぐ出る）/.test(a.E("nextTxt").textContent)) throw new Error("B2 top text: " + a.E("nextTxt").textContent);
  a.ck(a.E("btnStop")); if (a.st().take) a.ck(a.E("btnStop"));
  a.tb("start"); if (!/発進（B2）だけは、止まったらエンジンを切って待ち/.test(a.E("carSetBox").textContent) || !/B2 が終わったら、アイドリングストップをもう一度切る/.test(a.E("carSetBox").textContent)) throw new Error("car settings text");
  a.ck(a.E("btnCarSet")); if (!/発進 B2 は止まったらエンジンを切って待ち/.test(a.st().equip || "")) throw new Error("equip: " + a.st().equip);
  const c = fixed(); c.start(/車の通過/); c.tb("rec"); if (/エンジンを切って/.test(c.E("itFacts").textContent + c.E("clapStep").textContent)) throw new Error("engine-off text shown for A2");
});
step("本人 10/24〜25: クラクション E2 は三脚を y = +2.0 に置き、車は 2.0 m の線の北に止めたまま、マイクの向きだけ 0 → 90 → 180 → 270°（3 m）→ 270 → 180 → 90 → 0°（7 m）。90°・270° のチップ・回す手順・記録（三脚の向き・マイクから見た位置・特記）・次の発進は点検をやり直す", () => {
  const run = a => { const seq = []; let rot = 0; for (let i = 0; i < 10; i++) { const id = a.E("itId").textContent; if (id === "全部済み") break; if (!a.E("rotBox").hidden) { a.ck(a.E("btnRotDone")); a.ck(a.E("btnRotDone")); rot++; } seq.push(id); a.tk(); } return { seq: seq.join(), rot }; };
  const a = fixed(); a.start(/クラクション/); a.tb("rec"); const r0 = run(a);
  if (r0.seq !== "E2-03,E2-02,E2-01,E2-04,E2-08,E2-05,E2-06,E2-07" || r0.rot !== 6) throw new Error("order: " + JSON.stringify(r0));
  const hs = a.st().takes.map(t => t.tripodHeading).join(); if (hs !== "0,90,180,270,270,180,90,0") throw new Error("headings " + hs);
  const ev = a.W.__note.eventsCsv().trim().split("\n"), H = ev[0].split(","), row = ev.find(l => /E2-02/.test(l)).split(","), col = n => (row[H.indexOf(n)] || "").replace(/^"|"$/g, "");
  if (col("三脚の向き") !== "90" || col("x_m") !== "-3" || col("y_m") !== "2.0" || col("マイク_x_m") !== "0" || col("マイク_y_m") !== "2.0" || !/y = \+2\.0/.test(col("特記"))) throw new Error("csv " + [col("三脚の向き"), col("x_m"), col("y_m"), col("マイク_x_m"), col("マイク_y_m"), col("特記")]);   // Astra H25-01: 音源とマイクの現場の座標を別々に
  a.tb("start"); if (a.E("hornTripodBox").hidden || !/東へ 2 m（y = \+2\.0）/.test(a.E("hornTripodBox").textContent)) throw new Error("horn tripod box");
  const chipsTxt = [...a.D.querySelectorAll("#chipHeading .chip")].map(c => c.textContent).join("|"); if (!/90°（ロゴが現場の \+y＝東）/.test(chipsTxt) || !/270°（ロゴが現場の −y＝西）/.test(chipsTxt)) throw new Error("chips " + chipsTxt);
  a.ck(a.E("btnNextKubun")); a.E("inInherit").checked = true; a.ck(a.E("btnNewSession")); const b = a.st(); if (b.script !== "E3" || b.ritual.check || !b.ritual.calib || b.heading !== "0" || !/点検をやり直す/.test(b.note || "")) throw new Error("B2 after E2: " + JSON.stringify([b.script, b.ritual, b.heading, b.note]));
  const f = fixed(); f.start(/クラクション/); f.tb("rec"); f.E("pickBox").open = true; f.ck(f.chipIn("#itPicker", /E2-02$/));
  if (f.E("rotBox").hidden || !/0° → 90°/.test(f.E("rotTitle").textContent) || !/90°（上から見て反時計回り） ?回す/.test(f.E("rotSteps").textContent) || !/現場の \+y＝東/.test(f.E("rotSteps").textContent)) throw new Error("rot steps: " + f.E("rotTitle").textContent + " / " + f.E("rotSteps").textContent);
  f.ck(f.chipIn("#itPicker", /E2-05$/)); if (!/2\.0 m の線の x = −7（北）/.test(f.E("itFacts").textContent) || !/車はマイクの前/.test(f.E("itFacts").textContent)) throw new Error("E2-05 facts: " + f.E("itFacts").textContent);
  const g = fixed(); g.start(/発進/); g.tb("start"); if (!g.E("hornTripodBox").hidden) throw new Error("horn box on B2");
  const chipsB = [...g.D.querySelectorAll("#chipHeading .chip")].map(c => c.textContent).join("|"); if (/90°/.test(chipsB)) throw new Error("90° chip on B2: " + chipsB);
});
step("Astra H25-03（10/25）: 点検の引き継ぎは三脚の位置で決める。原点の区分（A2）→ クラクション（y = +2.0）は点検をやり直す・クラクション → クラクションは引き継ぐ・校正を引き継がない作り直しでも 90° を引き継ぐ。三脚の本の記録は マイク_x_m・マイク_y_m = 0", () => {
  const a = fixed(); a.start(/車の通過/); a.tk();
  const evA = a.W.__note.eventsCsv().trim().split("\n"), HA = evA[0].split(","), rA = evA.find(l => /A2-/.test(l)).split(",");
  if (rA[HA.indexOf("マイク_x_m")] !== "0" || rA[HA.indexOf("マイク_y_m")] !== "0") throw new Error("A2 mic cols " + [rA[HA.indexOf("マイク_x_m")], rA[HA.indexOf("マイク_y_m")]]);
  a.tb("start"); a.ck(a.E("btnCloseSession")); a.ck(a.E("btnCloseSession")); a.ck(a.card(/クラクション/)); a.E("inInherit").checked = true; a.ck(a.E("btnNewSession"));
  let e = a.st(); if (e.script !== "E2" || e.ritual.check || !e.ritual.calib || !/三脚の位置が変わった（原点 → y = \+2\.0）/.test(e.note || "")) throw new Error("A2 → E2: " + JSON.stringify([e.script, e.ritual, e.note]));
  a.ck(a.E("btnCheckDone")); a.tb("rec"); a.tk();                                   // E2-03（0°）
  if (!a.E("rotBox").hidden) { a.ck(a.E("btnRotDone")); a.ck(a.E("btnRotDone")); } a.tk();   // E2-02（90°）
  if (a.st().heading !== "90") throw new Error("heading after E2-02 " + a.st().heading);
  a.tb("start"); a.ck(a.E("btnCloseSession")); a.ck(a.E("btnCloseSession")); a.ck(a.card(/クラクション/)); a.E("inInherit").checked = true; a.ck(a.E("btnNewSession"));
  e = a.st(); if (e.script !== "E2" || !e.ritual.check || /位置が変わった/.test(e.note || "") || e.heading !== "90") throw new Error("E2 → E2: " + JSON.stringify([e.script, e.ritual, e.note, e.heading]));
  a.tb("start"); a.ck(a.E("btnCloseSession")); a.ck(a.E("btnCloseSession")); a.ck(a.card(/クラクション/)); a.E("inInherit").checked = false; a.ck(a.E("btnNewSession"));
  e = a.st(); if (e.heading !== "90" || e.ritual.calib) throw new Error("restart without inherit: " + JSON.stringify([e.heading, e.ritual]));
});
step("Fable の仕組みの地図 §8-3（10/25・Version 124）: E2 の本の「台本と違ったときだけ直す」の向きは 0/90/180/270・直したらマイクから見た向き（象限）も合わせる・マイクの位置の欄がある。A2 の本は 0/180 のままでマイクの欄は出ない", () => {
  const rowsOf = a => { const ev = a.W.__note.eventsCsv().trim().split("\n"), H = ev[0].split(","); return ev.slice(1).map(l => { const c = l.split(","), o = {}; H.forEach((h, i) => o[h] = (c[i] || "").replace(/^"|"$/g, "")); return o; }); };
  const openEdit = a => { a.tb("list"); a.ck(a.D.querySelectorAll("#listTable tr")[1].querySelector("button")); };
  const chipsOf = a => [...a.D.querySelectorAll("#editCard .chips .chip")].map(c => c.textContent);
  const labelsOf = a => [...a.D.querySelectorAll("#editCard .field > label")].map(l => l.textContent);
  const a = fixed(); a.start(/クラクション/); a.tk();   // E2-03（0°・後）
  let r = rowsOf(a).find(x => x["台本ID"] === "E2-03"); if (!r || r["三脚の向き"] !== "0" || r["象限"] !== "B") throw new Error("E2-03 before " + JSON.stringify(r && [r["三脚の向き"], r["象限"]]));
  openEdit(a); const ce = chipsOf(a); if (!ce.includes("90°") || !ce.includes("270°")) throw new Error("E2 edit chips " + ce.join("|"));
  const le = labelsOf(a); if (!le.some(l => /^マイク x m/.test(l)) || !le.some(l => /^マイク y m/.test(l))) throw new Error("mic fields " + le.join(" / "));
  a.ck(a.chipIn("#editCard", /^270°$/));
  r = rowsOf(a).find(x => x["台本ID"] === "E2-03"); if (r["三脚の向き"] !== "270" || r["象限"] !== "R") throw new Error("E2-03 after 270 " + [r["三脚の向き"], r["象限"]]);
  openEdit(a); const f = [...a.D.querySelectorAll("#editCard .field")].find(e => /^マイク y m/.test((e.querySelector("label") || {}).textContent || "")); const inp = f.querySelector("input"); inp.value = "1.8"; inp.dispatchEvent(new a.W.Event("change", { bubbles: true }));
  r = rowsOf(a).find(x => x["台本ID"] === "E2-03"); if (r["マイク_y_m"] !== "1.8" || r["マイク_x_m"] !== "0" || r["y_m"] !== "2.0") throw new Error("mic y after fix " + [r["マイク_x_m"], r["マイク_y_m"], r["y_m"]]);
  const b = fixed(); b.start(/車の通過/); b.tk(); openEdit(b); const cb = chipsOf(b);
  if (cb.includes("90°") || cb.includes("270°") || !cb.includes("180°")) throw new Error("A2 edit chips " + cb.join("|"));
  if (labelsOf(b).some(l => /^マイク [xy] m/.test(l))) throw new Error("mic fields on A2");
});
step("文書の点検（10/18）: ベルの前後の位置が空なら C2・歩く比較（自転車）で「最初にやること ④」と保存タブの指摘に出る（0 は値・H2 と車には出ない）。クラクションは拍手のあと 5 つ数えて合図・鳴らして 5 秒で停止。夜の step19h の行はキーを押さない", () => {
  const a = fixed(); a.start(/自転車のベル/); a.tb("rec");
  if (!/④ ベルの前後の位置/.test(a.E("ritualWarn").textContent) || a.E("ritualWarn").hidden) throw new Error("C2 reminder: " + a.E("ritualWarn").textContent);
  a.tk(); a.tb("save"); if (!/ベルの前後の位置（前輪の軸からベルまで）が空/.test(a.E("issues").textContent)) throw new Error("save issue missing");
  a.tb("start"); a.sv("inBellOff", "0"); a.tb("rec"); if (!a.E("ritualWarn").hidden) throw new Error("0 cm not accepted: " + a.E("ritualWarn").textContent);
  a.tb("start"); a.E("inBellSame").checked = true; a.E("inBellSame").dispatchEvent(new a.W.Event("change", { bubbles: true }));   // Astra 再監査 10/18: 10/15 と同じ自転車・同じ付け方の確かめを残す
  if (a.st().bellSame !== "y" || !/ベルの付け方: 10\/15 と同じ自転車・同じ付け方（確かめた）/.test(a.W.__note.sessionCsv())) throw new Error("bellSame not recorded");
  a.tb("save"); if (/ベルの前後の位置/.test(a.E("issues").textContent)) throw new Error("issue after 0 cm");
  const w = fixed(); w.start(/歩く比較（自転車）/, "ヘルメット"); w.tb("rec"); if (!/④ ベルの前後の位置/.test(w.E("ritualWarn").textContent)) throw new Error("W2_BIKE reminder");
  const h = fixed(); h.start(/自転車（ベルなし）/); h.tb("rec"); if (/ベルの前後/.test(h.E("ritualWarn").textContent)) throw new Error("H2 reminder shown");
  const e = fixed(); e.start(/クラクション/); e.tb("rec");
  if (!/拍手のあと 5 つ数えてから合図 → 短く 1 回鳴らす → 鳴らして 7 秒たったら停止/.test(e.E("itFacts").textContent) || /3 秒数えて/.test(e.E("itFacts").textContent)) throw new Error("E2 facts: " + e.E("itFacts").textContent);
  if (!/5 つ数えてから運転席の人に合図（短く 1 回鳴らす）。鳴らして 7 秒たったら/.test(e.E("clapStep").textContent)) throw new Error("E2 clap step: " + e.E("clapStep").textContent);
  e.ck(e.E("btnStart")); if (!/5 つ数えて合図（短く 1 回鳴らす） → 鳴らして 7 秒で/.test(e.E("nextTxt").textContent)) throw new Error("E2 top: " + e.E("nextTxt").textContent);
  e.ck(e.E("btnStop")); if (e.st().take) e.ck(e.E("btnStop"));
  const out = e.W.__note.cmdText(); if (!/キーは押さない/.test(out) || /q で保存して終了）/.test(out)) throw new Error("night step19h line");
});
step("2026-10-25（地図の調べ・敵対レビュー 重大 2）: 夜の命令の負例は none の行がある原本だけを 1 本ずつ、正例と同じ ann_all.csv に --append（raw/ を丸ごと渡さない＝正例の本を負例に切らない・注釈の無い録音で止まらない）", () => {
  const a = fixed(); a.start(/車の通過/); a.tk();
  const c = a.W.__note.cmdText(), negA = c.split("\n").filter(l => /--mode negative/.test(l));
  const carFile = a.st().takes[a.st().takes.length - 1].file;
  if (/--mode negative --in raw\/ /.test(c) || /ann_neg\.csv/.test(c) || negA.some(l => l.includes("raw/" + carFile + " "))) throw new Error("negative lines (car day): " + negA.join(" | "));
  const ev = c.split("\n").find(l => /--mode event/.test(l)) || "";
  if (!/--ann-out out\/realsmoke\/ann_all\.csv/.test(ev) || /--append/.test(ev)) throw new Error("event line: " + ev);   // 正例は先に新しく書く（--append なし）
  const g = fixed(); g.start(/何も来ない/); g.tk();
  const gFile = g.st().takes[g.st().takes.length - 1].file, cg = g.W.__note.cmdText();
  const line = cg.split("\n").find(l => /--mode negative/.test(l) && l.includes("--in raw/" + gFile + " ")) || "";
  if (!gFile || !/--ann-out out\/realsmoke\/ann_all\.csv --append/.test(line)) throw new Error("G negative line: " + line + " / file " + gFile);
  if (/--mode negative --in raw\/ /.test(cg) || !/回し直すときは上の step19b --mode event の行から順に/.test(cg)) throw new Error("folder-wide line or rerun note: " + cg.split("\n").filter(l => /負例/.test(l)).join(" | "));
});
step("2026-10-25（Version 128・#69 r43・r44）: 夜の命令の最初に台帳の下書き、校正の変換の行のすぐ前に較正の照らし合わせ（窓は既定 5-60・終了コード 3 は値と窓を変換の行へ・6 は止める）", () => {
  const a = fixed(); a.start(/車の通過/); a.tk();
  const lines = a.W.__note.cmdText().split("\n");
  const iLed = lines.findIndex(l => /field_ledger_draft\.py --raw raw\/ --date 20261022 --backup raw\/sheet\/20261022\/helmet_note_backup_20261022\.json --out-dir raw\/ledger/.test(l));
  const iCal = lines.findIndex(l => /^& \$PY scripts\/field_calib_check\.py --wav raw\/\S+ --slm raw\/slm\/20261022\/ --laeq \S+ --window 5-60 --json out\/realsmoke\/calib_check_\S+\.json --convention v12_149; \$rc = \$LASTEXITCODE; /.test(l));
  if (iLed < 0 || iCal < 0 || iLed > iCal) throw new Error(`order: ledger ${iLed} calib ${iCal}\n` + lines.filter(l => /field_|step19_realsmoke_convert/.test(l)).join("\n"));
  // Astra MAP25-04（Version 129）: 照らし合わせと変換は 1 行で終了コードを見て分ける（0 = 画面・4／5 = 要確認で画面・3 = json の値と窓・ほか = 止める）。変換だけの行は残さない
  const c = lines[iCal];
  if (!/if \(\$rc -eq 0\) \{ & \$PY scripts\/step19_realsmoke_convert\.py /.test(c) || !/elseif \(\$rc -in 4,5\) \{ Write-Host "要確認/.test(c)
      || !/elseif \(\$rc -eq 3\) \{ \$j = Get-Content out\/realsmoke\/calib_check_\S+\.json -Raw \| ConvertFrom-Json;/.test(c) || !/--laeq \$j\.laeq_use --laeq-window \$j\.window --gain-only/.test(c)
      || !/\} else \{ Write-Host "止める/.test(c)) throw new Error("calib branch line: " + c);
  if (!/6／2／ほか = 止める/.test(lines[iCal - 1])) throw new Error("calib comment line: " + lines[iCal - 1]);
  // 先行分 §3（10/26・Astra R30-08）: 照らし合わせ 1 か所と変換 3 か所（0・4／5・3 の分かれ道）に --convention v12_149（省くと旧の 143）・注記に 149 の約束
  if ((c.match(/--convention v12_149/g) || []).length !== 4 || !/149 の約束/.test(lines[iCal - 1])) throw new Error("convention 149: " + c);
  if (lines.some(l => /^& \$PY scripts\/step19_realsmoke_convert\.py/.test(l))) throw new Error("standalone convert line remains");
  if (process.env.DUMP_CMD) require("fs").writeFileSync(process.env.DUMP_CMD, lines.join("\n"), "utf8");   // PowerShell で分かれ道を回して確かめる用（試験の外の確かめ・10/25）
});
step("2026-10-26（Version 137・本体 §2-6 の L3）: 夜の命令の切り出しの後ろに確かめ step19b_check_history の 1 行（採点の門のとき・--warmup-v30 の注記）・step19c より前", () => {
  const a = fixed(); a.start(/車の通過/); a.tk();
  const lines = a.W.__note.cmdText().split("\n");
  const iH = lines.findIndex(l => /^# （切り出しのあと・採点の門のとき＝本人の合図のあと・149 の変換と一緒に）確かめ: & \$PY scripts\/step19b_check_history\.py out\/realsmoke\/ann_all\.csv --out out\/realsmoke\/history_check\.csv/.test(l));
  const iCut = lines.findIndex(l => /step19b_realsmoke_cut\.py --mode event/.test(l)), iVal = lines.findIndex(l => /step19c_ann_validate\.py/.test(l));
  if (iH < 0 || !(iCut < iH && iH < iVal) || !/--warmup-v30 --min-history 4\.0/.test(lines[iH])) throw new Error(`history check line: ${iH} cut ${iCut} validate ${iVal}`);
});
step("文書の点検 1-3（10/18）: 題の横に版の日付（前の晩に最新の版かを確かめる・自転車の日の流れ §3 の 8）", () => {
  if (!/^版 10\/(1[89]|2\d)[a-z]?$|^版 1[01]\/\d+[a-z]?$/.test($("verStamp").textContent)) throw new Error("version stamp: " + $("verStamp").textContent);
});
step("本人 10/18 17:59: 長回しと G の「録る」画面に、知らせる相手の決まり（車・自転車は 3.2 m より近くだけ・警告音は距離に関係なく）と、② を押す／録り直すものを出す。ほかの区分には出ない", () => {
  const n = fixed(); n.start(/長回し/); n.tb("rec");
  if (n.E("negRuleBox").hidden || !/3\.2 m より近くを通るときだけ知らせる/.test(n.E("negRuleBox").textContent) || !/距離に関係なく知らせる/.test(n.E("negRuleBox").textContent) || !/長回しで ② を押すのは/.test(n.E("negRuleAct").textContent) || !/人・話し声・遠くの車では押さない/.test(n.E("negRuleAct").textContent) || !/3\.2 m より近く/.test(n.E("negRuleAct").textContent)) throw new Error("long-run rule: " + n.E("negRuleBox").textContent);
  n.ck(n.E("btnStart")); pv(n.E, n.ck, "btnClap"); if (!/3\.2 m より近くを車・自転車が通ったら、または警告音/.test(n.E("nextTxt").textContent) || !/人・遠くの車は押さない/.test(n.E("btnEvent").textContent)) throw new Error("long-run running text: " + n.E("nextTxt").textContent + " / " + n.E("btnEvent").textContent);
  const g = fixed(); g.start(/何も来ない/); g.tb("rec");
  if (g.E("negRuleBox").hidden || !/G で録り直すのは/.test(g.E("negRuleAct").textContent) || !/遠くの車の音では録り直さない/.test(g.E("negRuleAct").textContent)) throw new Error("G rule: " + g.E("negRuleAct").textContent);
  const c = fixed(); c.start(/自転車のベル/); c.tb("rec"); if (!c.E("negRuleBox").hidden) throw new Error("rule shown for C2");
});
step("本人 10/18 21:28（Astra R105-01）: 長回しで ② のあと「② の終わり」が出て、押すと終わりの秒が入り events.csv の「終わり秒」に出る。もう一度 ② を押すとまた出る。ほかの区分には出ない", () => {
  const n = fixed(); n.start(/長回し/); n.tb("rec"); n.ck(n.E("btnStart")); pv(n.E, n.ck, "btnClap");
  if (!n.E("btnEventEnd").hidden) throw new Error("end button before any event");
  n.ck(n.E("btnEvent")); if (n.E("btnEventEnd").hidden) throw new Error("end button not shown after ②");
  n.ck(n.E("btnEventEnd")); const ev = n.st().take.events.find(e => e.cls !== "none"); if (ev.lapEnd === "" || ev.lapEnd == null || !n.E("btnEventEnd").hidden) throw new Error("end not recorded: " + JSON.stringify(ev));
  n.ck(n.E("btnEvent")); if (n.E("btnEventEnd").hidden) throw new Error("end button not shown for 2nd event");
  n.ck(n.E("btnStop")); if (n.st().take) n.ck(n.E("btnStop"));
  const L = n.W.__note.eventsCsv().trim().split("\n").map(l => l.split(",")); const ie = L[0].indexOf("終わり秒"); if (ie < 0 || !L.slice(1).some(r => r[ie] !== "")) throw new Error("終わり秒 column");
  const c = fixed(); c.start(/自転車のベル/); c.tb("rec"); c.ck(c.E("btnStart")); if (!c.E("btnEventEnd").hidden) throw new Error("end button shown for C2");
});
step("Astra R108-03（10/19）: 長回しで相手が重なっても「② の終わり」は選んだ相手に付く（サイレン → 車 → サイレンの終わり・同じ種類 2 つ・逆の順・閉じる・止めるときの指摘）", () => {
  const n = fixed(); n.start(/長回し/); n.tb("rec"); n.ck(n.E("btnStart")); pv(n.E, n.ck, "btnClap");
  const setCls = (i, name) => { const ed = n.E("evBox").children[i]; const b = [...ed.querySelectorAll("button")].find(x => x.textContent === name); if (!b) throw new Error("chip " + name); n.ck(b); };
  const pick = (id) => { const b = [...n.E("endPick").querySelectorAll("button")].find(x => x.dataset.ev === String(id)); if (!b) throw new Error("pick " + id + ": " + n.E("endPick").textContent); n.ck(b); };
  const ev = (id) => n.st().take.events.find(e => e.event_id === id);
  n.ck(n.E("btnEvent")); setCls(0, "サイレン");
  if (!/サイレン .* 秒〜/.test(n.E("btnEventEnd").textContent)) throw new Error("label 1: " + n.E("btnEventEnd").textContent);
  n.ck(n.E("btnEvent"));   // 車（既定）
  if (!/どれが終わったか選ぶ/.test(n.E("btnEventEnd").textContent)) throw new Error("label 2: " + n.E("btnEventEnd").textContent);
  n.ck(n.E("btnEventEnd")); if (n.E("endPick").hidden || n.E("endPick").querySelectorAll("button[data-ev]").length !== 2) throw new Error("chooser");
  pick(1); if (ev(1).lapEnd === "" || ev(1).lapEnd == null || (ev(2).lapEnd !== "" && ev(2).lapEnd != null)) throw new Error("wrong target: " + JSON.stringify([ev(1).lapEnd, ev(2).lapEnd]));
  if (!n.E("endPick").hidden || !/車 .* 秒〜/.test(n.E("btnEventEnd").textContent)) throw new Error("after pick: " + n.E("btnEventEnd").textContent);
  // 同じ種類が 2 つ・逆の順で終わる
  n.ck(n.E("btnEvent")); setCls(2, "サイレン"); n.ck(n.E("btnEvent")); setCls(3, "サイレン");
  n.ck(n.E("btnEventEnd")); pick(4); if (ev(4).lapEnd === "" || (ev(3).lapEnd !== "" && ev(3).lapEnd != null)) throw new Error("same class: " + JSON.stringify([ev(3).lapEnd, ev(4).lapEnd]));
  // 閉じる
  n.ck(n.E("btnEventEnd")); const close = [...n.E("endPick").querySelectorAll("button")].find(x => x.dataset.close); n.ck(close); if (!n.E("endPick").hidden || (ev(3).lapEnd !== "" && ev(3).lapEnd != null)) throw new Error("close");
  // 止めるとき: サイレン 3 の終わりが空 → 1 回目は止めずに指摘
  n.ck(n.E("btnStop")); if (!n.st().take || !/サイレン（.* 秒〜）の終わりが空/.test(n.E("btnStop").textContent)) throw new Error("stop warn: " + n.E("btnStop").textContent);
  n.ck(n.E("btnStop")); if (n.st().take) n.ck(n.E("btnStop"));
  const L = n.W.__note.eventsCsv().trim().split("\n").map(l => l.split(",")); const ie = L[0].indexOf("終わり秒"), ii = L[0].indexOf("event_id");
  const sid = n.st().id, is = L[0].indexOf("session_id"); const mine = L.slice(1).filter(r => r[is] === sid);   // 10/15 の記録の行も入っているので、今のセッションの行だけ
  const endOf = id => (mine.find(r => r[ii] === String(id)) || [])[ie];
  if (endOf(1) === "" || endOf(2) !== "" || endOf(3) !== "" || endOf(4) === "") throw new Error("csv: " + [1, 2, 3, 4].map(endOf).join("|"));
});
step("Astra R109-02（10/19）: 相手が 2 つ以上のとき、「② の終わり」を押した秒が終わりの秒になる（選ぶ間の時間を足さない）・閉じたら捨てる・選ぶ間に来た相手は候補にしない", () => {
  const n = fixed(); let sec = 0; const T0 = 1.9e12; n.W.Date.now = () => T0 + sec * 1000;
  n.start(/長回し/); n.tb("rec"); n.ck(n.E("btnStart")); pv(n.E, n.ck, "btnClap");
  const setCls = (i, name) => { const ed = n.E("evBox").children[i]; const b = [...ed.querySelectorAll("button")].find(x => x.textContent === name); if (!b) throw new Error("chip " + name); n.ck(b); };
  const pick = (id) => { const b = [...n.E("endPick").querySelectorAll("button")].find(x => x.dataset.ev === String(id)); if (!b) throw new Error("pick " + id + ": " + n.E("endPick").textContent); n.ck(b); };
  const ev = (id) => n.st().take.events.find(e => e.event_id === id);
  sec = 10; n.ck(n.E("btnEvent")); setCls(0, "サイレン");
  sec = 20; n.ck(n.E("btnEvent"));
  sec = 30; n.ck(n.E("btnEventEnd")); if (n.E("endPick").hidden || !/30 秒に終わった相手/.test(n.E("endPick").textContent)) throw new Error("hold text: " + n.E("endPick").textContent);
  sec = 35; pick(1); if (Number(ev(1).lapEnd) !== 30 || (ev(2).lapEnd !== "" && ev(2).lapEnd != null)) throw new Error("held end: " + JSON.stringify([ev(1).lapEnd, ev(2).lapEnd]));
  sec = 41; n.ck(n.E("btnEvent")); setCls(2, "サイレン");
  sec = 45; n.ck(n.E("btnEventEnd")); const close = [...n.E("endPick").querySelectorAll("button")].find(x => x.dataset.close); n.ck(close);
  sec = 50; n.ck(n.E("btnEventEnd")); pick(2); if (Number(ev(2).lapEnd) !== 50) throw new Error("after close: " + ev(2).lapEnd);
  sec = 60; n.ck(n.E("btnEvent")); setCls(3, "サイレン");
  sec = 62; n.ck(n.E("btnEventEnd"));
  sec = 63; n.ck(n.E("btnEvent"));   // 選ぶ間に来た相手
  if ([...n.E("endPick").querySelectorAll("button[data-ev]")].some(b => b.dataset.ev === "5")) throw new Error("late event offered");
  sec = 64; pick(3); if (Number(ev(3).lapEnd) !== 62 || (ev(5).lapEnd !== "" && ev(5).lapEnd != null)) throw new Error("late: " + JSON.stringify([ev(3).lapEnd, ev(5).lapEnd]));
  if (n.st().take.events.some(e => e.lapEnd !== "" && e.lapEnd != null && Number(e.lapEnd) < Number(e.lap))) throw new Error("end before start");
  sec = 70; n.ck(n.E("btnStop")); if (!/止める前に「② の終わり」を押す/.test(n.E("btnStop").textContent)) throw new Error("stop guide: " + n.E("btnStop").textContent);
  n.ck(n.E("btnStop")); if (n.st().take) n.ck(n.E("btnStop"));
});
step("10/20 本人「関係ない入力項目がある」: 隠した欄（hidden）が本当に見えない＝[hidden]{display:none!important} がある・歩く比較（自転車）を選ぶと車の動力とブザーの台の高さは隠れ、ベルの高さは出る", () => {
  const n = fresh();
  if (!/\[hidden\]\{display:none!important\}/.test(n.W.document.querySelector("style").textContent)) throw new Error("no [hidden] rule");
  const card = [...n.E("catGrid").children].find(b => /歩く比較（自転車）/.test(b.textContent) && !/#69 の前/.test(b.textContent)); if (!card) throw new Error("no W2_BIKE card"); n.ck(card);
  const shown = id => !n.E(id).hidden && n.W.getComputedStyle(n.E(id)).display !== "none";
  if (shown("fPower") || shown("fSrcZ")) throw new Error("irrelevant shown: " + [shown("fPower"), shown("fSrcZ")]);
  if (!shown("fBellZ")) throw new Error("bell height hidden");
  const hid = n.E("fPower"); if (n.W.getComputedStyle(hid).display !== "none") throw new Error("computed display of hidden .field = " + n.W.getComputedStyle(hid).display);
  const car = [...n.E("catGrid").children].find(b => /車の通過/.test(b.textContent) && !/#69 の前/.test(b.textContent)); n.ck(car);
  if (!shown("fPower") || shown("fBellZ") || shown("fSrcZ")) throw new Error("car form: " + [shown("fPower"), shown("fBellZ"), shown("fSrcZ")]);
});
step("敵対レビュー 10/21 A6: 前の日（10/15）の記録に片づけが要るものがあっても、その日（10/22）の CSV の保存は止めない。その日の記録なら止める", () => {
  const mk22 = (fx, open) => { const s = JSON.parse(JSON.stringify(fx.sessions[1])); s.id = "20261022_H2"; s.date = "2026-10-22"; s.takes = []; if (open) s.take = { take_id: 1, file: "261022_005.WAV", running: true, events: [], t0: 1 }; fx.sessions.push(s); };
  const run = fx => { const dm = boot(html, JSON.stringify(fx)), W = dm.window, D = W.document; D.querySelector("nav button[data-tab=save]").dispatchEvent(new W.Event("click", { bubbles: true })); return { dis: D.getElementById("btnFileS").disabled, txt: D.getElementById("issues").textContent }; };
  const a = fix1015(); a.sessions[0].take = { take_id: 99, file: "261015_099.WAV", running: true, events: [], t0: 1 }; mk22(a, false);
  const ra = run(a); if (ra.dis || !/前の日の記録の分 1 件は保存を止めない/.test(ra.txt)) throw new Error("past blocks: " + ra.dis + " " + ra.txt.slice(0, 120));
  const b = fix1015(); mk22(b, true);
  const rb = run(b); if (!rb.dis || !/片づけが要るもの 1 件/.test(rb.txt)) throw new Error("today not blocked: " + rb.dis + " " + rb.txt.slice(0, 120));
});
step("敵対レビュー 10/21 B10: はじめるの画面のベルの高さの欄に、同じ日に測った値（1.05）が出る", () => {
  const fx = fix1015(); const s = JSON.parse(JSON.stringify(fx.sessions[0])); s.id = "20261022_C2"; s.date = "2026-10-22"; s.bellZ = "1.05"; s.takes = []; fx.sessions.push(s);
  const dm = boot(html, JSON.stringify(fx)), W = dm.window, D = W.document; const E = id => D.getElementById(id); const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  E("inDate").value = "2026-10-22"; E("inDate").dispatchEvent(new W.Event("change", { bubbles: true }));
  const card = [...D.querySelectorAll("#catGrid .cat")].find(x => /歩く比較（自転車）/.test(x.textContent) && !/#69 の前/.test(x.textContent)); ck(card);
  if (E("inBellZ").value !== "1.05") throw new Error("bell height on form: " + E("inBellZ").value);
});
// Astra 監査 10/7 V04: 偽のサーバ（window.claude.use('db')）で、1 本録ってすぐ（1.5 秒以内に）次の区分を始めても、前のセッションがサーバに写る
async function stepAsync(name, fn) { try { await fn(); console.log("ok  ", name); } catch (e) { console.log("FAIL", name, "-", e.message); process.exitCode = 1; } }
step("#59 r4: 自転車・車の長さの欄。入れると session.csv の備考に入り、同じ日の次のセッションに引き継ぐ", () => {
  const a = fresh(); a.begin();
  if (a.E("rDims").hidden || a.E("fBikeDim").hidden || !a.E("fCarDim").hidden) throw new Error("bike card: " + [a.E("rDims").hidden, a.E("fBikeDim").hidden, a.E("fCarDim").hidden]);
  a.sv("inBikeWb", "104");
  if (a.st().bikeWb !== "104") throw new Error(JSON.stringify([a.st().bikeWb]));
  if (!/自転車の長さ: 軸の間 104 cm/.test(a.W.__note.sessionCsv())) throw new Error("bike not in csv");
  a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /自転車（ベルなし）/.test(x.textContent))); a.ck(a.E("btnNewSession"));
  if (a.st().kubun !== "H" || a.st().bikeWb !== "104") throw new Error("not inherited: " + JSON.stringify([a.st().kubun, a.st().bikeWb]));
  a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /車の通過/.test(x.textContent))); a.ck(a.E("btnNewSession"));
  if (a.E("fCarDim").hidden || !a.E("fBikeDim").hidden) throw new Error("car card");
  a.sv("inCarModel", "テスト車"); a.sv("inCarWb", "270");
  if (!/車の長さ: 車種 テスト車・軸の間 270 cm/.test(a.W.__note.sessionCsv())) throw new Error("car not in csv");
  if (a.E("inCarFront") || a.E("inBikeR")) throw new Error("front fields still shown (#59 r6)");
  const h = a.W.__note.sessionCsv().split(/\r?\n/)[0].split(","); if (h.length !== 23) throw new Error("session.csv columns " + h.length);
});
step("Astra MP11-01: 歩く比較は録った本のクラスで要る長さを決め、0 以下・範囲外も「確かめること」に出す。車種を変えると車の長さは空に戻る", () => {
  const base = { gain: "70", ritual: { calib: true, check: true }, checkFile: "c.WAV", checks: { "c.WAV": { h: "0" } }, ver: 54 };
  const tk = cls => [{ take_id: 1, file: "a.WAV", clapA: "", tripodHeading: "0", events: [{ lap: "", cls: cls, pair: "1" }] }];
  const w1 = Object.assign({ id: "W", kubun: "歩行対比", bikeWb: "104", bikeR: "34", takes: tk("car_drive") }, base);
  if (!w.__note.issuesOf(w1).some(x => /車の長さ.*空か 0 以下/.test(x.text))) throw new Error("walking car dims not warned: " + JSON.stringify(w.__note.issuesOf(w1)));
  const c0 = Object.assign({ id: "C0", kubun: "C", bikeWb: "0", bikeR: "34", takes: tk("bike_bell") }, base);
  if (!w.__note.issuesOf(c0).some(x => /自転車の長さ.*空か 0 以下/.test(x.text))) throw new Error("zero not warned");
  const cm = Object.assign({ id: "CM", kubun: "C", bikeWb: "1.04", bikeR: "0.34", takes: tk("bike_bell") }, base);
  if (!w.__note.issuesOf(cm).some(x => /範囲の外/.test(x.text))) throw new Error("meters not warned");
  const ok = Object.assign({ id: "OK", kubun: "C", bikeWb: "104", bikeR: "34", takes: tk("bike_bell") }, base);
  if (w.__note.issuesOf(ok).some(x => /長さ/.test(x.text))) throw new Error("false dims warning");
  const a = fresh(); a.begin();
  a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /車の通過/.test(x.textContent))); a.ck(a.E("btnNewSession"));
  a.sv("inCarModel", "車A"); a.sv("inCarWb", "270"); a.sv("inCarModel", "車B");
  if (a.st().carModel !== "車B" || a.st().carWb !== "") throw new Error("not cleared: " + JSON.stringify([a.st().carModel, a.st().carWb]));
});
step("#59 r6: ベル・クラクションの本は、出力タブに「鳴り始めをあとで読む本」として並ぶ（読み忘れを防ぐ）", () => {
  const a = fresh(); a.begin(); a.tk();
  const files = a.st().takes.filter(t => t.scriptId).map(t => t.file);
  if (!files.length) throw new Error("no scripted take");
  const c = a.W.__note.cmdText(); if (!new RegExp("# 鳴り始めをあとで読む本.*" + files[0].replace(".", "\\.")).test(c)) throw new Error(c.split("\n").filter(l => /鳴り始め/.test(l)).join(" / ") || "no onset line");
});
step("10/29 版（本人 10/27 12:35 (c)）: 夜の手順に「ベルは鳴り始めと鳴り終わりを波形で読む」（b と e・step19k の --warn-interval）・版の印は 10/29", () => {
  const a = fresh(); a.begin(); a.tk();
  const c = a.W.__note.cmdText();
  if (!/# ベル(と走りながらのクラクション（E3）)?は鳴り始めと鳴り終わりを波形で読む.*b = 鳴り始め・e = 鳴り終わり.*--warn-interval/.test(c)) throw new Error(c.split("\n").filter(l => /ベル/.test(l)).join(" / ") || "no bell end line");
  if (!/鳴り始めをあとで読む本（ベルは鳴り終わりも/.test(c) && /# 鳴り始めをあとで読む本/.test(c)) throw new Error("late list wording");
  if (!/^版 10\/29[a-z]?$/.test($("verStamp").textContent)) throw new Error("version stamp: " + $("verStamp").textContent);
});
step("10/29 版 b（本人 10/27 13:20）: クラクションは振り切ったらゲインを 10 下げて鳴らし直す・B2 から 70 に戻す／歩く比較（車）の間は原点の三脚を車の通り道から離す", () => {
  const box = $("hornTripodBox").textContent;
  if (!/振り切ったら「ハプニング → 振り切った（割れた）→ やり直す」/.test(box) || !/70 → 60 → 50 …/.test(box)) throw new Error("horn gain step: " + box.slice(0, 200));   // Astra R142-01（10/27）: 操作をハプニングの「振り切った」に
  if (!/発進（B2）からゲインを 70 に戻す/.test(box)) throw new Error("gain back to 70 on B2");
  const car = w.__note.scriptOf("W2_CAR"), bike = w.__note.scriptOf("W2_BIKE");
  if (!car.length || !bike.length) throw new Error("no W2 items");
  const fc = w.__note.factsOf(car[0], {}).join(" "), fb = w.__note.factsOf(bike[0], {}).join(" ");
  if (!/原点の三脚：歩く比較のあいだは使わない.*車の通り道から離しておく/.test(fc)) throw new Error("tripod line on W2_CAR: " + fc.slice(0, 300));
  if (/原点の三脚：歩く比較のあいだは使わない/.test(fb)) throw new Error("tripod line should not be on W2_BIKE");
});
step("10/29 版 c（本人 10/27 14:13 案 A）: 走りながらのクラクション E3 = 4 本（2.0 m・時速 15 km・手前の ∓8 で短く 1 回）・E2 → E3 → B2・夜の命令は E=12", () => {
  const L = w.__note.scriptOf("E3");
  if (L.length !== 4 || L.map(x => x.id).join(",") !== "E3-01,E3-03,E3-04,E3-02") throw new Error("E3 items: " + L.map(x => x.id).join(","));
  if (!L.every(x => x.cls === "horn" && x.moving && x.line === "2.0" && x.kmh === "15" && !x.src)) throw new Error("E3 fields");
  const f1 = w.__note.factsOf(L[0], {}).join(" "), f2 = w.__note.factsOf(L.find(x => x.dir === "前→後"), {}).join(" ");
  if (!/前バンパーが現場の x = -8（北）の横テープを越えたら、短く 1 回鳴らす/.test(f1)) throw new Error("honk line 後→前: " + f1.slice(0, 400));
  if (!/前バンパーが現場の x = \+8（南）の横テープを越えたら、短く 1 回鳴らす/.test(f2)) throw new Error("honk line 前→後: " + f2.slice(0, 400));
  if (/東側（カメラ側）の前輪を/.test(f1)) throw new Error("launch text leaked into E3");
  if (!/^版 10\/29[c-z]$/.test($("verStamp").textContent)) throw new Error("version stamp: " + $("verStamp").textContent);
  const a = fresh(); a.begin(); a.tk(); const c = a.W.__note.cmdText();
  if (!/D=0,E=12,H=60/.test(c) || /E=8,/.test(c)) throw new Error("plan E=12");
  if (!/# ベルと走りながらのクラクション（E3）は鳴り始めと鳴り終わりを波形で読む/.test(c)) throw new Error("night line E3");
  if (!/次の区分（走りながらのクラクション E3・4 本）は原点で点検をやり直す/.test($("hornTripodBox").textContent)) throw new Error("horn box E3 line");
});
step("10/29 版 d（#69 r49）: 採点の門の命令に field_gate_prep → step19l は reads_A_gate.csv・車の設定にホーンの前後の位置を測る", () => {
  const a = fresh(); a.begin(); a.tk(); const c = a.W.__note.cmdText();
  const iP = c.indexOf("scripts/field_gate_prep.py --reads out/realsmoke/reads/reads_A.csv"), iL = c.indexOf("scripts/step19l_when_where.py --reads out/realsmoke/reads/reads_A_gate.csv");
  if (iP < 0 || iL < 0 || iP > iL) throw new Error("gate prep before step19l: " + iP + " " + iL);
  if (/step19l_when_where\.py --reads out\/realsmoke\/reads\/reads_A\.csv/.test(c)) throw new Error("old step19l input remains");
  if (!/--out-warn out\/realsmoke\/reads\/warn_interval_A\.csv/.test(c)) throw new Error("warn interval out");
  if (!/ホーンの前後の位置/.test($("carSetBox").textContent) || !/前輪の軸から前へ N cm/.test($("carSetBox").textContent)) throw new Error("car box horn position");
  if (!/^版 10\/29[d-z]$/.test($("verStamp").textContent)) throw new Error("version stamp: " + $("verStamp").textContent);
});
step("#69: 歩く比較は自転車（W01〜W08・16 本）と車（W09〜W16・16 本）の 2 つのボタン。いつも 静止 → 歩行。途中で閉じても次のセッションは残りの組から", () => {
  const L = w.__note.scriptOf("W2_BIKE"), Lc = w.__note.scriptOf("W2_CAR"), all = w.__note.scriptOf("W2");
  if (L.length !== 16 || Lc.length !== 16 || L.some(x => x.cls === "car_drive") || Lc.some(x => x.cls !== "car_drive")) throw new Error("split counts " + L.length + "/" + Lc.length);
  if ([...L, ...Lc].map(x => x.id).sort().join() !== all.map(x => x.id).sort().join()) throw new Error("split changed the takes");
  if (L[0].id !== "W02-静止" || L.find(x => x.id === "W05-静止").cls !== "bicycle" || L.find(x => x.id === "W01-静止").bell !== "-6") throw new Error(L[0].id);
  const a = fresh(); a.begin(); a.tk();
  a.tb("start"); a.ck(a.E("btnNextKubun"));
  const card = re => [...a.D.querySelectorAll("#catGrid .cat")].find(x => re.test(x.textContent));
  if (!card(/歩く比較（自転車）.*16 本/) || !card(/歩く比較（車）.*16 本/)) throw new Error("cards");
  a.ck(card(/歩く比較（自転車）/)); a.ck(a.E("btnNewSession"));
  let s = a.st(); if (s.kubun !== "歩行対比" || s.script !== "W2_BIKE" || s.mount !== "ヘルメット" || s.id !== "20261015_W1") throw new Error(JSON.stringify([s.kubun, s.script, s.mount, s.id]));
  if (a.E("fBikeDim").hidden || !a.E("fCarDim").hidden) throw new Error("dims fields for the bike walking session");
  a.sv("inLaeq", "52.3"); a.ck(a.E("btnCheckDone")); a.sv("inHeight", "175"); a.ck(a.chipIn("#chipLevel", /真ん中/));
  a.tb("rec"); a.ck(a.E("btnWPractice")); a.ck(a.E("btnWPractice"));   // 練習の 1 回（敵対レビュー S5）
  if (!/ベル：前輪が現場の x = -6（北）の目印を越えたら 1 回（静止も歩行も同じ目印）/.test(a.E("itFacts").textContent)) throw new Error(a.E("itFacts").textContent);
  a.tk(); a.tk();
  s = a.st(); if (s.takes.map(t => t.scriptId).join() !== "W02-静止,W02-歩行") throw new Error(s.takes.map(t => t.scriptId).join());
  if (a.W.__note.issuesOf(s).some(x => /そろっていない/.test(x.text))) throw new Error("pair warned although complete");
  a.tk(); if (!a.W.__note.issuesOf(a.st()).some(x => /W04 は立ち止まりと歩きの 2 本がそろっていない/.test(x.text))) throw new Error("half pair not warned");
  a.edit(3); a.twice("btnExTake");   // 片方だけで終わった組は「この本を使わない（録り直す）」→ 次の日に組ごと
  if (a.W.__note.issuesOf(a.st()).some(x => /そろっていない/.test(x.text))) throw new Error("still warned after excluding the half pair");
  a.twice("btnCloseSession");
  a.tb("start"); a.ck(card(/歩く比較（自転車）/)); a.sv("inDate", "2026-10-22"); a.E("inFileNo").value = "1"; a.E("inVideoNo").value = "300"; a.ck(a.E("btnNewSession"));
  if (a.st().id !== "20261022_W1" || a.st().script !== "W2_BIKE") throw new Error(a.st().id);
  a.tb("rec"); if (a.E("itId").textContent !== "W04-静止") throw new Error("next on the new day: " + a.E("itId").textContent);
});
// ---------- 監査 2026-10-13（記録シートの全部の見直し・F01〜F33）----------
step("監査 F01: 10/15 より前の「本番」のセッション（10/7 の家のデモ）は C の「済み」に数えず、はじめるに知らせ、一覧の「練習に回す」で練習にできる", () => {
  const a0 = fresh(); a0.begin(); a0.tk(); a0.tk(); const demo = JSON.parse(JSON.stringify(a0.st())); demo.id = "20261007_C1"; demo.date = "2026-10-07"; demo.prefix = "261007";
  demo.takes.forEach(t => { t.file = t.file.replace("261015", "261007"); });
  const dm = boot(html, JSON.stringify({ sessions: [demo], cur: null, prep: {}, tab: "start" })); const W = dm.window, D = W.document, E = id => D.getElementById(id);
  W.addEventListener("error", e => errs.push(e.message));
  const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
  const card = [...D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent) && !/家のデモ/.test(x.textContent));
  if (!/0 \/ 16 本 済み/.test(card.textContent)) throw new Error("demo counted: " + card.textContent);
  if (E("staleBox").hidden || !/20261007_C1（2 本）/.test(E("staleBox").textContent) || !/本番の CSV/.test(E("staleBox").textContent) || /済み」に数えられる/.test(E("staleBox").textContent)) throw new Error("stale box: " + E("staleBox").textContent);   // 敵対レビュー L7
  if (!W.__note.issuesOf(JSON.parse(W.localStorage.getItem("helmet_note_v1")).sessions[0]).some(x => /より前の「本番」/.test(x.text))) throw new Error("no issue");
  ck(D.querySelector("nav button[data-tab=list]")); const yb = E("yoto_20261007_C1"); ck(yb); ck(yb);
  const st = JSON.parse(W.localStorage.getItem("helmet_note_v1")); if (st.sessions[0].yoto !== "調整用") throw new Error("not moved to practice");
  ck(D.querySelector("nav button[data-tab=start]")); if (!E("staleBox").hidden) throw new Error("stale box still shown");
});
step("変更票 #67（監査 F03 を含む）: ② は毎回 4 方向。C の 4 方向のあとの歩く比較（ヘルメット）も、次の三脚のセッションも 4 方向・切り替えのボタンは無い", () => {
  const a = fresh(); a.begin(); if (a.st().checkMode !== "4") throw new Error("first C not 4");
  a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /歩く比較（自転車）/.test(x.textContent))); a.ck(a.E("btnNewSession"));
  if (a.st().mount !== "ヘルメット" || a.st().checkMode !== "4") throw new Error("helmet session check mode " + a.st().checkMode);
  a.tb("start"); a.twice("btnCloseSession"); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /自転車（ベルなし）/.test(x.textContent))); a.E("inFileNo").value = "30"; a.E("inInherit").checked = false; a.ck(a.E("btnNewSession"));
  if (a.st().checkMode !== "4") throw new Error("tripod session should be 4 directions: " + a.st().checkMode);
  if (a.E("chk1") || a.E("chk4")) throw new Error("4 方向／前だけ の切り替えが残っている");
  const sc = a.W.__note.sessionCsv().split(/\r?\n/); const H = sc[0].split(","); if (sc.slice(1).filter(Boolean).some(l => l.split(",")[H.indexOf("点検方式")] !== "4方位")) throw new Error("点検方式 not 4方位");
});
step("監査 F09・F18・F20: デモのカードのあとに C を押すと本番。H の次は歩く比較（自転車）。測るタブは出さない", () => {
  const a = fresh(); a.tb("start");
  a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /家のデモ/.test(x.textContent))); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent) && !/家のデモ/.test(x.textContent)));
  a.sv("inDate", "2026-10-15"); a.E("inFileNo").value = "1"; a.ck(a.E("btnNewSession")); if (a.st().yoto !== "最終評価") throw new Error("C started as practice after the demo card");
  a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck(a.E("btnNewSession")); if (a.st().kubun !== "H") throw new Error("C→H " + a.st().kubun);
  a.tb("start"); a.ck(a.E("btnNextKubun")); if (!/歩く比較（自転車）/.test(a.E("btnNewSession").textContent)) throw new Error("H→ " + a.E("btnNewSession").textContent);
  if (!a.D.querySelector("nav button[data-tab=measure]").hidden) throw new Error("measure tab shown");
});
step("監査 F04・F05・F19: C では「動画を撮らない」を出さない・動画の番号が空なら知らせる。歩く比較の拍手は顔の前", () => {
  const a = fresh(); a.tb("start"); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent) && !/家のデモ/.test(x.textContent)));
  a.sv("inDate", "2026-10-15"); a.E("inFileNo").value = "1"; a.E("inVideoNo").value = ""; a.ck(a.E("btnNewSession")); a.sv("inLaeq", "52"); a.ck(a.E("btnCheckDone")); a.sv("inHeight", "160"); a.ck(a.chipIn("#chipLevel", /真ん中/));
  a.tb("rec"); if (!a.E("btnVideoToggle").hidden) throw new Error("no-video toggle shown in C"); if (a.E("noVideoNoBox").hidden) throw new Error("empty video number not shown");
  a.sv("inNextVideo", "120"); if (!a.E("noVideoNoBox").hidden) throw new Error("warning stays after the number");
  if (!/マイクの正面/.test(a.E("clapStep").textContent)) throw new Error("tripod clap text");
  a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /歩く比較（自転車）/.test(x.textContent))); a.ck(a.E("btnNewSession"));
  a.sv("inLaeq", "52"); a.ck(a.E("btnCheckDone")); a.sv("inHeight", "175"); a.ck(a.chipIn("#chipLevel", /真ん中/)); a.tb("rec");
  if (!/顔の前/.test(a.E("clapStep").textContent)) throw new Error("helmet clap text: " + a.E("clapStep").textContent);
});
step("監査 F07・F10・F33: 同じ台本の本が 2 本なら知らせる。ベルの高さは後から入れて録った本にも写る。歩く比較で組が欠けたまま閉じようとすると知らせる", () => {
  const a = fresh(); a.begin(); a.tk();
  {const x = JSON.parse(a.W.localStorage.getItem("helmet_note_v1")); const s = x.sessions.find(y => y.id === x.cur); const t2 = JSON.parse(JSON.stringify(s.takes[0])); t2.file = "261015_004.WAV"; t2.uid = "dup"; t2.take_id = 2; s.takes.push(t2); s.nextFile = 5;
   a.W.localStorage.setItem("helmet_note_v1", JSON.stringify(x));}   // 10/15 の json のような、同じ本が 2 本の古い記録（今は録り始めで前の本を外すので、画面からは作れない）
  if (!a.W.__note.issuesOf(a.st()).some(x => /同じ台本の本が 2 本（C2-13/.test(x.text))) throw new Error("duplicate not warned");
  a.tb("start"); a.sv("inBellZ2", "1.12"); if (a.st().bellZ !== "1.12" || !a.st().takes.every(t => t.sourceZ === "1.12")) throw new Error("bell height " + JSON.stringify(a.st().takes.map(t => t.sourceZ)));
  if (a.evRows()[0]["高さ_m"] !== "1.12") throw new Error("events.csv 高さ_m");
  a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /歩く比較（自転車）/.test(x.textContent))); a.ck(a.E("btnNewSession"));
  a.sv("inLaeq", "52"); a.ck(a.E("btnCheckDone")); a.sv("inHeight", "175"); a.ck(a.chipIn("#chipLevel", /真ん中/)); a.tb("rec"); a.ck(a.E("btnWPractice")); a.ck(a.E("btnWPractice")); a.tk();
  a.tb("start"); a.ck(a.E("btnCloseSession")); if (!/W02（静止）は 2 本そろっていない/.test(a.E("btnCloseSession").textContent)) throw new Error("close message: " + a.E("btnCloseSession").textContent);
});
step("監査 F06: ② の「録れた」を押し忘れて回したあとで押しても、② は 0° の点検のまま（今の向きの点検は回したときのもの）", () => {
  const a = fresh(); a.begin(false); a.tb("rec"); a.ck(a.chipIn("#itPicker", /C2-04$/)); if (a.E("rotBox").hidden) throw new Error("no rotation"); a.twice("btnRotDone");
  a.tb("start"); a.ck(a.E("btnCheckDone")); const s = a.st();
  if (s.checks[s.checkFile].h !== "0" || s.checkHeading !== "180" || s.validCheck !== s.rotCheck) throw new Error(JSON.stringify([s.checks, s.checkHeading, s.validCheck, s.rotCheck]));
});
step("監査 F14: 歩く比較の組の 2 本で相手の走る向きが違えば知らせる（ベルなし P41〜P50 は向きをその日に選ぶ）", () => {
  const base = { id: "W", kubun: "歩行対比", gain: "70", ritual: { calib: true, check: true }, checkFile: "c.WAV", checks: { "c.WAV": { h: "0" } }, ver: 54, bikeWb: "104" };
  const tk = (id, st, dir) => ({ take_id: id, file: "f" + id + ".WAV", clapA: "", tripodHeading: "0", events: [{ lap: "", cls: "bicycle", pair: "P41", stand: st, direction: dir }] });
  const bad = Object.assign({}, base, { takes: [tk(1, "静止", "前→後"), tk(2, "歩行", "後→前")] });
  if (!w.__note.issuesOf(bad).some(x => /P41 の 2 本で相手の走る向きが違う/.test(x.text))) throw new Error("not warned");
  const ok = Object.assign({}, base, { takes: [tk(1, "静止", "前→後"), tk(2, "歩行", "前→後")] });
  if (w.__note.issuesOf(ok).some(x => /向きが違う/.test(x.text))) throw new Error("false warning");
});
step("Astra V76-03・05・08: ② を録り直すと新しい原本は今の向き。測ったベルの高さは同じ日の次のセッションへ。動画を撮るに戻すボタンの文言", () => {
  const a = fresh(); a.tb("start"); a.ck([...a.D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent) && !/家のデモ/.test(x.textContent)));
  a.sv("inDate", "2026-10-15"); a.E("inFileNo").value = "1"; a.ck(a.E("btnNewSession")); a.sv("inLaeq", "52"); a.ck(a.E("btnCheckDone"));
  a.ck(a.chipIn("#chipHeading", /^180°/)); a.twice("btnCheckRedo"); a.ck(a.E("btnCheckDone"));
  const s = a.st(); if (s.checks[s.checkFile].h !== "180" || s.checkFileHeading !== "180") throw new Error("redo heading " + JSON.stringify([s.checkFile, s.checks, s.checkFileHeading]));
  a.sv("inBellZ2", "1.12"); a.sv("inCamera", "現場(-2,+12.4)・実測"); a.tb("start"); a.ck(a.E("btnNextKubun")); a.ck(a.E("btnNewSession"));
  if (a.st().kubun !== "H" || a.st().bellZ !== "1.12") throw new Error("bell height not inherited: " + a.st().bellZ);
  if (a.st().camera !== "現場(-2,+12.4)・実測") throw new Error("camera not inherited (敵対レビュー L2): " + a.st().camera);
  const st = JSON.parse(a.W.localStorage.getItem("helmet_note_v1")); st.sessions[0].noVideo = true; st.cur = st.sessions[0].id;
  const dm = boot(html, JSON.stringify(st)); const D2 = dm.window.document; D2.querySelector("nav button[data-tab=rec]").dispatchEvent(new dm.window.Event("click", { bubbles: true }));
  if (!/動画を撮る/.test(D2.getElementById("btnVideoToggle").textContent) || D2.getElementById("btnVideoToggle").hidden) throw new Error("toggle label: " + D2.getElementById("btnVideoToggle").textContent);
});
step("Astra V76-02: 今の形の写しからは、一覧（session_ids）に無いセッション（全部消す前の古い写し）を戻さない", () => {
  const meta = { layout: 2, session_ids: ["20261015_C1"], cur: "", prep: {} };
  const docs = [{ session: { id: "20261015_C1", takes: [] } }, { session: { id: "20261015_H1", takes: [] } }];
  const st = w.__note.rebuildFromServer(meta, docs); if (st.sessions.map(s => s.id).join() !== "20261015_C1") throw new Error(st.sessions.map(s => s.id).join());
  const old = w.__note.rebuildFromServer({ cur: "" }, docs); if (old.sessions.length !== 2) throw new Error("legacy meta should use all");
});
(async () => {
  await stepAsync("Astra V76-01: ほかの端末で戻しただけ・開いただけでは共有の写し sessions/<id> を書かない。その端末で本を録ったら書く", async () => {
    const a0 = fresh(); a0.begin(); a0.tk(); const real = JSON.parse(JSON.stringify(a0.st())); real.dev = "dIPAD";
    const writes = [];
    const col = p => { const docs = { notes: [{ device: "dIPAD", savedAt: "2026-10-15T10:00:00Z", n_sessions: 1, session_ids: [real.id] }], ["notes/dIPAD/sessions"]: [{ session_id: real.id, session: real }] };
      const L = (docs[p] || []).map(x => ({ data: () => JSON.parse(JSON.stringify(x)) })); const r = { empty: !L.length, docs: L }; return { orderBy: () => ({ limit: () => ({ get: async () => r }) }), get: async () => r }; };
    const fakeDb = { doc: path => ({ set: async obj => { writes.push(path); } }), collection: col };
    const W = new JSDOM("<!doctype html><html><head><meta charset='utf-8'></head><body>" + html + "</body></html>", { runScripts: "dangerously", url: "https://example.org/", pretendToBeVisual: true,
      beforeParse(win) { win.claude = { use: async name => (name === "db" ? fakeDb : { save: async () => {} }) }; } }).window;
    await new Promise(r => setTimeout(r, 2600));
    const st = JSON.parse(W.localStorage.getItem("helmet_note_v1") || '{"sessions":[]}'); if (st.sessions.length !== 1) throw new Error("not restored");
    const D = W.document; const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
    ck(D.querySelector("nav button[data-tab=list]")); ck([...D.querySelectorAll("#sessTable button")][0]);
    await new Promise(r => setTimeout(r, 1800));
    if (writes.some(p => p === "sessions/" + real.id)) throw new Error("opened on another device wrote the shared copy");
    ck(D.querySelector("nav button[data-tab=rec]")); ck(D.getElementById("btnStart")); ck(D.getElementById("btnStop")); if (!D.getElementById("running").hidden) ck(D.getElementById("btnStop"));
    await new Promise(r => setTimeout(r, 1800));
    if (!writes.some(p => p === "sessions/" + real.id)) throw new Error("recording on this device should write the shared copy");
  });
  await stepAsync("監査 F02: 自動で戻すのは 10/15 以降の本番の本がある端末が 1 台だけのとき（2 台なら選ばない・10/15 より前だけの端末は戻さない）", async () => {
    const a0 = fresh(); a0.begin(); a0.tk(); const real = JSON.parse(JSON.stringify(a0.st()));
    const old = JSON.parse(JSON.stringify(real)); old.id = "20261013_H1"; old.date = "2026-10-13";
    const open = (docs) => { const col = p => { const L = (docs[p] || []).map(x => ({ data: () => JSON.parse(JSON.stringify(x)) })); const r = { empty: !L.length, docs: L };
        return { orderBy: () => ({ limit: () => ({ get: async () => r }) }), get: async () => r }; };
      const fakeDb = { doc: () => ({ set: async () => {} }), collection: col };
      return new JSDOM("<!doctype html><html><head><meta charset='utf-8'></head><body>" + html + "</body></html>", { runScripts: "dangerously", url: "https://example.org/", pretendToBeVisual: true,
        beforeParse(win) { win.claude = { use: async name => (name === "db" ? fakeDb : { save: async () => {} }) }; } }).window; };
    const two = open({ notes: [{ device: "dPC", savedAt: "2026-10-15T11:00:00Z", n_sessions: 1 }, { device: "dIPAD", savedAt: "2026-10-15T10:00:00Z", n_sessions: 1 }],
      "notes/dPC/sessions": [{ session_id: real.id, session: real }], "notes/dIPAD/sessions": [{ session_id: real.id, session: real }] });
    const onlyOld = open({ notes: [{ device: "dPC", savedAt: "2026-10-15T11:00:00Z", n_sessions: 1 }], "notes/dPC/sessions": [{ session_id: old.id, session: old }] });
    await new Promise(r => setTimeout(r, 2600));
    const n2 = (JSON.parse(two.localStorage.getItem("helmet_note_v1") || '{"sessions":[]}').sessions || []).length; if (n2) throw new Error("restored although 2 candidates");
    const n1 = (JSON.parse(onlyOld.localStorage.getItem("helmet_note_v1") || '{"sessions":[]}').sessions || []).length; if (n1) throw new Error("restored a session before 10/15");
  });
  await stepAsync("Astra V04: すぐ次の区分を始めても、前のセッションもサーバに写る", async () => {
    const writes = [];
    const fakeDb = { doc: path => ({ set: async obj => { await new Promise(r => setTimeout(r, 20)); writes.push([path, obj]); } }),
      collection: () => ({ orderBy: () => ({ limit: () => ({ get: async () => ({ empty: true, docs: [] }) }) }), get: async () => ({ empty: true, docs: [] }) }) };
    const dm = new JSDOM("<!doctype html><html><head><meta charset='utf-8'></head><body>" + html + "</body></html>",
      { runScripts: "dangerously", url: "https://example.org/", pretendToBeVisual: true,
        beforeParse(win) { win.claude = { use: async name => (name === "db" ? fakeDb : { save: async () => {} }) }; } });
    const W = dm.window, D = W.document; const E = id => D.getElementById(id);
    const ck = el => el.dispatchEvent(new W.Event("click", { bubbles: true }));
    const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
    const tb = t => ck(D.querySelector("nav button[data-tab=" + t + "]"));
    await new Promise(r => setTimeout(r, 50));
    tb("start"); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /自転車のベル/.test(x.textContent) && !/家のデモ/.test(x.textContent)));
    sv("inDate", "2026-10-15"); E("inFileNo").value = "1"; ck(E("btnNewSession"));
    sv("inLaeq", "50"); ck(E("btnCheckDone")); sv("inHeight", "150"); ck([...D.querySelectorAll("#chipLevel .chip")][0]);
    tb("rec"); ck(E("btnStart")); pv(E, ck, "btnClap"); pv(E, ck, "btnEvent"); ck(E("btnStop")); if (!E("running").hidden) ck(E("btnStop"));
    tb("start"); ck(E("btnNextKubun")); ck([...D.querySelectorAll("#catGrid .cat")].find(x => /車の通過/.test(x.textContent))); ck(E("btnNewSession"));
    await new Promise(r => setTimeout(r, 2600));
    const c = writes.filter(w => w[0] === "sessions/20261015_C1").pop(); const a = writes.filter(w => w[0] === "sessions/20261015_A1").pop();
    if (!c || c[1].n_takes !== 1) throw new Error("C1 not synced: " + JSON.stringify(writes.map(w => [w[0], w[1].n_takes])));
    if (!a) throw new Error("A1 not synced");
  });
  await stepAsync("10/13 本人「時間たつと勝手に消える」: 端末の中が空なら、サーバでセッションのある端末から自動で戻す（空の端末は飛ばす・「全部消す」の印で止める・戻すのはセッションだけ・今のセッションは選ばない・ほかの端末の写しは書き換えない・中身があれば戻さない）", async () => {
    const a0 = fresh(); a0.begin(); a0.tk(); const sess = JSON.parse(JSON.stringify(a0.st()));   // 本物の形のセッション（20261015_C1）
    let docs = { notes: [{ device: "dEMPTY", savedAt: "2026-10-15T09:00:00Z", n_sessions: 0, prep: {} },
                         { device: "dOLD", savedAt: "2026-10-15T08:00:00Z", n_sessions: 1, session_ids: ["20261015_C1"], cur: "20261015_C1", prep: { x: true }, check4Done: true }],
                 "notes/dOLD/sessions": [{ session_id: "20261015_C1", session: sess }] };
    const open = (writes, pre) => {
      const col = p => { const L = (docs[p] || []).map(x => ({ data: () => JSON.parse(JSON.stringify(x)) })); const r = { empty: !L.length, docs: L };
        return { orderBy: () => ({ limit: () => ({ get: async () => r }) }), get: async () => r }; };
      const fakeDb = { doc: path => ({ set: async obj => { writes.push([path, obj]); } }), collection: col };
      return new JSDOM("<!doctype html><html><head><meta charset='utf-8'></head><body>" + html + "</body></html>",
        { runScripts: "dangerously", url: "https://example.org/", pretendToBeVisual: true,
          beforeParse(win) { if (pre) win.localStorage.setItem("helmet_note_v1", JSON.stringify(pre)); win.claude = { use: async name => (name === "db" ? fakeDb : { save: async () => {} }) }; } }).window;
    };
    const w1 = [], W1 = open(w1); await new Promise(r => setTimeout(r, 2600));
    const st = JSON.parse(W1.localStorage.getItem("helmet_note_v1"));
    if (!st || st.sessions.length !== 1 || st.sessions[0].id !== "20261015_C1" || st.sessions[0].takes.length !== 1 || st.cur !== null) throw new Error("not restored: " + JSON.stringify(st && [st.sessions.length, st.cur]));
    if (Object.keys(st.prep || {}).length || st.check4Done) throw new Error("prep / check4Done restored (they are per day)");
    if (w1.some(x => /^notes\/dOLD/.test(x[0]))) throw new Error("wrote to the other device");
    const dev = W1.localStorage.getItem("helmet_note_device");
    if (!w1.some(x => x[0] === "notes/" + dev + "/sessions/20261015_C1")) throw new Error("restored copy not written under this device: " + JSON.stringify(w1.map(x => x[0])));
    const keep = { sessions: [Object.assign({}, sess, { id: "20261016_H1", kubun: "H" })], cur: null, prep: {}, tab: "start", check4Done: false };
    const w2 = [], W2 = open(w2, keep); await new Promise(r => setTimeout(r, 2600));
    const st2 = JSON.parse(W2.localStorage.getItem("helmet_note_v1"));
    if (st2.sessions.length !== 1 || st2.sessions[0].id !== "20261016_H1") throw new Error("restored over a non-empty device: " + JSON.stringify(st2.sessions.map(s => s.id)));
    docs = Object.assign({}, docs, { notes: [{ device: "dWIPE", savedAt: "2026-10-15T10:00:00Z", n_sessions: 0, prep: {}, wipedAt: "2026-10-15T10:00:00Z" }, ...docs.notes] });
    const w3 = [], W3 = open(w3); await new Promise(r => setTimeout(r, 2600));
    const st3 = JSON.parse(W3.localStorage.getItem("helmet_note_v1") || '{"sessions":[]}');
    if (st3.sessions.length) throw new Error("restored past a wipe: " + JSON.stringify(st3.sessions.map(s => s.id)));
    docs = { notes: [{ device: "dDEMO", savedAt: "2026-10-07T08:37:02.712Z", n_sessions: 1, session_ids: ["20261015_C1"], cur: "", prep: {} }], "notes/dDEMO/sessions": [{ session_id: "20261015_C1", session: sess }] };
    const w5 = [], W5 = open(w5); await new Promise(r => setTimeout(r, 2600));
    const st5 = JSON.parse(W5.localStorage.getItem("helmet_note_v1") || '{"sessions":[]}');
    if (st5.sessions.length) throw new Error("restored the 10/7 demo (saved before RESTORE_SINCE)");
    const a1 = fresh(); a1.begin(); a1.tb("save"); a1.ck(a1.E("btnWipe")); a1.ck(a1.E("btnWipe"));
    const st4 = JSON.parse(a1.W.localStorage.getItem("helmet_note_v1")); if (st4.sessions.length || !st4.wipedAt) throw new Error("wipe mark missing");
  });
  if (errs.length) { console.log("window errors:", errs); process.exitCode = 1; }
  console.log(process.exitCode ? "NG" : "ALL OK");
  process.exit(process.exitCode || 0);   // 録音中のまま残した画面の時計（setInterval）で終わらなくならないように
})();
