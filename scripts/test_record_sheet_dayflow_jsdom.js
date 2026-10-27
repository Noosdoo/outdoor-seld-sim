// 記録シートの 1 日の通し試験（jsdom）: 10/22（自転車の日）と 10/29（車の日）を、画面のボタンだけで最後まで録る（2026-10-18・Opus）
// 名前の無い架空の 10/15 の記録（test_record_sheet_v19_jsdom.js の fix1015）から始める。止まる所・画面のエラー・本数・カード・E2 の順を確かめる。
// 使い方（jsdom は 30.1.1・Astra CAR24-10）: NODE_PATH=<jsdom の node_modules> node scripts/test_record_sheet_dayflow_jsdom.js <収録記録シート_v19.html> [書き出し先フォルダ]
// （書き出し先を渡すと、その日の session.csv・events.csv を書く。夜の道具 step19d --video-only に通す確かめに使える）
const fs = require("fs"), path = require("path");
const { JSDOM } = require("jsdom");
const html = fs.readFileSync(process.argv[2], "utf-8");
const outDir = process.argv[3] || null;
const testSrc = fs.readFileSync(path.join(__dirname, "test_record_sheet_v19_jsdom.js"), "utf-8");
eval(testSrc.slice(testSrc.indexOf("function fix1015(asis) {"), testSrc.indexOf("\nlet dom = boot(")));   // 試験と同じ 10/15 の架空の記録

let failed = false;
const check = (ok, msg) => { console.log((ok ? "ok   " : "FAIL ") + msg); if (!ok) failed = true; };

function rig() {
  const dom = new JSDOM("<!doctype html><html><head><meta charset='utf-8'></head><body>" + html + "</body></html>",
    { runScripts: "dangerously", url: "https://example.org/", pretendToBeVisual: true,
      beforeParse(w) { w.localStorage.setItem("helmet_note_v1", JSON.stringify(fix1015())); } });
  const W = dom.window, D = W.document, E = id => D.getElementById(id), errs = [];
  W.addEventListener("error", e => errs.push(e.message));
  const ck = el => { if (!el) throw new Error("no element"); el.dispatchEvent(new W.Event("click", { bubbles: true })); };
  const sv = (id, v) => { E(id).value = v; E(id).dispatchEvent(new W.Event("change", { bubbles: true })); };
  const tb = t => ck(D.querySelector("nav button[data-tab=" + t + "]"));
  const st = () => { const x = JSON.parse(W.localStorage.getItem("helmet_note_v1")); return x.sessions.find(y => y.id === x.cur); };
  const chip = (sel, re) => [...D.querySelectorAll(sel + " .chip")].find(x => re.test(x.textContent));
  const card = re => [...D.querySelectorAll("#catGrid .cat")].find(x => re.test(x.textContent) && !/家のデモ/.test(x.textContent));
  const vis = id => { const el = E(id); if (!el || el.disabled) return false; for (let x = el; x; x = x.parentElement) if (x.hidden) return false; return true; };   // Astra R105-05: 親が隠れていても押せない
  return { W, D, E, errs, ck, sv, tb, st, chip, card, vis };
}

function runKubun(r, first, mount, date) {
  r.tb("start");
  if (first) { r.ck(r.card(first)); r.sv("inDate", date); r.E("inFileNo").value = "1"; r.E("inVideoNo").value = "100"; r.ck(r.E("btnNewSession")); }
  else { r.ck(r.E("btnNextKubun")); r.ck(r.E("btnNewSession")); }
  const s0 = r.st();
  const entry = { heading: s0.heading, calib: !!(s0.ritual && s0.ritual.calib), check: !!(s0.ritual && s0.ritual.check), validCheck: !!s0.validCheck };   // Astra CAR24-05: 区分に入ったときの状態
  if (!s0.laeq) r.sv("inLaeq", "52");
  if (!s0.ritual || !s0.ritual.check) r.ck(r.E("btnCheckDone"));
  if (!r.st().height) r.sv("inHeight", mount === "ヘルメット" ? "175" : "160");
  if (!r.st().level) r.ck(r.chip("#chipLevel", /真ん中/));
  if (r.vis("inBellOff") && !r.E("fBellOff").hidden && !r.st().bellOff) r.sv("inBellOff", "25");
  if (!r.E("carSetBox").hidden) r.ck(r.E("btnCarSet"));
  r.tb("rec");
  if (/^W2/.test(r.st().script || "")) { r.ck(r.E("btnWPractice")); r.ck(r.E("btnWPractice")); }
  const seq = [], blocked = []; let rot = 0;
  for (let i = 0; i < 120; i++) {
    const id = r.E("itId").textContent;
    if (id === "全部済み") break;
    if (!r.E("rotBox").hidden) { r.ck(r.E("btnRotDone")); r.ck(r.E("btnRotDone")); rot++; }
    const n0 = r.st().takes.length;
    r.ck(r.E("btnStart"));
    if (!r.st().take) { r.ck(r.E("btnStart")); if (!r.st().take) { blocked.push(id); break; } blocked.push(id + "（1 回止まった）"); }
    if (r.vis("btnClap")) r.ck(r.E("btnClap")); if (r.vis("btnEvent")) r.ck(r.E("btnEvent"));
    r.ck(r.E("btnStop")); if (r.st().take) r.ck(r.E("btnStop"));
    if (r.st().takes.length !== n0 + 1) { blocked.push(id + "（本が増えない）"); break; }
    seq.push(id);
  }
  return { script: r.st().script, seq, rot, blocked, entry, ritualLeft: r.E("ritualWarn").hidden ? "" : r.E("ritualWarn").textContent };
}

function cardsOf(r) {
  r.tb("start"); if (r.st()) { r.ck(r.E("btnCloseSession")); r.ck(r.E("btnCloseSession")); }
  const m = {}; [...r.D.querySelectorAll("#catGrid .cat")].forEach(e => { m[(e.querySelector(".nm") || {}).textContent.replace(/\s+/g, " ").trim()] = (e.querySelector(".pr") || {}).textContent; }); return m;
}

function day(name, date, plan, wantCards, r) {
  r = r || rig(); const logs = plan.map((p, i) => runKubun(r, i === 0 ? p.card : null, p.mount, date));
  plan.forEach((p, i) => {
    const l = logs[i];
    check(l.script === p.script && l.seq.length === p.n && !l.blocked.length && !l.ritualLeft,
      name + " " + p.script + ": " + l.seq.length + " 本（予定 " + p.n + "）・向きを変えた " + l.rot + " 回" + (l.blocked.length ? "・止まった " + l.blocked.join(" ") : "") + (l.ritualLeft ? "・" + l.ritualLeft : ""));
    if (p.order) check(l.seq.join(",") === p.order, name + " " + p.script + " の順: " + l.seq.join(" "));
    if (p.entry) check(Object.entries(p.entry).every(([k, v]) => l.entry[k] === v), name + " " + p.script + " に入ったとき: " + JSON.stringify(l.entry) + "（予定 " + JSON.stringify(p.entry) + "）");
  });
  const c = cardsOf(r);
  for (const [k, v] of Object.entries(wantCards)) check(Object.keys(c).some(n => n.startsWith(k) && c[n] === v), name + " のカード「" + k + "」= " + v + "（" + (Object.entries(c).find(([n]) => n.startsWith(k)) || ["", "無い"])[1] + "）");
  check(!r.errs.length, name + " の画面のエラー: " + (r.errs.join(" | ") || "なし"));
  if (outDir) { fs.writeFileSync(path.join(outDir, name + "_events.csv"), r.W.__note.eventsCsv()); fs.writeFileSync(path.join(outDir, name + "_session.csv"), r.W.__note.sessionCsv()); }
  return r;
}

day("d1022", "2026-10-22", [
  { card: /自転車のベル/, mount: "三脚", script: "C2", n: 12 },
  { mount: "三脚", script: "H2", n: 36 },
  { mount: "ヘルメット", script: "W2_BIKE", n: 16 },
], { "自転車のベル": "16 / 16 本 済み", "自転車（ベルなし）": "60 / 60 本 済み", "歩く比較（自転車）": "16 / 16 本 済み" });
// 本人 10/24 16:33: 車の日は E2 → B2 → A2 → 歩く比較（車）（Astra CAR24-05 で通し試験を新しい順に）
const CAR_PLAN = () => [
  { card: /クラクション/, mount: "三脚", script: "E2", n: 8, order: "E2-03,E2-02,E2-01,E2-04,E2-08,E2-05,E2-06,E2-07", entry: { heading: "0", calib: false, check: false } },   // 本人 10/25: 三脚は y = +2.0・向きは 0 → 90 → 180 → 270 → 270 → 180 → 90 → 0°
  { mount: "三脚", script: "E3", n: 4, order: "E3-01,E3-03,E3-04,E3-02", entry: { heading: "0", calib: true, check: false } },   // 本人 10/27 案 A: 走りながらのクラクション。三脚を原点へ戻したので点検をやり直す（校正は E2 と同じゲインで引き継ぐ）・E2 の最後の 0° のまま
  { mount: "三脚", script: "B2", n: 4, order: "B2-04,B2-02,B2-01,B2-03", entry: { heading: "180", calib: true, check: true, validCheck: true } },   // E3 の最後（右・180°）のまま始まる・同じ原点・同じ向きなので点検も引き継ぐ
  { mount: "三脚", script: "A2", n: 32, entry: { heading: "0", calib: true, check: true, validCheck: true } },   // B2 の最後（左・0°）のまま（10/27 に E3 を足して B2 が 180° から始まり 0° で終わるようになった）
  { mount: "ヘルメット", script: "W2_CAR", n: 16, entry: { calib: false, check: false } },   // 付け替えたら校正と点検をやり直す（引き継がない）
];
day("d1029", "2026-10-29", [
  ...CAR_PLAN(),
], { "車の通過": "32 / 32 本 済み", "発進": "4 / 4 本 済み", "クラクション": "8 / 8 本 済み", "走りながらのクラクション": "4 / 4 本 済み", "歩く比較（車）": "16 / 16 本 済み" });
// Astra R105-05: 10/22 を録り終えた同じ端末で、そのまま 10/29 を録る（10/15 の架空の記録 → 10/22 → 10/29）
{ const r = day("d1022c", "2026-10-22", [
    { card: /自転車のベル/, mount: "三脚", script: "C2", n: 12 }, { mount: "三脚", script: "H2", n: 36 }, { mount: "ヘルメット", script: "W2_BIKE", n: 16 },
  ], {});
  day("d1029c", "2026-10-29", CAR_PLAN(), { "自転車のベル": "16 / 16 本 済み", "自転車（ベルなし）": "60 / 60 本 済み", "歩く比較（自転車）": "16 / 16 本 済み", "車の通過": "32 / 32 本 済み", "発進": "4 / 4 本 済み", "クラクション": "8 / 8 本 済み", "走りながらのクラクション": "4 / 4 本 済み", "歩く比較（車）": "16 / 16 本 済み" }, r); }
console.log(failed ? "NG" : "ALL OK");
process.exitCode = failed ? 1 : 0;
