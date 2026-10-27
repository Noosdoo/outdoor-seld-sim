// #44: generated browser state only; no network, no actual recording data.
// NODE_PATH=<directory containing jsdom> node scripts/test_helmet_orientation_jsdom.js
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const {JSDOM, VirtualConsole} = require('jsdom');
const html = fs.readFileSync(path.join(__dirname, '../md/design/実録_記入用CSV_2026-09-05/ヘルメット収録ノート.html'), 'utf8');
const doms = [], errors = [];
function boot(storage) {
  const vc = new VirtualConsole(); vc.on('jsdomError', e => errors.push(e.message));
  const dom = new JSDOM(html, {url:'https://example.org', runScripts:'dangerously', virtualConsole:vc,
    beforeParse(w) {if(storage) w.localStorage.setItem('helmet_note_v1', storage);}});
  doms.push(dom); return dom.window;
}
let n = 0;
function check(label, fn) {fn(); n++; console.log('PASS '+label);}
try {
  const w = boot(), d = w.document, $ = id => d.getElementById(id);
  const click = el => el.dispatchEvent(new w.Event('click', {bubbles:true}));
  const set = (id,v) => {$(id).value=v; $(id).dispatchEvent(new w.Event('change',{bubbles:true}));};
  const tab = id => click(d.querySelector('nav button[data-tab="'+id+'"]'));
  tab('session'); click($('btnNewSession'));
  set('inLaeq','52'); set('inHeight','160'); click($('btnCheckDone'));
  click([...d.querySelectorAll('#chipLevel .chip')].find(e=>e.textContent==='y'));
  tab('take'); click($('btnRec')); click($('btnLap'));
  set('inLapSec','5');
  check('向きの既定は 0°',()=>assert.equal($('inTripodHeading').value,'0'));
  set('inTripodHeading','180'); set('inTakeCheck','TURN_CHECK.wav');
  click($('btnLap')); set('inLapSec','7');
  // Restore a closed test take through the same backup API used on other devices.
  const backup = JSON.parse(w.__note.backupJson());
  const session = backup.state.sessions[0];
  session.take.running=false; session.takes.push(session.take); session.take=null;
  w.__note.importBackup(JSON.stringify(backup));
  const csv = w.__note.eventsCsv(), lines = csv.trim().split('\n'), headers = lines[0].split(',');
  check('既存29列の順序を保ち、新項目を末尾に追加',()=>assert.deepEqual(headers.slice(26,29),['拍手_録音側秒','回転後点検原本','三脚の向き']));
  check('同じテイクの全イベントに向き・点検原本',()=>{
    assert.equal(lines.length,3); for(const line of lines.slice(1)) assert.match(line,/,TURN_CHECK.wav,180,/);
  });
  const w2=boot(w.localStorage.getItem('helmet_note_v1'));
  check('再起動で欄とCSVを保持',()=>assert.equal(w2.__note.eventsCsv(),csv));
  const w3=boot(); w3.__note.importBackup(w.__note.backupJson());
  check('バックアップ復元でCSV一致',()=>assert.equal(w3.__note.eventsCsv(),csv));
  const legacy=JSON.parse(w.__note.backupJson());
  const t=legacy.state.sessions[0].takes[0]; delete t.tripodHeading; delete t.takeCheck;
  t.events[0].note='三脚180°'; w.__note.importBackup(JSON.stringify(legacy));
  check('旧チップが一行だけでも全行180°',()=>{
    for(const line of w.__note.eventsCsv().trim().split('\n').slice(1)) assert.match(line,/,180,/);
  });
  tab('take'); set('inTripodHeading','0');
  check('明示0°が古い特記より優先',()=>{
    for(const line of w.__note.eventsCsv().trim().split('\n').slice(1)) assert.match(line,/,0,/);
  });
  set('inTripodHeading','180'); set('inTakeCheck','TURN_CHECK.wav'); tab('out');
  check('再点検のコマンドと音を戻さない案内',()=>{
    assert.match($('outCmd').textContent,/--in raw\/TURN_CHECK.wav/);
    assert.match($('outCmd').textContent,/三脚 180° を --yaw に足さない/);
    assert.doesNotMatch($('outCmd').textContent,/--yaw\s+180/);
  });
  // 2026-10-01: 向きと回転後点検原本は前のテイクから引き継ぐ（選び忘れ防止）。向きを変えたら点検原本は空に戻る
  const w4=boot(), d4=w4.document, $4=id=>d4.getElementById(id);
  const click4=el=>el.dispatchEvent(new w4.Event('click',{bubbles:true}));
  const set4=(id,v)=>{$4(id).value=v; $4(id).dispatchEvent(new w4.Event('change',{bubbles:true}));};
  const tab4=id=>click4(d4.querySelector('nav button[data-tab="'+id+'"]'));
  tab4('session'); click4($4('btnNewSession'));
  set4('inLaeq','52'); set4('inHeight','160'); click4($4('btnCheckDone'));
  click4([...d4.querySelectorAll('#chipLevel .chip')].find(e=>e.textContent==='y'));
  tab4('take'); click4($4('btnRec'));
  set4('inTripodHeading','180'); set4('inTakeCheck','TURN_CHECK.wav');
  click4($4('btnLap')); set4('inLapSec','5');
  click4($4('btnStop')); click4($4('btnStop'));   // 記入漏れの注意は 2 回押しで閉じる
  check('次のテイクは向きと点検原本を引き継ぐ',()=>{
    click4($4('btnRec'));
    assert.equal($4('inTripodHeading').value,'180'); assert.equal($4('inTakeCheck').value,'TURN_CHECK.wav');
  });
  check('向きを変えると点検原本は空に戻る',()=>{
    set4('inTripodHeading','0'); assert.equal($4('inTripodHeading').value,'0'); assert.equal($4('inTakeCheck').value,'');
  });
  check('何も来なかったテイクも向きを引き継ぐ',()=>{
    click4($4('btnAbort')); click4($4('abNotRec')); click4($4('btnNoneTake'));
    const ts=JSON.parse(w4.__note.backupJson()).state.sessions[0].takes;
    assert.equal(ts.length,2); assert.equal(ts[1].tripodHeading,'180'); assert.equal(ts[1].takeCheck,'TURN_CHECK.wav');
  });
  check('装着と高さの既定はセッションから',()=>{
    tab('take'); click($('btnRec'));
    assert.equal($('inTakeMount').value,'ヘルメット'); assert.equal($('inTakeHeight').value,'160');
  });
  check('装着・高さ・固定音源座標をCSV末尾へ',()=>{
    set('inTakeMount','三脚'); set('inTakeHeight','175');
    set('inSourceX','0'); set('inSourceY','3'); set('inSourceZ','0.8');
    click($('btnLap')); set('inLapSec','8');
    click([...d.querySelectorAll('#chipClass .chip')].find(e=>e.textContent.includes('クラクション')));
    assert.match($('classEquipmentGuide').textContent,/エンジン/);
    set('inEquipmentNote','車は横向き エンジンOFF 写真001');
    click($('btnStop')); click($('btnStop'));
    const lines=w.__note.eventsCsv().trim().split('\n'); const values=lines.at(-1).split(',');const h=lines[0].split(',');
    for(const [key,value] of Object.entries({'装着':'三脚','マイク高さ_cm':'175','x_m':'0','y_m':'3','高さ_m':'0.8'}))assert.equal(values[h.indexOf(key)],value);
  });
  check('次のテイクは装着・高さを継承し固定位置は空に',()=>{
    click($('btnRec'));assert.equal($('inTakeHeight').value,'175');assert.equal($('inTakeMount').value,'三脚');assert.equal($('inSourceY').value,'');
  });
  check('P50・通過方向・線・ベルを180°の現場座標で出力',()=>{
    click($('btnLap'));set('inLapSec','8');set('inTripodHeading','180');
    click([...d.querySelectorAll('#chipPair .chip')].find(e=>e.textContent==='P50'));
    click([...d.querySelectorAll('#chipDirection .chip')].find(e=>e.textContent==='前→後'));
    click([...d.querySelectorAll('#chipRunSpeed .chip')].find(e=>e.textContent==='歩行速'));
    click([...d.querySelectorAll('#chipBellMark .chip')].find(e=>e.textContent==='-10'));
    set('inLineM','3'); click($('btnStop'));click($('btnStop'));
    const lines=w.__note.eventsCsv().trim().split('\n'),h=lines[0].split(','),v=lines.at(-1).split(',');
    for(const [key,value] of Object.entries({pair_id:'P50',direction:'後→前',line_m:'3','ベル目印_m':'10','走行速さ':'歩行速'}))assert.equal(v[h.indexOf(key)],value);
  });
  check('負例露出100分はラップ上限内、通常本は600秒',()=>{
    const fixture=JSON.parse(w.__note.backupJson());const se=fixture.state.sessions[0];
    se.take={take_id:99,file:'TEST.wav',running:true,tripodHeading:'0',events:[{event_id:1,cls:'horn',lap:6000,quad:'L',stand:'静止',note:''}]};
    for(const [kind,stops] of [['負例露出',true],['C',false]]){
      se.kubun=kind;w.__note.importBackup(JSON.stringify(fixture));tab('take');click($('btnStop'));
      const now=JSON.parse(w.__note.backupJson()).state.sessions[0];assert.equal(now.take===null,stops);
    }
  });
  check('新しい欄も再起動・バックアップで保持',()=>{
    const restored=boot(w.localStorage.getItem('helmet_note_v1')); assert.equal(restored.__note.eventsCsv(),w.__note.eventsCsv());
  });
  check('日付式の原本名と手動接頭辞、日付変更を保持',()=>{
    const nw=boot(),nd=nw.document,$n=id=>nd.getElementById(id);
    const clickn=e=>e.dispatchEvent(new nw.Event('click',{bubbles:true}));
    const setn=(id,v)=>{$n(id).value=v;$n(id).dispatchEvent(new nw.Event('change',{bubbles:true}));};
    clickn(nd.querySelector('nav button[data-tab="session"]'));
    setn('inDate','2026-10-10');assert.equal($n('inPrefix').value,'261010');
    setn('inDate','2026-10-11');assert.equal($n('inPrefix').value,'261011');
    setn('inPrefix','CUSTOM');setn('inDate','2026-10-12');assert.equal($n('inPrefix').value,'CUSTOM');
    setn('inPrefix','');setn('inDate','2026-10-10');clickn($n('btnNewSession'));
    const ns=JSON.parse(nw.__note.backupJson()).state.sessions[0];
    assert.equal(ns.calibFile,'261010_001.WAV');assert.equal(ns.checkFile,'261010_002.WAV');
    const old=JSON.parse(nw.__note.backupJson());old.state.sessions[0].calibFile='ZOOM0001.wav';
    nw.__note.importBackup(JSON.stringify(old));assert.match(nw.__note.sessionCsv(),/ZOOM0001.wav/);
  });
  check('起動・編集時のJavaScript例外なし',()=>assert.deepEqual(errors,[]));
  console.log(n+'/'+n+' ALL PASS');
} finally {for(const dom of doms) dom.window.close();}
