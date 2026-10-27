// Synthetic records only. No network or existing browser storage is used.
// NODE_PATH=<jsdom node_modules> node scripts/test_home_demo_note.js <new output directory>
const fs=require('fs'), path=require('path'), assert=require('assert/strict');
const {JSDOM,VirtualConsole}=require('jsdom');
const out=path.resolve(process.argv[2]);fs.mkdirSync(out,{recursive:true});
const html=fs.readFileSync(path.join(__dirname,'../md/design/実録_記入用CSV_2026-09-05/ヘルメット収録ノート.html'),'utf8');
const errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e.message));
const dom=new JSDOM(html,{url:'https://example.org',runScripts:'dangerously',virtualConsole:vc});
const w=dom.window,d=w.document,$=id=>d.getElementById(id);
const click=e=>{assert.ok(e,'UI element exists');e.dispatchEvent(new w.Event('click',{bubbles:true}));};
const set=(id,v)=>{$(id).value=v;$(id).dispatchEvent(new w.Event('change',{bubbles:true}));};
const chip=(id,label)=>click([...$(id).querySelectorAll('button')].find(e=>e.textContent===label));
const tab=id=>click(d.querySelector('nav button[data-tab="'+id+'"]'));
try {
  tab('session');set('inDate','2026-10-04');chip('chipYoto','調整用');
  click($('btnNewSession'));set('inLaeq','52');set('inHeight','160');
  click($('btnCheckDone'));chip('chipLevel','y');tab('take');
  for(let i=0;i<4;i++){
    if(i===2)set('inNextFile','6'); // 005 is the second four-direction check.
    click($('btnRec'));set('inTakeMount','三脚');set('inTakeHeight','160');
    set('inTripodHeading',i>=2?'180':'0');
    if(i>=2)set('inTakeCheck','261004_005.WAV');
    if(i<3){set('inVideo','synthetic.mp4');set('inClapA','3');set('inClapF','3');set('inClapV','3');}
    else {set('inSourceX','0');set('inSourceY','2');set('inSourceZ','1.6');}
    click($('btnLap'));chip('chipClass',i===3?'クラクション':'車');
    chip('chipStand','静止');chip('chipQuad',i>=2?'右 R':'左 L');set('inLapSec',i===3?'8':'10');
    if(i<3){set('inDist','2');set('inLineM','2');chip('chipDirection',i===0?'後→前':'前→後');}
    set('inEvNote',i===3?'合成デモ・声のプーを想定（評価に使わない）':'合成デモ（評価に使わない）');
    click($('btnStop'));click($('btnStop'));
  }
  click($('btnNoneTake'));tab('out');
  const commands=$('outCmd').textContent;
  assert.match(commands,/raw_demo\//);assert.match(commands,/ann_orig_tuning\.csv/);assert.doesNotMatch(commands,/out\/realsmoke\//);
  assert.match(commands,/--mode negative --in raw_demo\/261004_008.WAV.*--append/);
  const backup=JSON.parse(w.__note.backupJson()),s=backup.state.sessions[0];
  assert.equal(s.yoto,'調整用');assert.equal(s.takes.length,5);assert.equal(s.calibFile,'261004_001.WAV');
  assert.deepEqual(Array.from(s.takes,t=>t.file),['261004_003.WAV','261004_004.WAV','261004_006.WAV','261004_007.WAV','261004_008.WAV']);
  for(const [name,text] of Object.entries({'session.csv':w.__note.sessionCsv(),'events.csv':w.__note.eventsCsv(),'commands.ps1':$('outCmd').textContent,'backup.json':JSON.stringify(backup,null,2)}))fs.writeFileSync(path.join(out,name),text);
  assert.deepEqual(errors,[]);
  console.log('PASS jsdom: tuning session, 5 takes, date filenames, rotation check, fixed coordinates, none; exported '+out);
} finally {dom.window.close();}
