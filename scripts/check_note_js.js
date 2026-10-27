// html の <script> ブロックを取り出して構文検査（node --check 相当を new Function で）
const fs = require('fs');
const path = process.argv[2];
const html = fs.readFileSync(path, 'utf8');
const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
let m, n = 0, ok = true;
while ((m = re.exec(html)) !== null) {
  n++;
  const src = m[1];
  const isModule = /type\s*=\s*["']module["']/i.test(m[0]);
  try {
    if (isModule) {
      // module 構文（import/await）は vm の SourceTextModule が要るので、await を関数内に包んで検査
      new Function('(async()=>{' + src.replace(/^\s*import[^\n]*\n/gm, '') + '\n})');
    } else {
      new Function(src);
    }
    console.log('script #' + n + (isModule ? ' (module)' : '') + ': OK (' + src.length + ' chars)');
  } catch (e) {
    ok = false;
    console.log('script #' + n + ': SYNTAX ERROR ' + e.message);
  }
}
// id の整合: onchange/value を触っている id が HTML に存在するか
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(x => x[1]));
const used = new Set([...html.matchAll(/\$\('([^']+)'\)/g)].map(x => x[1]));
const missing = [...used].filter(i => !ids.has(i));
console.log('ids used in JS but missing in HTML:', missing.length ? missing.join(', ') : 'none');
process.exit(ok && !missing.length ? 0 : 1);
