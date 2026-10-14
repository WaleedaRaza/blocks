// Contact sheets: every test photo through the current app and the new engine.
//   node tools/sheet.js [outDir]
// Test photos live in tools/images (not committed). Writes one PNG per photo.
const path = require('path'), fs = require('fs');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const imgDir = path.join(__dirname, 'images');
const outDir = path.resolve(process.argv[2] || path.join(__dirname, 'out'));
const url = p => 'file://' + p;
const dataUrl = p => 'data:image/' + (/\.png$/i.test(p) ? 'png' : 'jpeg') + ';base64,' + fs.readFileSync(p).toString('base64');

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const files = fs.readdirSync(imgDir).filter(f => /\.(png|jpe?g)$/i.test(f) && !f.startsWith('ref-') && !f.startsWith('mask-')).sort();
  const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' }).catch(() => chromium.launch());
  const page = await browser.newPage({ viewport: { width: 3000, height: 900 } });
  const errors = []; page.on('pageerror', e => errors.push(e.message));

  // 1. what the app (index.html) shows at step 2, without a subject mask, for comparison
  const old = {};
  for (const f of files) {
    await page.goto(url(path.join(root, 'index.html')));
    await page.setInputFiles('#file', path.join(imgDir, f));
    await page.waitForSelector('.grid canvas');
    old[f] = await page.$$eval('.grid figure canvas', cs => cs[2].toDataURL('image/png'));
  }

  // 2. new engine
  const jobs = files.map(f => {
    const base = f.replace(/\.\w+$/, '');
    const extra = fs.readdirSync(imgDir).filter(x => x.startsWith('ref-' + base + '-')).sort()
      .map(x => ({ src: dataUrl(path.join(imgDir, x)), caption: 'Artist: ' + x.replace(/^ref-[^-]+-|\.\w+$/g, '').replace(/_/g, ' ') }));
    const mp = path.join(imgDir, 'mask-' + base + '.png');
    return { name: base, src: dataUrl(path.join(imgDir, f)), mask: fs.existsSync(mp) ? dataUrl(mp) : null, old: old[f], extra };
  });
  await page.addInitScript(j => { window.JOBS = j; }, jobs);
  await page.goto(url(path.join(__dirname, 'sheet.html')));
  await page.waitForFunction(() => document.body.dataset.done || document.body.dataset.error, null, { timeout: 180000 });
  const err = await page.evaluate(() => document.body.dataset.error);
  if (err) { console.error(err); process.exit(1); }
  const rows = await page.$$('.row');
  for (let i = 0; i < rows.length; i++) {
    const name = jobs[i].name;
    await rows[i].screenshot({ path: path.join(outDir, name + '.png') });
    console.log(await rows[i].$eval('h2', h => h.textContent));
  }
  if (errors.length) console.error('page errors:', errors);
  await browser.close();
})();
