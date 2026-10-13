# Contact sheets

`node tools/sheet.js [outDir]` renders every photo in `tools/images/` through the current
app (`index.html`) and the new engine (`engine.js`), one PNG row per photo.

- `tools/images/<name>.png|jpg` — test photo (not committed: some are other people's work)
- `tools/images/mask-<name>.png` — optional subject cut-out (white = subject)
- `tools/images/ref-<name>-<caption>.png` — optional artist reference shown at the end of the row

Needs Playwright with Chromium.
