import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const executable = process.env.EDGE_PATH || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if (!existsSync(executable)) throw new Error('Set EDGE_PATH to a local Chromium/Edge executable. Browser test was not run.');
const built = await build({ entryPoints: ['tests/task-controls.browser.jsx'], bundle: true, write: false, outfile: 'fixture.js', jsx: 'automatic', define: { 'process.env.NODE_ENV': '"development"' } });
const javascript = built.outputFiles.find(file => file.path.endsWith('.js')).text;
const css = built.outputFiles.find(file => file.path.endsWith('.css')).text;
const server = createServer((req, res) => {
  if (req.url === '/fixture.js') { res.setHeader('Content-Type', 'text/javascript'); res.end(javascript); }
  else if (req.url === '/fixture.css') { res.setHeader('Content-Type', 'text/css'); res.end(css); }
  else if (req.url === '/') { res.setHeader('Content-Type', 'text/html; charset=utf-8'); res.end('<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="/fixture.css"></head><body><div id="root"></div><script src="/fixture.js"></script></body></html>'); }
  else { res.writeHead(404); res.end('Unknown mock route'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const directory = mkdtempSync(path.join(tmpdir(), 'aivideo-ui-test-'));
const browser = spawn(executable, ['--headless=new', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', `--user-data-dir=${directory}`, '--window-size=1500,1100', 'about:blank'], { windowsHide: true, stdio: 'ignore' });
let socket;
const checks = [];
async function waitUntil(check, message, limit = 10000) {
  const deadline = Date.now() + limit;
  while (Date.now() < deadline) { if (await check()) return; await delay(50); }
  throw new Error(`Timed out: ${message}`);
}
try {
  await waitUntil(() => existsSync(path.join(directory, 'DevToolsActivePort')), 'browser startup', 20000);
  const port = readFileSync(path.join(directory, 'DevToolsActivePort'), 'utf8').split('\n')[0];
  const targets = await fetch(`http://127.0.0.1:${port}/json/list`).then(response => response.json());
  socket = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }); });
  let serial = 0; const pending = new Map();
  socket.addEventListener('message', ({ data }) => {
    const item = JSON.parse(data); const callback = pending.get(item.id);
    if (callback) { pending.delete(item.id); item.error ? callback.reject(new Error(JSON.stringify(item.error))) : callback.resolve(item.result); }
  });
  function cdp(method, params = {}) {
    return new Promise((resolve, reject) => { const id = ++serial; pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })); });
  }
  async function evaluate(expression) {
    const result = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  const click = text => evaluate(`(() => { const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === ${JSON.stringify(text)}); if (!b || b.disabled) throw new Error('Button unavailable: '+${JSON.stringify(text)}); b.click(); })()`);
  const clickAction = (id, label) => evaluate(`(() => { const b = document.querySelector(${JSON.stringify(`[aria-label="${label} ${id}"]`)}); if (!b || b.disabled) throw new Error('Action unavailable'); b.click(); })()`);
  const select = (label, value) => evaluate(`(() => { const s = document.querySelector(${JSON.stringify(`[aria-label="${label}"]`)}); s.value = ${JSON.stringify(value)}; s.dispatchEvent(new Event('change',{bubbles:true})); })()`);
  const body = () => evaluate('document.body.innerText');
  const refresh = () => evaluate(`document.querySelector('[title="刷新任务状态"]').click()`);
  await cdp('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/` });
  await waitUntil(async () => (await body()).includes('p1-image'), 'initial task state');
  assert.equal(await evaluate('document.querySelectorAll("[data-task-id]").length'), 5);
  assert.equal(await evaluate(`Boolean(document.querySelector('[data-task-id="p2-image"]'))`), false);
  checks.push('default project scope excludes other projects');
  await click('暂停项目派发'); await waitUntil(async () => (await body()).includes('恢复项目派发'), 'project paused');
  await click('恢复项目派发'); await waitUntil(async () => (await body()).includes('暂停项目派发'), 'project resumed');
  assert.deepEqual(await evaluate('fixture.requests.filter(r => r.path.endsWith("/pause")).map(r => [r.path, r.body.paused])'), [['/api/v1/projects/p1/pause', true], ['/api/v1/projects/p1/pause', false]]);
  checks.push('pause/resume use explicitly selected project');
  await select('筛选任务类型', 'image_generation'); await delay(100);
  assert.equal(await evaluate('document.querySelectorAll("[data-task-id]").length'), 2);
  await select('筛选任务状态', 'needs_attention'); await delay(100);
  assert.equal(await evaluate('document.querySelectorAll("[data-task-id]").length'), 1);
  await clickAction('p1-unknown', '重新对账'); await waitUntil(async () => await evaluate('fixture.requests.some(r => r.body?.action === "reconcile")'), 'reconcile request');
  assert.equal(await evaluate('fixture.requests.find(r => r.body?.action === "reconcile").body.confirm_duplicate_execution'), false);
  await evaluate('fixture.confirmation = false'); await delay(100); await clickAction('p1-unknown', '确认重新执行'); await delay(100);
  assert.equal(await evaluate('fixture.requests.filter(r => r.body?.action === "confirm_retry").length'), 0);
  await evaluate('fixture.confirmation = true'); await clickAction('p1-unknown', '确认重新执行');
  await waitUntil(async () => await evaluate('fixture.requests.some(r => r.body?.action === "confirm_retry")'), 'confirmed duplicate execution');
  assert.equal(await evaluate('fixture.requests.find(r => r.body?.action === "confirm_retry").body.confirm_duplicate_execution'), true);
  checks.push('filters intersect; reconcile and duplicate-risk confirmation stay distinct');
  await select('筛选任务类型', ''); await select('筛选任务状态', ''); await delay(100);
  await clickAction('p1-image', '取消任务'); await waitUntil(async () => (await body()).includes('正在取消'), 'cancellation waiting state');
  assert.equal(await evaluate('fixture.tasks.find(t=>t.task_id==="p2-image").state'), 'running');
  assert.match(await body(), /等待执行端确认/);
  await select('筛选批次', 'b1'); await delay(100);
  assert.equal(await evaluate('document.querySelectorAll("[data-task-id]").length'), 4);
  await click('暂停批次成员派发'); await waitUntil(async () => (await body()).includes('恢复批次成员派发'), 'filtered member pause');
  assert.equal(await evaluate('fixture.batch.items[1].paused'), false);
  await click('恢复批次成员派发'); await waitUntil(async () => (await body()).includes('暂停批次成员派发'), 'filtered member resume');
  await click('重试筛选内失败项 (1)'); await waitUntil(async () => await evaluate('fixture.tasks.find(t=>t.task_id==="p1-failed").state === "queued"'), 'failed retry');
  assert.equal(await evaluate('fixture.tasks.find(t=>t.task_id==="p1-outside-batch").state'), 'failed');
  assert.equal(await evaluate('fixture.requests.find(r=>r.body?.action==="retry").body.scope'), 'batch');
  checks.push('cancel waits for confirmation; filtered batch retry excludes other batch/project tasks');
  await select('筛选项目', 'p2'); await waitUntil(async () => await evaluate(`Boolean(document.querySelector('[data-task-id="p2-image"]'))`), 'project switch');
  assert.equal(await evaluate('document.querySelectorAll("[data-task-id]").length'), 1);
  await evaluate('fixture.offline = true'); await refresh(); await waitUntil(async () => (await body()).includes('连接中断'), 'disconnected banner');
  assert.equal(await evaluate('document.querySelectorAll("[data-task-id]").length'), 1);
  assert.equal(await evaluate(`document.querySelector('[aria-label="取消任务 p2-image"]').disabled`), true);
  await evaluate('fixture.offline = false'); await refresh(); await waitUntil(async () => !(await body()).includes('连接中断'), 'reconnected snapshot');
  checks.push('disconnect retains snapshot and disables controls; reconnect synchronizes');
  // A healthy task endpoint must stay actionable when all auxiliary endpoints fail.
  await evaluate('fixture.failAuxiliary = true'); await refresh();
  await waitUntil(async () => (await body()).includes('任务控制仍可使用'), 'auxiliary failure banner');
  assert.equal(await evaluate(`document.querySelector('[aria-label="取消任务 p2-image"]').disabled`), false);
  await select('筛选项目', 'p1'); await waitUntil(async () => await evaluate(`Boolean(document.querySelector('[aria-label="重试失败 p1-outside-batch"]'))`), 'failed task with unavailable auxiliary services');
  await clickAction('p1-outside-batch', '重试失败');
  await waitUntil(async () => await evaluate('fixture.tasks.find(t=>t.task_id==="p1-outside-batch").state === "queued"'), 'retry despite auxiliary failure');
  await select('筛选项目', 'p2'); await waitUntil(async () => await evaluate(`Boolean(document.querySelector('[aria-label="取消任务 p2-image"]'))`), 'return to cancellation task');
  await evaluate('fixture.hangReadsAfterCommand = true');
  await clickAction('p2-image', '取消任务');
  await waitUntil(async () => await evaluate('fixture.requests.some(r=>r.body?.action==="cancel" && r.body.task_ids.includes("p2-image"))'), 'cancel despite auxiliary failure');
  await waitUntil(async () => await evaluate(`!document.querySelector('[aria-label="取消任务 p2-image"]').disabled`), 'busy released while refresh hangs', 1000);
  await evaluate('fixture.hangReadsAfterCommand = false; fixture.hangReads = false; fixture.failAuxiliary = false; fixture.releases.splice(0).forEach(resolve=>resolve())');
  await refresh();
  checks.push('auxiliary failures preserve controls; command lock releases before hung refresh finishes');
  await evaluate('fixture.mount("batch")'); await waitUntil(async () => (await body()).includes('编排中，任务总数尚未确定'), 'dynamic batch');
  assert.equal(await evaluate('document.querySelectorAll(".batch-progress").length'), 0);
  await evaluate(`document.querySelector('.batch-member .task-control-buttons button').click()`);
  await waitUntil(async () => await evaluate('fixture.batch.items[0].paused'), 'member pause');
  assert.equal(await evaluate('fixture.batch.items[1].paused'), false);
  await evaluate(`document.querySelector('.batch-member .task-control-buttons button').click()`);
  await waitUntil(async () => await evaluate('!fixture.batch.items[0].paused'), 'member resume');
  checks.push('dynamic batches hide percentage; member pause/resume cannot affect another member');
  await evaluate(`fixture.batches = []; fixture.startFailures = 1; document.querySelector('[title="刷新批量状态"]').click()`);
  await waitUntil(async () => await evaluate('!document.querySelector(".batch-project-choice input").disabled'), 'project can be selected');
  await evaluate('document.querySelector(".batch-project-choice input").click()');
  await click('启动 1 个项目');
  await waitUntil(async () => (await body()).includes('草稿已保留'), 'failed start preserves draft');
  assert.equal(await evaluate('fixture.requests.filter(r=>r.path==="/api/v1/batches" && r.method==="POST").length'), 1);
  assert.equal(await evaluate('fixture.batches.length'), 1);
  assert.equal(await evaluate('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="启动草稿")'), true);
  await click('继续启动“1 个项目批次”');
  await waitUntil(async () => await evaluate('fixture.batches[0].state === "running"'), 'same draft started');
  assert.equal(await evaluate('fixture.requests.filter(r=>r.path==="/api/v1/batches" && r.method==="POST").length'), 1);
  const firstDraftId = await evaluate('fixture.batches[0].batch_id');
  assert.equal(await evaluate('fixture.requests.filter(r=>r.path.endsWith("/start")).every(r=>r.path.includes(fixture.batches[0].batch_id))'), true);
  // Drafts surviving a page reload expose their own start action.
  await evaluate('fixture.batches[0].state = "draft"; fixture.mount("tasks")'); await delay(50);
  await evaluate('fixture.mount("batch")'); await waitUntil(async () => (await body()).includes('启动草稿'), 'persisted draft start entry');
  await click('启动草稿'); await waitUntil(async () => await evaluate('fixture.batches[0].state === "running"'), 'persisted draft starts');
  assert.equal(await evaluate('fixture.batches[0].batch_id'), firstDraftId);
  // Lost create response must reconcile by the stable ID instead of creating again.
  await evaluate(`fixture.batches = []; fixture.loseCreateResponse = true; document.querySelector('[title="刷新批量状态"]').click()`);
  await waitUntil(async () => await evaluate('!document.querySelector(".batch-project-choice input").disabled'), 'second selection ready');
  await evaluate('document.querySelector(".batch-project-choice input").click()'); await click('启动 1 个项目');
  await waitUntil(async () => (await body()).includes('未确认创建'), 'lost create response');
  await click('继续启动“1 个项目批次”');
  await waitUntil(async () => await evaluate('fixture.batches[0].state === "running"'), 'lost response reconciled');
  assert.equal(await evaluate('fixture.requests.filter(r=>r.path==="/api/v1/batches" && r.method==="POST").length'), 2);
  checks.push('failed starts and lost creation responses reuse draft identity; persisted drafts can be started');
  if (process.env.TASK_UI_SCREENSHOT) {
    if (process.env.TASK_UI_SCREENSHOT_VIEW === 'tasks') { await evaluate('fixture.mount("tasks")'); await waitUntil(async () => (await body()).includes('p1-image'), 'task screenshot'); }
    const screenshot = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    writeFileSync(path.resolve(process.env.TASK_UI_SCREENSHOT), Buffer.from(screenshot.data, 'base64'));
  }
  console.log(JSON.stringify({ browser: 'local Edge headless', checks: checks.length, passed: checks, production_requests: 0 }, null, 2));
  await cdp('Browser.close').catch(() => {});
} finally {
  socket?.close(); browser.kill(); server.close();
  await delay(250);
  try { rmSync(directory, { recursive: true, force: true }); } catch { /* Browser shutdown can briefly retain profile locks. */ }
}
