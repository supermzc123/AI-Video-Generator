import assert from 'node:assert/strict';
import test from 'node:test';
import { build } from 'esbuild';
const load = async (file) => {
  const result = await build({ entryPoints: [file], bundle: true, write: false, platform: 'node', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
};
const status = await load('src/task-status.ts');
const { TaskCommandClient, describeCommandReceipt } = await load('src/task-commands.ts');
const task = (overrides = {}) => ({ task_id: 'image-1', project_id: 'p1', kind: 'image_generation', state: 'failed', available_actions: ['retry'], attempt: 1, max_attempts: 3, depends_on: [], ...overrides });
const batch = (overrides = {}) => ({ batch_id: 'b1', items: [{ project_id: 'p1', task_ids: ['image-1'] }], ...overrides });

test('all recovery states have distinct labels; absent server action metadata grants no permission', () => {
  assert.equal(new Set(['recovering', 'retry_wait', 'cancelling', 'needs_attention'].map((state) => status.taskStateLabels[state])).size, 4);
  assert.equal(status.canTaskAction(task({ available_actions: undefined }), 'retry'), false);
  assert.equal(status.canTaskAction(task({ available_actions: [] }), 'retry'), false);
});
test('project, batch, kind and state filters intersect instead of broadening control scope', () => {
  const tasks = [task(), task({ task_id: 'other-project', project_id: 'p2' }), task({ task_id: 'other-batch' }), task({ task_id: 'video', kind: 'h3_generation' })];
  assert.deepEqual(status.filterTasks(tasks, { projectId: 'p1', batch: batch(), kind: 'image_generation', state: 'failed' }).map((item) => item.task_id), ['image-1']);
});
test('blocked task identifies its actual dependency, task type, and failure state', () => {
  const explanation = status.taskExplanation(task({ task_id: 'encode', kind: 'conditioning_encoding', state: 'blocked', depends_on: ['image-1'] }), [task()]);
  assert.match(explanation, /图片生成 image-1（失败）/);
});
test('dynamic orchestration and missing members never expose a precise percentage', () => {
  assert.equal(status.batchProgress(batch(), [task({ kind: 'llm_planning', state: 'blocked' })]).percent, null);
  assert.equal(status.batchProgress(batch(), []).percent, null);
  assert.equal(status.batchProgress(batch(), [task()]).percent, 100);
  assert.notEqual(status.batchProgress(batch({ all_tasks_ended: true, all_tasks_succeeded: false }), [task()]).label, '全部成功');
});
test('batch stages report registered work and scoped orchestration phase', () => {
  const members = [
    task({ task_id: 'plan', kind: 'llm_planning', state: 'running', current_phase: 'batch_operation_started:image:portrait' }),
    task({ task_id: 'image-1', state: 'succeeded' }),
    task({ task_id: 'image-2', state: 'running' }),
  ];
  const result = status.batchMemberProgress(members.map((item) => item.task_id), members);
  assert.equal(result.stages[0].succeeded, 1);
  assert.equal(result.stages[0].registered, 2);
  assert.equal(result.stages[0].totalKnown, false);
  assert.match(status.taskPhaseLabel(members[0].current_phase), /portrait/);
  assert.equal(status.canTaskAction(task({ available_actions: ['restart'] }), 'restart'), true);
});
test('unknown external execution cannot be retried without explicit duplicate-risk confirmation', async () => {
  let calls = 0;
  const client = new TaskCommandClient(async () => { calls++; });
  await assert.rejects(client.run({ scope: 'project', scope_id: 'p1' }, 'confirm_retry', [task({ available_actions: ['confirm_retry'] })]), /重复执行风险/);
  assert.equal(calls, 0);
});
test('client rejects cross-project and unavailable actions before HTTP', async () => {
  let calls = 0;
  const client = new TaskCommandClient(async () => { calls++; });
  await assert.rejects(client.run({ scope: 'project', scope_id: 'p2' }, 'retry', [task()]), /不属于/);
  await assert.rejects(client.run({ scope: 'project', scope_id: 'p1' }, 'cancel', [task()]), /不允许/);
  assert.equal(calls, 0);
});
test('lost response and interrupted receipt retain idempotency key; completed receipt allows a new command', async () => {
  const payloads = []; let serial = 0;
  const client = new TaskCommandClient(async (payload) => {
    payloads.push(payload);
    if (payloads.length === 1) throw new Error('connection lost after acceptance');
    return { ...payload, status: payloads.length === 2 ? 'pending' : 'completed', results: [] };
  }, () => `key-${++serial}`);
  const scope = { scope: 'project', scope_id: 'p1' };
  await assert.rejects(client.run(scope, 'retry', [task()]));
  await client.run(scope, 'retry', [task()]); await client.run(scope, 'retry', [task()]); await client.run(scope, 'retry', [task()]);
  assert.deepEqual(payloads.map((payload) => payload.idempotency_key), ['key-1', 'key-1', 'key-1', 'key-2']);
});
test('cancelling and partial errors do not produce a successful cancellation claim', () => {
  const cancelling = describeCommandReceipt({ status: 'completed', results: [{ task_id: 'a', ok: true, state: 'cancelling' }] });
  assert.match(cancelling, /正在取消/); assert.doesNotMatch(cancelling, /已取消/);
  assert.match(describeCommandReceipt({ status: 'completed', results: [{ task_id: 'a', ok: false, error: 'lease still active' }] }), /1 项未完成.*lease still active/);
  assert.match(describeCommandReceipt({ status: 'needs_attention', results: [] }), /尚未完整/);
});

test('server pause and recovery reasons override generic queued explanation', () => {
  assert.equal(status.taskExplanation(task({ state: 'queued', blocked_reason: '批次成员已暂停' })), '批次成员已暂停');
  assert.equal(status.taskPhaseLabel('batch_image_child'), '等待图片素材就绪');
});

const { withRequestTimeout } = await load('src/request-timeout.ts');
test('control deadline rejects even a transport ignoring AbortSignal and aborts transport', async () => {
  let signal;
  await assert.rejects(withRequestTimeout((value) => { signal = value; return new Promise(() => {}); }, 10), /请求超时/);
  assert.equal(signal.aborted, true);
});
test('successful control requests do not abort after their timer was cleared', async () => {
  let signal;
  assert.equal(await withRequestTimeout(async (value) => { signal = value; return 42; }, 10), 42);
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(signal.aborted, false);
});
