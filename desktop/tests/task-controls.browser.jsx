import React from 'react';
import { createRoot } from 'react-dom/client';
import { TasksView } from '../src/TasksView';
import { BatchView } from '../src/BatchView';
import { newProject } from '../src/project-store';
import '../src/styles.css';

const base = { schema_version: '1.0', idempotency_key: 'test', input_fingerprint: 'test', depends_on: [], execution_target: 'local', worker_id: null, affinity_key: null, priority: 0, attempt: 1, max_attempts: 3, lease_expires_at: null, comfyui_prompt_id: null, error_code: null, error_message: null, updated_at: '2026-09-24T12:00:00Z' };
const tasks = [
  { ...base, task_id: 'p1-image', project_id: 'p1', kind: 'image_generation', state: 'running', available_actions: ['cancel'] },
  { ...base, task_id: 'p1-failed', project_id: 'p1', kind: 'h3_generation', state: 'failed', available_actions: ['retry'], error_message: '模拟网络错误\n诊断详情' },
  { ...base, task_id: 'p1-unknown', project_id: 'p1', kind: 'image_generation', state: 'needs_attention', available_actions: ['reconcile', 'confirm_retry', 'cancel'] },
  { ...base, task_id: 'p1-parent', project_id: 'p1', kind: 'llm_planning', state: 'blocked', available_actions: ['cancel'], depends_on: ['p1-image'] },
  { ...base, task_id: 'p1-outside-batch', project_id: 'p1', kind: 'export', state: 'failed', available_actions: ['retry'] },
  { ...base, task_id: 'p2-image', project_id: 'p2', kind: 'image_generation', state: 'running', available_actions: ['cancel'] },
];
const batch = { schema_version: '1.0', batch_id: 'b1', name: '模拟批次', state: 'running', items: [
  { project_id: 'p1', task_ids: tasks.filter(t => t.project_id === 'p1' && t.task_id !== 'p1-outside-batch').map(t => t.task_id), start_boundary: 'next_ready', priority: 0, paused: false },
  { project_id: 'p2', task_ids: ['p2-image'], start_boundary: 'next_ready', priority: 0, paused: false },
], created_at: '2026-09-24T12:00:00Z', updated_at: '2026-09-24T12:00:00Z', all_tasks_ended: false, all_tasks_succeeded: false };
const project = { ...newProject(), projectId: 'p1', name: '项目一' };
const projects = [{ project_id: 'p1', name: '项目一' }, { project_id: 'p2', name: '项目二' }];
const root = createRoot(document.getElementById('root'));
const fixture = window.fixture = { offline: false, confirmation: true, requests: [], confirmations: [], tasks, batch, batches: [batch], failAuxiliary: false, hangReadsAfterCommand: false, hangReads: false, releases: [], startFailures: 0, loseCreateResponse: false, paused: {}, mount(view) {
  root.render(view === 'batch' ? <BatchView project={project} projects={projects} /> : <TasksView project={project} projects={projects} />);
} };
window.confirm = message => { fixture.confirmations.push(message); return fixture.confirmation; };
window.fetch = async (input, init = {}) => {
  const url = typeof input === 'string' ? input : input.url;
  if (fixture.offline) throw new Error('mock disconnected');
  const path = new URL(url, location.origin).pathname;
  const method = init.method ?? 'GET';
  if (method === 'GET' && fixture.hangReads) await new Promise(resolve => fixture.releases.push(resolve));
  if (fixture.failAuxiliary && (path === '/api/v1/batches' || path === '/api/v1/scheduler/health' || path.endsWith('/run-state'))) return new Response(JSON.stringify({ detail: 'mock auxiliary failure' }), { status: 503 });
  const body = init.body ? JSON.parse(init.body) : undefined;
  const request = { path, method, body };
  if (method !== 'GET') fixture.requests.push(request);
  let value;
  if (path === '/api/v1/tasks') value = tasks;
  else if (path === '/api/v1/batches' && method === 'GET') value = fixture.batches;
  else if (path === '/api/v1/batches' && method === 'POST') {
    if (fixture.batches.some(item => item.batch_id === body.batch_id)) return new Response('{}', { status: 409 });
    fixture.batches.push(body); value = body;
    if (fixture.loseCreateResponse) { fixture.loseCreateResponse = false; throw new Error('mock response lost after create'); }
  } else if (path.startsWith('/api/v1/batches/') && path.endsWith('/start')) {
    if (fixture.startFailures > 0) { fixture.startFailures--; return new Response(JSON.stringify({ detail: 'mock start failed' }), { status: 409 }); }
    value = fixture.batches.find(item => item.batch_id === path.split('/')[4]); value.state = 'running';
  }
  else if (path === '/api/v1/scheduler/health') value = { owns_dispatcher: true, task_counts: {} };
  else if (path.endsWith('/run-state')) value = { project_id: path.split('/')[4], paused: fixture.paused[path.split('/')[4]] ?? false, outline_approved: true };
  else if (path === '/api/v1/task-commands') {
    const results = body.task_ids.map(id => {
      const task = tasks.find(t => t.task_id === id);
      if (body.action === 'cancel') { task.state = 'cancelling'; task.available_actions = ['reconcile']; }
      if (body.action === 'retry' || body.action === 'confirm_retry') { task.state = 'queued'; task.available_actions = ['cancel']; }
      return { task_id: id, ok: true, state: task.state };
    });
    value = { ...body, results, status: 'completed' };
    if (fixture.hangReadsAfterCommand) fixture.hangReads = true;
  } else if (path.startsWith('/api/v1/projects/') && path.endsWith('/pause')) {
    fixture.paused[path.split('/')[4]] = body.paused; value = { paused: body.paused };
  } else if (/\/batches\/b1\/projects\/p[12]\/(pause|resume)$/.test(path)) {
    const item = batch.items.find(item => item.project_id === path.split('/')[6]);
    item.paused = path.endsWith('/pause'); value = batch;
  } else if (path.endsWith('/events')) value = [{ event_id: 1, task_id: path.split('/')[4], kind: 'recovery', created_at: 1790000000, payload: { evidence: 'mock' } }];
  else throw new Error(`Unexpected fixture request: ${method} ${path}`);
  return new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
};
fixture.mount('tasks');
