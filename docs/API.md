# API 契约

独立开发控制平面默认地址为`http://127.0.0.1:8000`。Tauri桌面端每次启动选择空闲回环端口，前端通过桌面命令读取实际Origin；OpenAPI位于`/docs`。

## 核心端点

```text
GET  /api/v1/health
GET  /api/v1/setup/status
GET  /api/v1/postprocessing/capabilities
GET  /api/v1/projects/{project_id}/execution-preflight
GET  /api/v1/workers/local/capabilities
POST /api/v1/chains/compile-shot
POST /api/v1/plans/dry-run

POST /api/v1/projects
GET  /api/v1/projects
GET  /api/v1/projects/{project_id}
POST /api/v1/projects/{project_id}/revisions
GET  /api/v1/projects/{project_id}/revisions
GET  /api/v1/projects/{project_id}/revisions/{revision}

POST /api/v1/workflows/inspect
POST /api/v1/workflows/templates
GET  /api/v1/workflows/templates/{template_id}/{revision}
POST /api/v1/workflows/compile
POST /api/v1/llm/workflows/map

POST /api/v1/h3/workflows/inspect
POST /api/v1/h3/workflows/compile
POST /api/v1/h3/workflows/profiles
GET  /api/v1/h3/workflows/profiles?profile_id={profile_id}
GET  /api/v1/h3/workflows/profiles/{profile_id}/{revision}

POST /api/v1/tasks
GET  /api/v1/tasks
GET  /api/v1/tasks/{task_id}
POST /api/v1/tasks/{task_id}/cancel
POST /api/v1/tasks/{task_id}/review/accept
POST /api/v1/tasks/{task_id}/review/reject
POST /api/v1/exports/plan
```

`/health`返回固定`service_id`、构建版本和本次启动nonce。桌面端必须同时校验三者，不能只依据HTTP 200识别控制平面。

项目设置按`project_id + revision`不可变保存。`GET /projects`和`GET /projects/{project_id}`返回最新修订，历史通过`/revisions`读取；重复提交相同内容是幂等操作，试图覆盖同一修订返回`409`。

审核接口只接受`needs_review`任务。接受后任务进入`succeeded`；拒绝后进入`failed`并以`review_rejected`保存可选的`feedback`（最多4000字符）。同一审核决定可安全重试，反向决定返回`409`。

`/workflows/inspect`只接受ComfyUI API格式JSON。UI格式的`nodes`/`links`图会被拒绝。LLM返回的binding草案不能直接登记，必须先通过节点、字段、类型和范围校验，再由用户把模板状态确认为`approved`。

H3工作流作为受控资源随软件版本发布，GUI不提供节点图导入或编辑。`/h3/workflows/inspect`和`/h3/workflows/compile`保留为开发、迁移与兼容性验证接口，不是面向最终用户的工作流准备步骤。实际项目任务使用打包图，并按GUI设置类型化替换模型、LoRA和受支持的运行参数；调度前仍会对目标Worker重新验证节点Schema和模型清单。

`POST /h3/workflows/profiles`仅用于开发人员登记不可变受控图修订，不在产品GUI中开放。Profile必须显式设置`approval: "approved"`，登记时服务端始终从当前ComfyUI读取`/object_info`复验。工作流JSON、节点Schema或官方Turbo来源变化时必须递增修订；旧修订永久保留用于审计。

Turbo仅接受锁定官方插件中的`MiniMaxH3TurboLoRA`和`MiniMaxH3TurboSampler`，并要求它们与`BasicScheduler(simple)`接入同一个`SamplerCustomAdvanced`采样链。完整约束见[H3受控工作流](H3_WORKFLOWS.md)。

## 远程Worker

远程端点只有配置`AIVIDEO_WORKER_AUTH_TOKEN`后才启用，并要求`Authorization: Bearer <token>`：

```text
POST /api/v1/workers/register
POST /api/v1/workers/{worker_id}/leases/claim
POST /api/v1/workers/{worker_id}/leases/{task_id}/renew
POST /api/v1/workers/{worker_id}/leases/{task_id}/result
WSS  /api/v1/workers/{worker_id}/events
PUT  /api/v1/artifacts/blobs/{sha256}
GET  /api/v1/artifacts/blobs/{sha256}
POST /api/v1/workload-manifests
GET  /api/v1/workload-manifests/{sha256}
```

生产部署应由反向代理提供HTTPS/WSS。ComfyUI本身不能暴露到公网。Token由桌面应用存入Windows Credential Manager，Linux Worker存入系统Keyring；项目数据库和Git仓库不得保存明文凭据。

外部任务采用“租约 + 至少一次提交 + 指纹去重 + ComfyUI history对账”，不宣称严格exactly-once。
Worker应仅建立出站HTTPS/WSS连接。结果使用稳定`report_id`提交；控制平面持久化收据，重复提交相同报告会返回同一收据，不会重复改变任务状态。产物必须先按SHA-256上传并校验，再提交包含其哈希的结果。HTTP认领在响应丢失后会返回该Worker仍持有的活动租约，避免重试时领取第二个任务。

Ubuntu客户端基础实现在`ai_video_generator.workers.remote_worker`。生产环境默认拒绝明文HTTP/WS；`allow_insecure_http`只供回环开发与测试使用。安装`worker` extra后可使用WSS事件轮询。
部署、重试和当前执行manifest缺口见[Ubuntu远程Worker基础](REMOTE_WORKER.md)。
# 运行编排接口

除既有工作流、H3、任务租约和导出接口外，控制平面提供以下持久化编排接口：

- POST/GET /api/v1/projects/{project_id}/workspace：保存和读取完整 UI 工作区不可变修订。
- GET /api/v1/projects/{project_id}/run-state：读取执行模式、暂停、审核和预算状态。
- POST /api/v1/projects/{project_id}/mode：请求 guided / batch 切换；运行中任务结束后生效。
- POST /api/v1/projects/{project_id}/pause：停止后续任务派发，不抢占原子任务。
- GET /api/v1/projects/{project_id}/preflight：批量启动前的确定性阻塞项和警告。
- POST /api/v1/projects/{project_id}/agent/operate：项目智能体受限 Patch 预览或提交。
- POST/GET /api/v1/projects/{project_id}/memory：完整事件归档与 FTS5 检索。
- POST/GET /api/v1/projects/{project_id}/decisions：项目决定和锁定字段账本。
- POST /api/v1/tasks/{task_id}/run|pause|resume|redo：精细化任务控制。
- POST/GET /api/v1/batches 与 POST /api/v1/batches/{id}/{action}：批量工作台。
- POST /api/v1/reviews/apply-timeouts：应用到期人工审核并切换项目级 AI 接管。
- POST /api/v1/performance/samples|estimate：保存性能基线并估算后续任务。
- POST/GET /api/v1/model-residency：记录模型加载、卸载、复用与缓存命中。

## Harness

- GET/POST /api/v1/harnesses：列出或登记 Harness Bundle。
- POST /api/v1/harnesses/{id}/validate：确定性校验 Markdown、Schema、哈希和工作流绑定。
- POST/GET /api/v1/harnesses/{id}/revisions：不可变 Harness 修订。
- POST /api/v1/harnesses/{id}/revisions/{revision}/test：只运行契约预检，不启动 GPU。
- POST /api/v1/harnesses/h3-default/sources/{community|official}/install：运行时下载固定提交。
- POST /api/v1/harnesses/h3-default/assemble?approve=false：从本机固定源组装 H3 Harness。

官方 MiniMax 源没有明确许可证，因此安装器只写入用户数据目录，不随 GPL 安装包分发。
