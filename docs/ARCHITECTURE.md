# 系统架构

## 1. 总体结构

首版采用独立应用控制 ComfyUI，不把产品状态塞进 ComfyUI 工作流，也不让前端直接编排节点。

```text
Tauri 2 + React Desktop UI
     | HTTP + WebSocket
Python Control Plane
     |-- Project / Revision Service
     |-- LLM Harness + Validators
     |-- DAG Compiler + Scheduler
     |-- Artifact Store + Review Service
     |-- FFmpeg Exporter
     |
H3 Worker Adapter
     |
versioned controlled H3 workflows + validated ComfyUI nodes
```

首版采用 Python 3.11/3.12、FastAPI、Pydantic和SQLite WAL。媒体文件保存在项目目录，SQLite只保存索引、状态和血缘；使用`httpx`与WebSocket对接Worker。桌面端采用Tauri 2 + React/TypeScript/Vite，关闭窗口后驻留托盘。暂不引入Redis、Celery、Temporal或Kubernetes。

## 2. 核心领域模型

- `Project`：项目设置、活动版本和公共素材。
- `OutlineRevision`：大纲的不可变修订。
- `ShotRevision`：电影语义上的镜头，可以超过 15 秒。
- `AssetRevision`：角色、场景、道具或风格素材的不可变版本。
- `GenerationSegmentRevision`：H3 执行单元，时长必须 `> 0 && <= 15s`。
- `ContinuationChainRevision`：片段顺序、Motion Context 边和显式 reset 边界。
- `PromptRevision`：由结构化事实编译出的模型输入，不是事实源。
- `Review`：对大纲、分镜、片段计划或产物的审核记录。
- `GenerationRun`：一次生成执行及其模型、工作流、seed 和输入输出。
- `Artifact`：图片、视频、音频、预览或 Motion Context 文件及其 SHA-256。

已批准修订不得原地修改。变更产生新修订，下游旧结果标记为 `STALE`，但不立即删除。

```text
OutlineRevision
  -> ShotRevision
  -> GenerationSegmentRevision
  -> PromptRevision
  -> GenerationRun / Artifact
  -> Motion Context descendants
  -> Final composition
```

依赖必须显式保存。修改角色素材只影响绑定该素材版本的片段；修改片段 prompt 或 seed 会使该片段及其 Motion Context 后继失效，直到 reset；不相关镜头不得重跑。

## 3. LLM 边界

LLM 可以起草大纲、电影分镜、素材需求、长镜头语义拆分点、H3 提示词和局部修复建议。

确定性程序负责 JSON Schema、引用完整性、并发版本、锁定字段、15 秒硬限制、帧数计算、链路无环、缓存失效、任务提交、预算和人工审核身份。

```text
LLM 草稿
-> JSON Schema 校验
-> 领域与连续性校验
-> 最多 2 至 3 次 LLM 局部修复
-> 再次确定性校验
-> 人工审核
-> 提交不可变修订
```

LLM 只建议叙事拆分点，不直接计算 H3 帧数、overlap 或 trim。提示词必须保存模板版本和所有来源修订 ID。

## 4. 长镜头编译

电影镜头超过 15 秒时，编译器生成多个 `GenerationSegmentRevision`：

1. LLM 建议动作缓冲、遮挡、转身、运镜减速等自然拆分点。
2. 确定性校验器计算 `sample_frames`、`visible_frames`、context 和 trim。
3. 第一段使用镜头入口参考；后继片段绑定公共素材和前一段 Motion Context 产物。
4. 连续链允许显式 reset。reset 后以新关键帧开始，不再依赖更早的 Motion Context。
5. 重生成中间片段时，只有依赖它的后继需要重新生成。

`sample_frames` 与 `visible_frames` 必须分开。继承头可能占用 H3 的 15 秒预算，所以每段不一定净增 15 秒，具体算法由 P0 实测冻结。公共参考与 Motion Context 是两类独立依赖。

## 5. H3 Worker 边界

H3工作流与本项目的静态conditioning缓存、Motion Context和任务产物契约高度耦合，因此产品随版本发布经过验证的受控节点图，不要求用户自行准备。图形拓扑不在GUI中开放；用户只能从Worker能力清单选择扩散模型、文本编码器、视频/音频VAE、Turbo LoRA及受支持的运行参数。每个受控图、节点Schema、模型选择和插件版本都进入任务指纹。

产品数据库仍是项目、素材、镜头和修订的唯一真相源。任何工作流内置素材库、提示词增强器、弹窗时间线或基于`node_id`的缓存身份都不作为产品接口。Motion Context可以由经验证的外部节点提供，但同一Profile必须只有一套明确的继承路径和patch owner。

Worker 的`custom_nodes`必须报告来源、提交和节点Schema。Turbo只接受锁定官方`ComfyUI-MiniMax-H3-Turbo`的LoRA加载器与sampler；注意力切换只接受 ComfyUI 原生 `ModelAttentionBackend` 的 `comfy kitchen attention`；TeaCache在标准和Turbo Profile中都被拒绝。发现重复、未知或来源不符的实现时拒绝接任务。

主应用只依赖稳定 Adapter API：

```text
GET  /v1/health
GET  /v1/capabilities
POST /v1/chains/validate
POST /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/events
POST /v1/jobs/{job_id}/cancel
POST /v1/jobs/{job_id}/retry
GET  /v1/jobs/{job_id}/manifest
POST /v1/assets/materialize
```

`POST /v1/jobs` 接受幂等键。事件携带 `shot_id`、`segment_id`、stage 和 progress，而不是只暴露 ComfyUI node ID。

## 6. 条件预编码与模型驻留

条件编码是首版核心性能路径，而不是后续优化。对已批准且不可变的生成批次，Worker 按以下阶段执行：

```text
冻结批次计划
-> 加载文本/多模态条件编码器
-> 批量编码所有可提前计算的 segment conditioning
-> 校验并持久化编码产物
-> 卸载编码器并释放显存
-> 加载 H3 扩散模型
-> 按依赖链生成片段
```

文本 prompt、公共参考和片段局部参考只要不依赖前一段动态输出，就应在扩散阶段前统一编码。Motion Context/AV latent 依赖前一段结果，不能假装预计算，但它不妨碍提前编码整条链的静态条件。

每个 `ConditioningArtifact` 必须是正式的内容寻址产物。缓存指纹至少包含：

```text
normalized prompt
+ all reference asset SHA-256 values and binding order
+ resolution, fps, sample frames and generation mode
+ text/image encoder model hashes
+ H3 model, VAE, LoRA and relevant node versions
+ conditioning schema version
```

prompt、素材、尺寸、时长或编码栈变化时，只把受影响的 conditioning 及其下游 generation 标为 `STALE`。未命中的条件可以增量补编码，不要求因一个镜头变化而重编码全项目。

调度器维护显式模型驻留状态，例如 `ENCODING_RESIDENT`、`DIFFUSION_RESIDENT` 和 `UNLOADED`，并把模型切换作为批次级操作。默认先完成当前已批准批次的全部可用编码，再切换到扩散阶段；交互式紧急重试可以开启新批次，不破坏原批次缓存。

缓存文件仅允许加载本 Worker 生成并通过哈希、schema 和版本校验的内容。不得直接信任外部 `.pt`，也不得使用 `torch.load(weights_only=False)` 加载不可信文件。

## 7. 任务与恢复

```text
BLOCKED -> READY -> QUEUED -> RUNNING
                         -> SUCCEEDED
                         -> FAILED
                         -> NEEDS_REVIEW
                         -> CANCELLED
                         -> STALE
```

整个运行可以 `PAUSED`，节点不使用暂停状态。Adapter 内部至少使用 ComfyUI 的 `/object_info`、`/system_stats`、`/prompt`、`/ws`、`/history/{prompt_id}`、`/interrupt` 和队列删除能力。

`prompt_id` 在提交成功后立即持久化。应用重启时先与 ComfyUI 历史对账，再恢复轮询或重派。外部生成只能保证“至少一次提交 + 指纹去重 + 历史对账”，不能宣称严格 exactly-once。

任务指纹包含任务类型、标准化输入、上游产物哈希、模型/节点/workflow 版本和生成参数。

## 8. 产物存储

```text
project.db
artifacts/
  blobs/<sha256>
  previews/
work/
logs/
```

产物先写临时文件，完成媒体探测和 SHA-256 后原子移动，再在事务中登记。manifest 保存 conditioning、AV latent tail、视频和音频的 URI/哈希，以及上游 revision、prompt、参考素材、模型、VAE、LoRA、采样配置、trim 帧数、cache schema version 和审核状态。

## 9. 许可证与备选路径

项目代码采用GPL-3.0；外部节点继续遵守各自许可证，官方MiniMax H3 Turbo插件为Apache-2.0。fork和分发时保留原版权、NOTICE与第三方许可证。模型许可独立于节点代码许可，发布前另行核对地域与商用条款。
