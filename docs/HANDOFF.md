# AI 视频生成工具项目交接文档

> 最后核对：2026-08-21  
> 当前分支：`codex/release-readiness`  
> 当前提交：`65748f4 add guided ComfyUI node installation`  
> 产品版本：Python `0.2.0b1`，桌面端 `0.2.0-beta.1`

本文档面向接手项目的新对话或开发者。它优先记录当前真实状态、尚未提交的工作、用户已经确定的产品规则，以及下一步应如何验证，而不是重复早期规划。

## 1. 项目目标

项目是一个 Windows Tauri 2 桌面应用，通过本地或远程 ComfyUI Worker 完成 AI 视频生产。目标流水线为：

```text
项目配置
-> 创意与参考图
-> 大纲
-> 电影分镜
-> 素材规划与图片生成
-> 显式 H3 视频提示词
-> H3 条件编码与视频生成
-> AI/人工审核与局部返工
-> 可选超分、插帧和字幕
-> FFmpeg 导出
```

系统同时支持两种执行方式：

- 精细化模式：用户在流水线中直接运行任务、查看媒体、编辑提示词、驳回和重做。
- 批量模式：系统跨项目按依赖和模型亲和性调度。任务列表只用于展示进度，不承担执行操作。

首个正式作品验收目标仍是约 60 秒、至少 4 个电影分镜和 6 个 H3 执行片段，并包含至少一个超过 15 秒、使用 Motion Context 连续生成的镜头。

## 2. 已冻结的产品规则

后续实现和重构不得无意改变以下规则：

### 2.1 H3 与长镜头

- H3 单个执行片段最多 15 秒，电影语义镜头可以超过 15 秒。
- 超过 15 秒的镜头必须拆成多个片段，每段独立编写提示词。
- Motion Context 会把前一段末尾 AV latent 注入下一段，并在生成后裁掉继承头。
- 继承头占用下一段采样预算，续段必须至少预留约 2 秒冗余，不能把 15 秒全部写成新内容。
- 分段不要求机械地采用 `15+15`，例如 30 秒可以按叙事节奏拆成 `10+10+10`，可以由AI或者用户决定。如果用户的划分超出限制必须显式提醒并阻止通过。
- 条件编码、模型卸载和模型切换属于系统任务，不需要用户审批。
- Turbo 是可选项，不能强制开启。TeaCache 禁用。Kitchen Attention 可选，使用 ComfyUI 原生 `ModelAttentionBackend`，不依赖外部注意力插件。
- H3 工作流与缓存、Motion Context和任务契约高度耦合，使用项目维护的受控拓扑；GUI只暴露模型、Turbo和受支持参数，不提供图形拓扑编辑。

### 2.2 提示词与 Harness

- 图片和视频提示词最终都汇集到用户可编辑的文本框；用户点击下一步即确认文本。
- 文本 Reviewer 可保留为审计信息，但不得在用户确认后继续阻塞 DAG。
- 项目主管是项目记忆和决定的唯一写入者；子代理无状态。
- H3 模式路由优先确定性计算，普通单镜头不应无意义地重复调用 Director 和 Planner。
- 当前 MiniMax 官方规范要求 H3 执行描述使用英文；只有对白、歌词和画面内文字保留原语言。
- Ref2VA 的 `detailed_description` 通常应达到 350 至 500 个英文词；当前确定性最低门槛为 300 词。
- Ref2VA必须明确限制参考特征的作用范围，禁止服装纹理、图案或材质传播到脸、皮肤、肢体、其他主体或背景。
- 旧数据库与旧提示词不进入新运行时；使用 `scripts/reset-control-plane.ps1` 创建干净控制库后重新建立项目。

### 2.3 素材与图片工作流

- 用户在创意阶段即可上传、命名和删除图片，素材阶段继续显示并提供给 LLM。
- 素材以稳定 `asset_id` 引用，重命名不改变语义关系。
- 缩略图应支持放大预览；已生成图片可以按原提示词单独重新生成，并先形成候选版本。
- 图片生成分辨率独立于视频分辨率。AI建议总像素约控制在 `1280*1280`，用户可以手动修改。
- 图片工作流允许用户导入 ComfyUI API JSON，通过 LLM 或手动标定字段，最后选择真实输出节点。
- 不使用 `AVG_OUTPUT_IMAGE` 等特殊标题或硬编码节点名作为导入契约。
- 一个图片工作流绑定一套用户可编辑 Harness；项目主管将项目需求和素材上下文直接交给图片提示词子代理。

### 2.4 审核、执行与交互

- 审核模式包括 `human_ai`、`ai_only` 和 `none`。
- `none` 只跳过语义审美审核，媒体完整性、时长、黑帧、音画等确定性检查不能跳过。
- 需要用户审批的图片或视频必须在当前页面可预览或播放。
- AI拒绝应按 `project_id + segment_id + source_task_id` 精确创建局部返工，不得从任务ID猜项目编号。
- 精细化任务只能从流水线执行；批量任务只能从批量工作台执行；任务列表保持只读。
- 已完成的图片、H3片段和其他真实产物应尽可能复用。启用超分或插帧不能重新执行完整H3链。
- 失败任务重试必须清理旧租约、ComfyUI prompt关联和错误状态，再回到可执行边界。
- 生成页面的“重新开始”用于重新读取当前全局H3设置，使旧执行链失效并建立新的H3链。

### 2.5 桌面、发布与安全

- UI以中文为主，错误信息必须醒目，不能藏在副标题中。
- 宽高输入应允许清空，失焦或确认后自动四舍五入为32的倍数。
- 控制平面、调度器和桌面端总RSS目标低于800MB，不包含ComfyUI、模型进程、FFmpeg和系统WebView共享进程。
- LLM密钥不得写入Git、日志或回复。目标存储为Windows Credential Manager。
- 未经用户再次明确授权，不构建安装包。用户最近明确要求过“我还没有测试，不要打包”。
- 未签名安装包不能作为正式公开Release。

## 3. 技术架构

```text
Tauri 2 + React 19 + TypeScript + Vite
                 |
          HTTP / SSE / WebSocket
                 |
Python 3.11/3.12 + FastAPI + Pydantic
  |-- Project / immutable revision service
  |-- Project memory + FTS5
  |-- LLM harness + deterministic validators
  |-- Task DAG + scheduler + leases
  |-- Artifact / review / rework services
  |-- FFmpeg and post-processing executors
                 |
       ComfyUI local or remote Worker
```

主要设计：

- SQLite WAL保存项目修订、记忆、决定、任务、租约、审核、预算、模型驻留和产物索引。
- 大型媒体保存在内容寻址Blob目录，SQLite只保存元数据、哈希、血缘和状态。
- H3静态conditioning以受校验的`safetensors + json-v1`保存。
- 动态Motion Context不伪装成静态缓存。
- ComfyUI通过`/object_info`、`/prompt`、WebSocket、history、interrupt和队列接口集成。
- 所有任务使用幂等键、显式依赖、租约、能力manifest和stale传播。
- LLM使用OpenAI-compatible接口，阶段结果经结构化Schema和领域校验后提交。
- Ubuntu远程Worker协议和CLI已存在，但仍是实验性支持。

## 4. 代码导航

| 路径 | 职责 |
|---|---|
| `src/ai_video_generator/api.py` | FastAPI入口、项目/任务/设置/Worker/审核接口、本地派发器 |
| `src/ai_video_generator/domain/` | 项目、提示词、任务、审核、工作流、后处理等公共类型 |
| `src/ai_video_generator/persistence/task_store.py` | SQLite schema、不可变修订、任务、租约、记忆和恢复 |
| `src/ai_video_generator/services/project_tasks.py` | 项目状态编译为任务DAG、依赖和失效传播 |
| `src/ai_video_generator/prompting_api.py` | 显式提示词阶段、片段拆分、Harness编排和checkpoint |
| `src/ai_video_generator/llm/h3_prompt.py` | H3 Writer/Reviewer Harness、官方规范和确定性校验 |
| `src/ai_video_generator/workers/` | ComfyUI探测、提交、取消、历史和能力适配 |
| `desktop/src/App.tsx` | 桌面应用状态、项目加载、全局路由和设置衔接 |
| `desktop/src/PipelineView.tsx` | 精细化九阶段流水线和主要用户操作 |
| `desktop/src/BatchView.tsx` | 批量工作台 |
| `desktop/src/TasksView.tsx` | 只读任务进度表 |
| `desktop/src/SettingsView.tsx` | LLM、ComfyUI、模型和节点安装设置 |
| `desktop/src-tauri/` | Tauri生命周期、动态端口、控制平面资源和桌面打包配置 |
| `comfyui_nodes/` | 项目专属AVG conditioning缓存节点 |
| `scripts/` | 启动、节点安装、控制平面和发布资源脚本 |
| `tests/` | Python契约、回归和执行层测试 |
| `docs/` | 架构、API、H3、后处理、远程Worker和构建快照 |

## 5. 当前实现范围

以下能力已经有代码和相应测试，但不代表全部经过真实用户验收：

- 新建、打开和持久化项目。
- 项目配置、创意、大纲、分镜、素材、提示词、生成、审核和交付流水线。
- 创意阶段图片上传、命名、引用和删除。
- 素材缩略图、改名、AI需求素材、候选重新生成、接受和放弃。
- 用户图片工作流导入、节点探测、LLM或手动字段标定、输出选择、类型化编译和版本指纹。
- 全局ComfyUI路径和项目节点一键安装。
- 项目级LLM对话、流式输出、FTS5记忆、受限Patch和调用审计。
- H3提示词专用Harness、超过15秒镜头拆段和Motion Context续段描述。
- H3静态条件编码、模型切换、Turbo可选、原生 Kitchen Attention 可选、TeaCache阻断。
- H3生成、ComfyUI取消/历史对账、产物登记和局部失效。
- AI审核、人工审核、审核超时接管、局部返工接口和媒体Artifact API。
- 精细化DAG、批量DAG、租约、模式切换、暂停、恢复、取消和重试基础。
- FFmpeg母版与交付导出。
- 后处理Profile、Worker模型选择、48/60/120fps插帧任务设计。
- Windows动态控制平面端口、服务身份nonce、LocalAppData和凭据迁移代码。
- Ubuntu远程Worker协议、CLI和实验性部署资料。

## 6. 当前工作树：必须保留

当前分支为`codex/release-readiness`，跟踪远端同名分支。`main`仍是较早版本。当前有13个已修改但未提交的文件，约551行新增、68行删除：

```text
desktop/src/PipelineView.tsx
desktop/src/api.ts
desktop/src/types.ts
src/ai_video_generator/api.py
src/ai_video_generator/domain/orchestration.py
src/ai_video_generator/llm/h3_prompt.py
src/ai_video_generator/persistence/task_store.py
src/ai_video_generator/prompting_api.py
src/ai_video_generator/services/project_tasks.py
tests/test_h3_prompt_harness.py
tests/test_orchestration.py
tests/test_project_task_compilation.py
tests/test_task_store.py
```

禁止使用`git reset --hard`、`git checkout --`或其他方式丢弃这些修改。接手后先运行`git status --short --branch`确认状态。

### 6.1 未提交修改包含的功能

#### 生成阶段“重新开始”

- 新增`generation_revision`。
- `restart_h3=true`时重新读取全局H3配置。
- 取消并使旧H3执行链失效。
- 重建条件编码、模型切换、H3、审核和导出任务。
- 将模型、Turbo、步数等实际Profile写入任务指纹。
- 前端生成页新增“重新开始”。

#### 失败任务真正重试

- 新增`prepare_task_retry()`。
- 清理旧ComfyUI prompt关联、租约和错误。
- 恢复到`READY`边界。
- 正确处理attempt并拒绝超过上限的自动重试。

#### 简化提示词确认

- AI和人工提示词进入同一文本框。
- 文本非空且用户点击下一步即确认。
- 文本Reviewer只作审计，不再阻塞DAG。
- 媒体审核仍正常保留。

#### H3官方规范修复

- H3执行描述恢复为英文。
- Ref2VA `detailed_description`确定性最低门槛为300英文词。
- 基础模式描述最低60英文词。
- `<d>`中的中文对白合法。
- 增加参考纹理不得污染脸、皮肤、肢体或背景的约束。
- UI同步说明官方英文执行稿规则。

## 7. 最近验证结果

最近一次在上述Harness修改后完成：

```text
Python全量测试：232 passed
H3 Harness专项：10 passed
Ruff：通过
前端 npm run build：通过
```

本次交接文档创建过程没有再次运行全量测试，也没有调用LLM或GPU。因此接手者应把上述结果理解为“最近一次已知通过”，不是当前机器此刻重新执行的结果。

`docs/BUILD_STATUS.md`是2026-08-16的构建快照，其中`217 passed, 1 skipped`、GPU批次和打包结果仍可作为历史证据，但它没有覆盖当前13个未提交修改，也不能代表公开发布就绪状态。

## 8. 本机开发环境

```text
工作区：<项目根目录>
ComfyUI：<用户选择的 ComfyUI 安装目录>
GPU：NVIDIA RTX 3080 Ti 12GB
系统内存：约32GB
开发API：通常为 http://127.0.0.1:8000
Vite前端：http://127.0.0.1:1420
ComfyUI：http://127.0.0.1:8188
应用数据：%LOCALAPPDATA%\supermzc123\AI Video Generator\data
```

常用命令：

```powershell
# 启动桌面开发环境和控制平面
.\start-app.bat

# 仅重启控制平面
.\restart-backend.bat

# Python检查
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q

# 前端检查
Set-Location .\desktop
npm run build
```

2026-08-21最后一次运行状态检查中，控制平面8000和ComfyUI 8188均离线。这只是进程未启动，不是已知代码故障。需要体验时优先运行`start-app.bat`。

### 8.1 本机关键模型

本机已知包含：

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`
- Qwen3-VL 32B MiniMax H3编码器
- H3视频和音频VAE
- SeedVR2 7B INT8

模型、媒体、数据库和本机Harness不得加入Git。

### 8.2 凭据安全

不要读取、打印或提交任何API Key。即使工作区中存在历史`api-key.txt`或旧设置文件，也不得在日志、文档、回复或Git中暴露内容。应用目标行为是使用Windows Credential Manager；网络地址和模型名可记录，密钥值不可以。

## 9. 当前质量问题诊断

### 9.1 纸飞机基线

- 项目ID：`b9d6f60c-e86b-4a46-93d6-733133d5e19f`
- T2VA、FL2VA模型、Turbo 4步、SageAttention、`low_vram=true`。
- 主体和场景简单，历史结果相对干净。

### 9.2 最新舞蹈项目

- 项目ID：`a9511d17-b671-4741-828f-3d3bb7d99880`
- Ref2VA、Turbo 6步、SageAttention、`low_vram=false`。
- 320x544、24fps、约12秒。
- 问题是人物服装纹理从首帧开始扩散到脸、手臂和身体，出现严重结构污染。
- Ref2VA checkpoint选择本身符合参考生成用途。
- 旧提示词只有约759字符，其中详细描述约195字符，远低于官方建议。

当前最可信判断：问题主要来自错误且过短的Ref2VA Context-IR、复杂参考图和320短边共同作用；Turbo和Sage可能放大问题，但尚未通过控制变量实验确认。不能仅凭一次结果认定Ref2VA模型错误。

MiniMax官方参考：

- `https://github.com/MiniMax-AI/MiniMax-H3/blob/main/README.zh-CN.md`
- `https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt`

## 10. 未完成项和发布风险

按优先级排列：

### P0：真实生成验收

1. 尚未用新英文Harness真实生成Ref2VA对照片段。
2. 尚未完成约60秒、至少6段的首次整片验收。
3. 最新舞蹈伪影根因尚未完成控制变量实验。
4. 当前13个修改尚未提交，不能直接打Tag或Release。

### P1：功能与体验

1. 自动审核过去出现过过度放行，需要用明确坏样本继续验证阈值、证据和局部返工。
2. 批量模式尚未由用户从大纲边界开始完整体验。
3. AI+人工模式中，一个素材返工不应串行挂起后续所有独立审核，需要持续检查调度行为。
4. 精细化模式、失败重试、生成“重新开始”和产物复用需要真实UI回归。
5. SeedVR2会放大颗粒或纹理泄漏，只能先生成短预览再决定是否接受。
6. GIMM和SeedVR2缺少可发布的通用API Profile时应明确显示未就绪，不能伪装可用。

### P2：平台与发布

1. Ubuntu Worker尚未在真实Ubuntu GPU主机端到端验证。
2. Windows安装包未签名，禁止正式公开Release。
3. 高分辨率H3在Windows下可能耗尽内存；GPU冒烟应使用低分辨率和短时长。
4. 必须在干净Windows账户、空数据库、无系统Python/FFmpeg依赖的环境重新验收安装流程。
5. 用户尚未完成本轮功能测试，并明确要求暂不打包。

## 11. 推荐开发和验收计划

### 阶段A：确认当前代码基线

1. 检查`git status`，确认13个未提交文件仍存在。
2. 运行Ruff、Python全量测试和前端生产构建。
3. 若失败，只修复当前修改引入的回归，不做无关重构。
4. 更新测试数量和失败证据，但暂不修改历史GPU数据。

退出条件：确定性测试全部通过，工作树变化可解释。

### 阶段B：Ref2VA四秒控制实验

使用同一舞蹈参考图，先做低成本实验：

```text
模式：Ref2VA
提示词：新Harness生成的官方英文完整IR
时长：4秒
短边：建议512
Turbo：开启，6步
SageAttention：关闭
其他参数：尽量保持原项目一致
```

比较原320短边中文短提示词结果，重点观察：

- 服装纹理是否继续污染脸和皮肤。
- 人物身份和服装局部特征是否稳定。
- 动作是否符合提示词时间线。
- 参考图中场景和角色职责是否发生交叉污染。

若仍严重：

1. 测Ref2VA标准采样约20步，Sage关闭。
2. 与Turbo 6步、Sage关闭结果对比。
3. 必要时移除场景参考图，只保留角色参考，检查多参考冲突。
4. 每次只改变一个变量，保存任务manifest、提示词、模型哈希和产物。

退出条件：至少找到一个明显改善的组合，或得到足以定位模型/参考图限制的对照证据。

### 阶段C：精细化全链路回归

以新项目从头完成：

1. 新建项目并设置独立图片/视频分辨率。
2. 创意阶段上传、命名、改名、预览和删除图片。
3. 自动生成并应用大纲、分镜和素材规划。
4. 生成需求图片，测试原提示词重新生成、候选对比、接受和放弃。
5. 生成并手动修改H3提示词，点击下一步应直接确认。
6. 在生成页直接执行GPU任务，系统任务不请求审批。
7. 制造一次可恢复失败，验证“重试”真正重新派发。
8. 修改H3全局设置后点击“重新开始”，确认旧H3链失效而图片结果复用。
9. 在审核页播放媒体，执行接受、人工驳回、AI驳回和局部返工。
10. 导出H.264/AAC文件并显示明确产物路径。

退出条件：用户无需进入任务列表执行操作，所有需要审批的媒体均可见，重试和重新开始行为符合语义。

### 阶段D：长镜头和Motion Context

1. 建立一个18至24秒连续镜头。
2. 验证LLM选择自然衔接点，而不是机械15秒切割。
3. 验证每段有独立完整提示词，续段明确继承前段结束状态。
4. 验证续段至少预留2秒context预算。
5. 检查继承头裁切、最终时长、音画同步和接缝。
6. 使用支持视频模态的Gemini审核前段、后段、接缝中心代理和最终成片；网关拒绝视频时记录原因并回退抽帧。

退出条件：无重复可见头帧，时长和音画误差满足契约，接缝审核结果有时间点证据。

### 阶段E：审核与批量

1. 用正常透视遮挡、明确形体异常、黑帧和真实H3样本验证Reviewer。
2. 正常裁切、自遮挡和运动模糊不得仅凭单帧判定身体残缺。
3. 临界置信度进入人工审核，不直接自动通过。
4. AI拒绝和人工拒绝都能准确定位项目、片段和源任务。
5. 从大纲批准边界把两个短项目加入批量，一个纯文本、一个带参考图。
6. 验证模型亲和调度、aging、暂停、恢复、取消、批量转精细化和失败隔离。
7. 确认一个素材返工不会阻塞其他独立素材审核。

退出条件：两个项目真实完成或按预期取消，批次状态、DAG状态和GPU状态一致。

### 阶段F：60秒整片与发布准备

1. 完成至少4个电影分镜、6个H3片段和一个超过15秒连续镜头。
2. 首次整片默认关闭SeedVR2和插帧，先验收H3、审核、恢复和FFmpeg主链。
3. 主链通过后再分别启用短片段SeedVR2和48/60/120fps插帧，确认不重跑H3。
4. 更新`docs/BUILD_STATUS.md`为最新真实状态。
5. 用户明确确认后再提交当前修改。
6. 只有用户完成验收并再次明确要求时，才构建和检查安装包。

退出条件：真实整片可播放、可恢复、可局部重做，用户批准进入打包阶段。

## 12. 新对话接手清单

新对话开始时建议按以下顺序操作：

1. 阅读本文档，不要仅依赖旧`BUILD_STATUS.md`。
2. 运行`git status --short --branch`，保护13个未提交文件。
3. 阅读`src/ai_video_generator/llm/h3_prompt.py`和对应测试，确认官方英文Harness修改仍在。
4. 检查控制平面和ComfyUI是否启动；离线时用`start-app.bat`，不要把离线当代码失败。
5. 先运行确定性测试，再决定是否启动GPU。
6. GPU验证先做4秒、低分辨率控制实验，避免Windows内存溢出。
7. 不读取或输出密钥，不删除数据库、媒体、模型或用户未提交修改。
8. 不打包、不Tag、不Release，除非用户在新对话中明确授权。

可直接提供给新对话的开场指令：

```text
请先阅读 docs\HANDOFF.md，并检查当前git状态。
保护现有未提交修改，不读取或输出任何API Key，不打包。
先运行确定性验证，然后按文档“阶段B”执行Ref2VA四秒控制实验；
每次只改变一个变量，保存manifest和产物，并根据结果修复回归。
```

## 13. 相关文档

- `docs/ARCHITECTURE.md`：系统边界和领域模型。
- `docs/API.md`：主要API契约。
- `docs/H3_WORKFLOWS.md`：受控H3工作流与Profile。
- `docs/H3_MOTION_CONTEXT.md`：连续生成、上下文和裁切。
- `docs/POST_PROCESSING.md`：超分、插帧和字幕Profile。
- `docs/COMFYUI_NODES.md`：项目节点安装和能力要求。
- `docs/REMOTE_WORKER.md`：远程Worker协议与实验性Ubuntu路径。
- `docs/BUILD_STATUS.md`：2026-08-16历史构建及GPU测试快照。

本文档是当前交接入口；若它与更早的规划文档冲突，以本文记录的最新用户决策和实际工作树为准，并通过代码和测试再次确认。

## 14. 2026-08-21 H3 Harness v2 实施状态

本轮已完成代码改造，但尚未执行真实 GPU 生成：

- H3 Harness revision 使用冻结的 v2 manifest；新任务只使用被批准修订中的文档快照，不回读安装目录或旧 revision。
- 运行链已拆为 Preflight、Director、可选 Planner、Writer、确定性 Validator、Reviewer、定向 Repair 和 Finalize，并记录阶段轨迹、假设、错误和调用统计。
- H3 英文稿字段为 `execution_prompt`；中文对照按需生成，不参与执行或审核。
- 图片、视频和音频均已成为正式参考模态。受控 ComfyUI 编译链分别使用 `LoadImage`、`LoadVideo + GetVideoComponents` 和 `LoadAudio`，上限为 9 张图片、3 个视频、3 段独立音频。
- 视频上传使用 ffprobe 验证可解码视频流、2 至 15 秒时长和精确 24fps，并记录宽高、时长、帧率及音轨状态；音频上传验证可解码音频流和正时长。
- 带音轨视频会占用对应 `<Audio N>` 标签，独立音频从后续编号开始，避免提示词标签与 H3 tokenizer 的实际顺序不一致。
- Motion Context 的 `continuationOf` 不再信任提示词中的可编辑字符串；任务编译和批次调度统一按当前分镜的 `shotId + segmentIndex` 推导紧邻前驱。H3 任务同时登记视频与实际 latent 路径，缺少 latent 时不会把前段误记为成功。
- 桌面端支持图片、视频和音频上传、播放/预览，并展示素材职责、自动假设、阶段轨迹和按需中文对照。
- 新增 30 例脱敏离线 H3 路由评测集，以及多模态工作流、manifest 防篡改、翻译缓存和 Validator 回归测试。

2026-08-21 验证结果：

```text
ruff check .        passed
pytest -q           241 passed
desktop npm build   passed
real ffprobe smoke  24fps MP4 with audio + WAV passed
```

仍需按本文阶段 B 至 D 执行真实 GPU 验收：舞蹈 Ref2VA 四秒对照、T2VA、边界帧模式、多镜头、视频参考、音频参考和 Motion Context 续段。不得把上述离线通过记录描述为 GPU 已验收。

2026-08-21 同轮追加修复：项目负责人对话将模型内部指令与用户可见文字分离，历史自动初始化只显示一条简短记录，并阻止阶段切换重复触发同一自动尝试。批量工作台已增加项目搜索、批次命名、全选、活动/历史筛选、成员进度和错误摘要；后端新增受校验的运行中批次 DAG 扩展操作，解决大纲后自动编排无法把新任务加入冻结批次的问题，并在 DAG 生成后才应用所选执行边界。项目不能同时加入两个活动批次。
