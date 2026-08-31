# H3 受控工作流

MiniMax H3 工作流与项目的静态 conditioning 缓存、Motion Context 连续链、素材寻址、任务恢复和产物登记高度耦合。产品因此随版本提供经过验证的受控 ComfyUI API 节点图，不要求用户自行准备、导入或维护 H3 工作流，也不在 GUI 中提供图形编辑器。

图片生成工作流不受此规则影响。图片模型与节点组合很多，仍由用户导入 ComfyUI API JSON 并绑定 Harness。

## 用户可配置项

全局设置通过当前 Worker 的 `/object_info` 读取候选项，GUI只开放：

- H3 扩散模型；
- 文本编码器；
- 视频 VAE 和音频 VAE；
- Turbo LoRA；
- Turbo 步数、SageAttention和低显存模式。

节点 ID、连线、采样器、conditioning 保存/加载节点、Motion Context 输入和输出提取规则都属于工作流契约，不向用户开放。保存设置只更新后续任务配置，不提交 GPU 任务。

当前受控工作流是官方 Turbo 路径，Turbo 固定启用，步数限制为 4 至 8。标准非 Turbo 路径只有在另一个节点图完成验证并作为新受控修订发布后才会出现在 GUI 中，不通过运行时绕过节点模拟。

## 受控资源

发行资源位于 `src/ai_video_generator/resources/h3/`，开发验收副本位于 `workflows/h3/default/`：

- `conditioning.api.json`：静态文本和参考图 conditioning；
- `initial.api.json`：连续链首段扩散生成；
- `continuation.api.json`：读取上一段 Motion Context 的续段生成；
- `multi-reference-source.api.json`：多图参考输入源；
- `diffusion.api.json`：扩散阶段基图。

运行时按片段生成不可变 workload manifest。工作流文件哈希、节点Schema、模型选择、插件版本、提示词、素材哈希和前段产物都进入任务指纹。所有静态 conditioning 完成后才切换到扩散模型；Motion Context 只在前段完成后作为运行时输入，不进入静态缓存。

## 加速与安全

Turbo加载器和采样器只接受官方 `ComfyUI-MiniMax-H3-Turbo` 节点：

```text
Load Diffusion Model
-> MiniMaxH3TurboLoRA
-> optional ModelAttentionBackend (`comfy kitchen attention`)
-> guider / BasicScheduler model path

MiniMaxH3TurboSampler -> SamplerCustomAdvanced.sampler
BasicScheduler(simple, 4..8 steps) -> SamplerCustomAdvanced.sigmas
```

默认使用 6 步。4 步速度更快，但快速大动作更容易拖影；6 至 8 步通常更稳。Comfy Kitchen Attention 可选且由新版 ComfyUI 原生提供，低显存模式用于 Windows 内存压力较高的环境。

TeaCache始终禁止。Turbo少步图叠加TeaCache会明显降低质量，因此运行时和能力预检都不得接受包含TeaCache的受控图修订。

官方Turbo节点当前锁定提交为`4274783a23afcfdbea3b4876cb79effd6c510785`。插件更新必须先在隔离环境验证并发布新的受控工作流修订，生产Worker不得自动`git pull`。

## 开发接口

以下接口保留用于受控图开发、迁移和兼容性检查，不是用户执行项目的前置步骤：

```text
POST /api/v1/h3/workflows/inspect
POST /api/v1/h3/workflows/compile
POST /api/v1/h3/workflows/profiles
```

产品GUI不会调用这些接口要求用户上传H3图。正常执行直接使用发行包内资源，并在调度前对目标Worker的节点、模型和版本能力重新校验。
# 用户 H3 双工作流套件

内置受控 H3 工作流继续作为默认实现。用户也可以在“工作流模板”中导入两份
ComfyUI API 格式 JSON，分别将用途设为“H3 编码”和“H3 扩散”，完成字段映射并登记。

编码工作流必须暴露以下标准绑定：

- `prompt`：最终执行提示词。
- `width`、`height`：生成尺寸。
- `frame_count`：包含 Motion Context 预算的采样总帧数。
- `conditioning_fingerprint`：编码与扩散阶段共享的稳定缓存键。
- 输出节点必须标记为 `conditioning`。

扩散工作流必须暴露以下标准绑定：

- `conditioning_fingerprint`：读取编码阶段产物的同一个缓存键。
- `seed`：当前分段 Seed。
- `output_prefix`：最终视频输出前缀。
- `motion_context_input`：首段为空；续段为上一段保存的锚点路径。
- `motion_context_output_prefix`：当前段供下一段继承的锚点输出前缀。
- 输出节点必须标记为 `video`。

参考图片、视频和音频可在编码模板中按各自模态从索引 1 开始映射。工作流必须把
conditioning 写入 `output/ai-video-generator/conditioning/<fingerprint>.safetensors`
及同名 JSON 元数据，并按传入的 Motion Context 路径读取和保存锚点。登记只验证
字段、节点 Schema 和输出契约；ComfyUI 图内部的模型兼容性仍由工作流作者负责。

登记两份模板后，在“全局设置 -> MiniMax H3 工作流 -> H3 工作流套件”选择
“用户工作流”并分别选择编码和扩散模板。两份模板必须同时选择；切回“内置受控
工作流”即可恢复原实现。更换套件会改变执行指纹，新任务不会复用旧套件产物。

## 使用 LLM 标定用户工作流

1. 在 ComfyUI 中分别准备编码图和扩散图，并使用“保存（API 格式）”导出 JSON。普通 UI 工作流 JSON 不能登记。
2. 需要由程序赋值的字段必须保留为未连接的字面量输入。LLM 不会、也不允许拆开已有连线后重新接线。
3. 在“工作流模板”导入编码 JSON，将用途选为“H3 编码”，确认节点后点击“LLM 标定 H3”。检查映射并补齐提示的缺失字段，再选择 conditioning 保存节点作为输出并登记。
4. 导入扩散 JSON，将用途选为“H3 扩散”，再次点击“LLM 标定 H3”。选择最终视频保存节点作为输出并登记。
5. 在“全局设置”选择这对工作流并保存。系统不允许只选择其中一份。
6. 先运行 4 秒无参考素材测试，再分别测试图片、视频和音频参考，最后测试至少两个 Motion Context 分段。

LLM 标定只识别“哪个现有节点输入对应哪个标准字段”，不会改写 ComfyUI 拓扑，也不会证明模型、VAE 或自定义节点彼此兼容。无法可靠识别的字段会被省略并显示 warning；登记时确定性契约会再次阻止缺字段或错误输出类型的模板。

### 工作流作者约定

- 编码图必须使用 `conditioning_fingerprint` 写入固定缓存目录，扩散图使用同一值读取。
- 扩散图必须能处理空的 `motion_context_input`：空值表示首段，不能无条件执行文件加载节点；非空值表示续段，必须读取该锚点。
- 每段都必须使用 `motion_context_output_prefix` 保存供下一段继承的锚点。
- `output_prefix` 只用于最终视频，不得与 Motion Context 锚点目录混用。
- 三种参考模态分别从索引 1 开始；图片 1、视频 1 和音频 1 可以同时存在。
- 动态组合字段必须先在 ComfyUI 中选择实际存在的父选项，再暴露对应子字段。
- `width` 和 `height` 是 H3 编码输入画布，不是最终 MP4 分辨率承诺。潜空间放大可以
  输出不同尺寸；系统会使用 ffprobe 记录每段实际尺寸，并以活动片段中像素数最高的
  尺寸合并母版。其他片段按比例缩放和填充到该画布，不会降回项目输入尺寸。

### 标定失败排查

- “缺少标准绑定”：手动添加映射，或回到 ComfyUI 暴露该输入。
- “目标是已连接输入”：该字段属于图内部拓扑，运行时不能安全覆盖。
- “input is not exposed by object_info”：检查自定义节点版本和动态父选项。
- “conditioning 完成但缓存产物缺失”：编码图没有按固定指纹目录保存 safetensors 和 JSON 元数据。
- 第二段 Motion Context 失败：检查首段是否保存锚点，且续段没有写死输入文件名。
