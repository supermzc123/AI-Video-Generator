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
-> optional MiniMaxH3MemoryEfficientSageAttentionPatch
-> guider / BasicScheduler model path

MiniMaxH3TurboSampler -> SamplerCustomAdvanced.sampler
BasicScheduler(simple, 4..8 steps) -> SamplerCustomAdvanced.sigmas
```

默认使用 6 步。4 步速度更快，但快速大动作更容易拖影；6 至 8 步通常更稳。SageAttention可选，低显存模式用于Windows内存压力较高的环境。

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
