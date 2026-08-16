# 后处理模型与执行策略

## 1. 结论

MVP 后处理分成两类：

- 默认确定性处理：FFmpeg 拼接、裁切、统一画幅/帧率、H.264/AAC、音量和 manifest。它不依赖生成模型，始终可用。
- 显式 AI 处理：SeedVR2 修复放大、RIFE/GIMM插帧、Whisper转写。全部默认关闭，按项目选择Profile和实际模型，并形成独立任务和产物修订。

推荐执行顺序：

```text
已批准的 H3 镜头
-> 按镜头运行 SeedVR2（可选）
-> 按镜头运行 RIFE 或 GIMM（可选，禁止跨切点插帧）
-> FFmpeg 拼接、规格统一和音频封装
-> Whisper 转写并生成 SRT/VTT（可选）
-> 字幕软封装或烧录（可选）
-> 媒体探测、哈希和 manifest
```

模型任务必须串行占用 GPU。进入 Whisper 或 FFmpeg 阶段前卸载 SeedVR2/RIFE，避免 Windows 内存和显存压力叠加。

## 2. SeedVR2

Profile不固定模型。当前机器可选择`seedvr2_7b_int8_convrot.safetensors`与`seedvr2_ema_vae_fp16.safetensors`作为已验证组合，其他安装可选择自身节点实际枚举出的兼容权重。

选择理由：

- 官方模型用途是单步视频修复，覆盖真实世界退化、细节恢复和时序一致性。
- 本机已有 Comfy-Org 转换的 7B INT8 ConvRot 权重，无需新增模型下载。
- ByteDance Seed 的 7B 模型卡和 Comfy-Org 转换仓库均标记为 Apache-2.0。
- 本机完整链路已经通过，不依赖实验性第三方 SeedVR2 节点。

本机实测基线：

| 项目 | 结果 |
|---|---:|
| GPU | RTX 3080 Ti 12 GB |
| 输入 | 1 秒、24 帧 |
| 处理尺寸 | 512x320 |
| temporal chunk | 5 帧、overlap 0 |
| 采样 | 10 步、CFG 1、Euler、simple |
| 总耗时 | 56.057 秒 |
| 峰值显存 | 11,300 MiB |
| 最低空闲 RAM | 10.13 GiB |

因此 7B 不是全局默认开关，而是“高质量修复”预设。启用时必须先运行 1 至 3 秒 A/B 预览并由用户确认。默认预览为 512x320、1 秒、5 帧 chunk、0 overlap。高分辨率、增大 chunk 或 overlap 都需要重新预检；不得根据低分辨率成功结果推断整片一定可运行。

当前简单方块样片显示边缘与表面细节明显增强，同时也放大背景颗粒并产生轻微颜色纹理泄漏。正式质量验收必须换用包含真人、纹理、运动和镜头切换的素材。

3B 版本只作为 7B OOM 时的兼容回退，不作为当前首选质量预设。`sharp` 变体、人脸修复和额外锐化不进入 MVP 默认链路，避免身份与纹理被二次改写。

## 3. RIFE

RIFE与GIMM作为不同Profile，checkpoint由用户从Worker枚举结果中选择。仓库当前携带RIFE API工作流；SeedVR2/GIMM若没有经过验证的API格式工作流会显示未就绪。

使用边界：

- 只把已批准片段转成48、60或120fps，不改变H3生成帧率和时长计算。
- 24→60先以5倍生成120fps中间结果，再由FFmpeg确定性二等分采样；24→120直接保留5倍结果，不降采样。
- 每个镜头独立插帧，禁止在剪辑点、黑场边界或 Motion Context reset 边界之间生成中间帧。
- 快速遮挡、细线、透明物、字幕和闪光画面可能出现重影，启用后必须逐段审核。
- 24 -> 48 是首选整数倍路径；24 -> 60 只在交付端明确要求时使用。

Practical-RIFE 上游已有更新版本，但 4.9.2 的现有 ComfyUI 生态和可复现性更好。升级模型版本必须产生新的后处理 Profile 和产物指纹，不能静默替换权重。

## 4. Whisper

首选模型：OpenAI Whisper `large-v3-turbo`，运行时可采用 `faster-whisper`/CTranslate2 的 `float16` 或 `int8_float16`。Whisper 代码和权重为 MIT。

使用边界：

- 只负责原语言语音转写、时间戳和 SRT/VTT，不承担中文到英文翻译。
- H3 原生音频在最终拼接和音量处理后再转写，避免片段级时间轴二次对齐。
- 默认输出独立字幕文件；烧录字幕是显式选项，并且必须保留未烧录母版。
- 人名、专有名词和无清晰语音的生成音频必须人工复核。

官方说明中 `turbo` 是优化后的 `large-v3`，约 809M 参数、典型显存需求约 6 GB，速度约为 large 的 8 倍，但不适合翻译任务。

## 5. 不采用的默认模型

- TeaCache：Turbo H3 上会显著降低质量，继续由 H3 Profile 校验器拒绝。
- CodeFormer/GFPGAN：可能改变角色身份和面部特征，不作为自动步骤。
- 自动去闪烁扩散模型：可能掩盖生成失败并引入新时序变化，MVP 只做检测和人工返工。
- 通用锐化、降噪模型叠加：SeedVR2 已负责修复，重复处理容易产生塑料感、halo 和纹理漂移。

## 6. 任务与验收

每个 AI 后处理步骤必须保存输入哈希、模型文件哈希、节点/运行时版本、参数和输出哈希。上游片段重做后，只使引用该片段的后处理任务与最终导出失效。

验收至少包括：

- SeedVR2：细节改善、身份一致性、背景颗粒、颜色泄漏和时序稳定性。
- RIFE：切点、快速运动、遮挡、细线和字幕区域无明显重影。
- Whisper：抽查时间戳、人名、标点和静音段误识别。
- FFmpeg：时长误差不超过一帧，音画漂移不超过一帧，无黑帧、重复帧和缺失音轨。

研究来源：

- [ByteDance-Seed/SeedVR2-7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)
- [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2)
- [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
- [openai/whisper](https://github.com/openai/whisper)
