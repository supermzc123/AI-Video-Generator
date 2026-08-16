# 构建状态

## 当前阶段

截至 2026-08-16，仓库已具备 Windows `0.2.0-beta.1` 内部测试安装包，并完成低分辨率、短时长的受控 GPU 验证。

已经实现：

- FastAPI 控制平面、统一错误边界和 OpenAPI 接口。
- 项目不可变修订、任务 DAG、审核决定、幂等、租约、恢复、血缘和 stale 传播。
- 用户 ComfyUI API 工作流导入、LLM或手动字段标定、类型化编译和版本指纹；不依赖特殊节点标题。
- ComfyUI V3 `COMFY_AUTOGROW_V3` 点路径输入的确定性校验。
- H3 外部 Profile、官方 Turbo provider 校验、可选 KJ SageAttention 和 TeaCache 阻断。
- H3 static conditioning 与初始 AV latent 的安全 `safetensors+json-v1` 缓存。
- 全部 conditioning 编码完成后卸载编码器，再加载扩散模型的阶段屏障。
- Niko H3 Motion Context v0.3.0 provider、双流 AV latent、续段裁切和能力探测。
- OpenAI-compatible 多模态 LLM Harness；`gemini-3.7-flash` 已完成真实 JSON、图片和工作流映射联调。
- 内容寻址的远程 Worker workload manifest、HTTPS/WSS、租约、传输和结果收据。
- FFmpeg 导出计划和原子导出函数。
- React/Vite 工作台和 Tauri 2 控制平面生命周期源码。
- 七阶段门控制作流程：创意、大纲、电影分镜、素材、生成、审核和交付；返回上游修改会清除真实下游批准状态。
- 精细化与批量模式共用持久化 DAG；模式切换在原子任务边界生效，项目暂停不会抢占运行中任务。
- 项目工作区已从浏览器 localStorage 迁移到 SQLite 不可变修订。
- 项目专属智能体会注入项目事实、决定、任务、预算和可检索记忆，只能提交受限 Patch，并记录输入/输出哈希。
- 项目记忆使用 SQLite FTS5；锁定决定不能被后续 LLM 或子代理覆盖。
- Harness Bundle/Revision、图片工作流 Markdown Harness 编辑器，以及固定提交 H3 Harness 运行时安装器。
- 审核策略支持 human_ai、ai_only 和 none；人工超时会在项目范围持续切换为 AI 接管。
- 时间预算、性能签名、模型亲和加分与 aging、防饥饿调度、任务检查点和模型驻留事件。
- 批量工作台支持指定任务、优先级、启动、暂停、恢复和取消；启动前执行确定性项目预检。
- 后处理使用版本化 Profile 描述拓扑，项目选择当前 Worker 实际安装的模型；48、60、120fps 插帧均进入任务指纹。
- RIFE API Profile 已随控制平面资源分发，并按片段执行；母版拼接、可选 Faster Whisper 和最终字幕封装已有本地执行器。
- SeedVR2 与 GIMM Profile 在缺少可发布的 ComfyUI API 工作流时明确显示未就绪，不再仅凭节点存在而错误放行。
- Tauri 动态选择控制平面端口并以 service ID、版本和实例 nonce 验证身份；应用数据写入 LocalAppData/XDG。
- Windows Credential Manager 保存 LLM API Key，旧明文设置迁移后从 JSON 删除。
- PyInstaller one-folder 控制平面、固定哈希 FFmpeg/ffprobe、许可证和 H3 资源已进入 MSI/NSIS。

自动验证：214 passed、1 skipped；Ruff、React生产构建和`cargo check --locked`通过。

## GPU 验证

全部 H3 批次使用 RTX 3080 Ti、416x256、124 帧、24fps、6 步 `simple` scheduler、官方 Turbo 节点、`low_vram=true` 和 KJ SageAttention。每批完成后调用 `/free` 卸载模型。Z-Image 使用 256x256、9 步。

| 批次 | 结果 | 耗时 | 峰值显存 | 最低空闲 RAM |
|---|---:|---:|---:|---:|
| Z-Image 单图 | 成功 | 25.535s | 11,647 MiB | 2.32 GB |
| H3 完整单段 | 成功 | 61.148s | 11,912 MiB | 1.48 GB |
| H3 static encode | 成功 | 20.512s | 11,773 MiB | 2.19 GB |
| 缓存扩散 | 成功 | 40.821s | 11,779 MiB | 2.02 GB |
| 换 seed 缓存复用 | 成功 | 30.987s | 11,759 MiB | 2.09 GB |
| Motion Context 首段 | 成功 | 40.706s | 11,794 MiB | 2.14 GB |
| Motion Context 续段 | 成功 | 35.689s | 11,761 MiB | 2.15 GB |
| 公共双图 reference encode | 成功 | 25.546s | 11,691 MiB | 2.33 GB |
| 公共双图缓存扩散 | 成功 | 35.674s | 11,755 MiB | 2.14 GB |
| SeedVR2 7B INT8（1秒/512x320） | 成功 | 56.057s | 11,300 MiB | 10.13 GiB |

关键产物与结论：

- Z-Image 输出清晰的 256x256 PNG，真实 Worker Schema 识别出 7 个输入绑定和图片输出。
- H3 输出为 416x256 H.264、24fps，包含 32kHz 双声道 AAC，时长 5.167 秒。
- static bundle 含 1 组 conditioning 及 video/audio 两个 AV latent，缓存约 2.23 MB；扩散图不再包含 CLIP。
- 换 seed 后复用同一 bundle 成功，证明不是 ComfyUI 整图缓存假命中。
- Motion Context 首段 AV latent metadata 为 `h3_motion_context_av_v1`，video/audio 张量均为有限值。
- 续段加载精确 predecessor 文件，应用 22 帧画面和 24 帧音频上下文；裁切后为 102 帧、4.250 秒，音画漂移 0.00ms。
- 首段尾帧与续段首帧 SSIM 为 0.9825，PSNR 为 40.58dB，无可见跳切。
- 双图 reference 缓存约 6.49 MB；扩散图不含 CLIP、LoadImage 或 ReferenceToVideo，输出保持同一主体身份。
- SeedVR2使用ComfyUI官方节点和本机7B INT8 ConvRot权重完成24帧修复；边缘更清晰，但简单样片也显示背景颗粒放大和轻微颜色纹理泄漏，因此只作为短预览后启用的可选步骤。

测试工作流、metrics、history、抽帧和对比图保存在 `work/gpu-tests/`。ComfyUI 输出保存在 `D:\Comfy_new\ComfyUI\output\gpu-tests`。

## 本机基线

- ComfyUI：`0.30.0`，commit `344b43989e8c56b5bb4a66cf028c834192ab59dd`
- GPU：NVIDIA GeForce RTX 3080 Ti，12 GB VRAM
- 系统 RAM：约 32 GB
- H3：FL2VA pruned INT8、Qwen3-VL-32B INT8、video/audio VAE
- 官方 Turbo：commit `4274783a23afcfdbea3b4876cb79effd6c510785`
- 本次 LoRA：`minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`
- Motion Context：v0.3.0，commit `658ba11ae91737391a247cf9758d0063c43491b3`
- AVG conditioning 节点：已安装并由 `/object_info` 验证
- TeaCache：未在任何验收工作流中使用，Profile 继续阻断

## 剩余验收

- 尚未导入并执行用户提供的图片工作流；已验证内置 Z-Image 和通用模板路径。
- 双图测试使用简单方块素材，尚未对真实人物多角度身份一致性做审美验收。
- H3 音轨结构、Motion Context 音频裁切和零漂移已验证，但静态方块提示生成近静音音频，尚未验证有节奏内容的听感接缝。
- 尚未生成约 60 秒、至少 6 段的 MVP 成片；当前只验证单段和两段连续链。
- Ubuntu 主动 HTTPS/WSS Worker 协议已有测试，尚未部署到真实 Ubuntu GPU 主机。
- SeedVR2官方模板和本机GIMM安装没有可直接登记的API格式工作流，相关Profile保持阻断，不能作为本测试版已完成能力宣传。
- Ubuntu `aivideo-worker` CLI 与 systemd 示例已提供，但声明式 ComfyUI workload执行器仍需真实Ubuntu GPU验收，因此继续标记为实验性。
- MSI/NSIS均为unsigned内部测试包；没有可信Authenticode证书前禁止正式公开Release。
- 32 GB Windows 主机最低只剩约 1.48 GB 空闲 RAM。后续继续串行、低分辨率测试；高分辨率或并发任务很可能触发 OOM。

本机内部测试包：

- `desktop/src-tauri/target/release/bundle/msi/AI Video Generator_0.2.0-beta.1_x64_en-US.msi`
- `desktop/src-tauri/target/release/bundle/nsis/AI Video Generator_0.2.0-beta.1_x64-setup.exe`

MSI已做管理提取检查，确认包含桌面EXE、控制平面、FFmpeg/ffprobe、许可证、H3和RIFE资源；打包控制平面已在独立数据目录完成动态端口/nonce健康检查。

## 验证命令

```powershell
uv sync --extra dev
uv run ruff check .
uv run pytest

Set-Location desktop
npm install
npm run build
```

真实 GPU 测试结果只能追加或明确修订，不得用 dry-run 结论覆盖。
