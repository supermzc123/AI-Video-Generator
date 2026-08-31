# AI Video Generator

一个面向 MiniMax H3 的本地优先视频生成工作台。系统让 LLM 深度参与从创意、大纲、电影分镜到生成计划的编排，同时把时长限制、素材引用、连续性、任务恢复和审核交给确定性程序控制。

## 当前状态

当前发布版本为 `0.2.0-beta.1`，这是可运行的早期 Beta 测试版本，采用 [GPL-3.0](LICENSE) 协议。项目的大部分核心功能已经实现并可以实际操作，包括项目编排、分镜与素材管理、H3 提示词生成、ComfyUI 任务派发、批量任务、视频审核和导出。但系统仍存在较多 Bug、边界场景失败、模型/工作流兼容性问题以及任务状态恢复风险，使用时应保留备份并及时检查生成结果；当前版本不应作为生产环境或唯一数据来源。发布边界和已知风险见 [Beta 发布说明](docs/BETA_RELEASE.md)。

电影分镜与模型执行片段是两个不同层级：

- `Shot` 表示电影语义上的完整镜头，可以超过 15 秒。
- `GenerationSegment` 表示一次 H3 生成，必须不超过 15 秒。
- 长镜头会按动作和运镜语义拆分，并通过 Motion Context 连接相邻片段。
- 项目公共素材（例如主角参考图）可以自动绑定到所有相关片段。

## 已知漏洞
1.批量工作台很可能不能正常使用
2.自定义视频工作流未经过严谨测试

## 快速开始
- 克隆仓库到本地（不建议使用携带版，bug更多）
- 点击start-app.bat
- 访问网页



## 文档

- [项目计划](docs/PROJECT_PLAN.md)
- [系统架构](docs/ARCHITECTURE.md)
- [职责与协作](docs/RESPONSIBILITIES.md)
- [H3受控工作流与加速策略](docs/H3_WORKFLOWS.md)
- [后处理模型与执行策略](docs/POST_PROCESSING.md)
- [当前构建状态](docs/BUILD_STATUS.md)
- [面向用户的使用说明](docs/USER_GUIDE.md)
- [Harness 文件与生效范围](docs/HARNESS_GUIDE.md)

## 已实现架构

- Tauri 2 + React/TypeScript/Vite桌面工作台
- Python、FastAPI、Pydantic和SQLite WAL控制平面
- 通过稳定的 Worker Adapter 接入本地 ComfyUI
- 随软件发布的版本化H3受控工作流，GUI只暴露模型与受支持的运行参数
- 生成批次先统一完成条件编码并落盘，再卸载编码器、加载扩散模型执行生成
- 用户导入、节点标定并批准的 ComfyUI API 图片工作流；不内置依赖特定模型文件名的工作流
- OpenAI-compatible LLM Harness、持久任务DAG、远程Worker协议和FFmpeg导出
- 九阶段门控制作台，以及基于Profile和本机模型选择的SeedVR2、RIFE/GIMM、Whisper后处理配置

真实 H3 Worker 的完整启动方式将在 P0 执行层验证后补充。

## 本地开发

项目ComfyUI节点位于`comfyui_nodes/ai_video_generator_nodes`。H3图作为受控资源随软件发布，用户无需准备或导入；全局设置从当前Worker的`/object_info`模型清单中选择扩散模型、文本编码器、VAE和Turbo LoRA。图片生成工作流仍由用户按需导入。

```powershell
.\scripts\setup.ps1
.\scripts\run-api.ps1
```

启动后访问 `http://127.0.0.1:8000/docs`。本机配置放在被 Git 忽略的 `.env`，可从 `.env.example` 创建。

Windows 开发环境可运行 `powershell -ExecutionPolicy Bypass -File .\scripts\restart-api.ps1` 重启后端；脚本会校验端口占用者并等待健康检查通过。
也可以直接双击项目根目录的 `restart-backend.bat`。该脚本只重启 API；要同时确保前后端在线并打开网页，请双击 `start-app.bat`。

桌面前端开发：

```powershell
Set-Location desktop
npm install
npm run dev
```

访问`http://127.0.0.1:1420`。构建Tauri安装包需要Rust/Cargo、固定版本FFmpeg和PyInstaller：

```powershell
Set-Location desktop
npx tauri build --bundles msi nsis
```

没有配置可信Authenticode证书时，产物只能作为unsigned内部测试包。

检查：

```powershell
uv run ruff check .
uv run pytest
```

API和节点安装说明见[API契约](docs/API.md)与[ComfyUI节点安装](docs/COMFYUI_NODES.md)。当前完成度和未执行验收门见[构建状态](docs/BUILD_STATUS.md)。
