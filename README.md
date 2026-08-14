# AI Video Generator

一个面向 MiniMax H3 的本地优先视频生成工作台。系统让 LLM 深度参与从创意、大纲、电影分镜到生成计划的编排，同时把时长限制、素材引用、连续性、任务恢复和审核交给确定性程序控制。

## 当前状态

项目处于规划与技术验证阶段，采用 [GPL-3.0](LICENSE) 协议。

首版目标是跑通以下闭环：

```text
创意输入 -> 大纲审核 -> 电影分镜审核 -> H3 片段编译
         -> ComfyUI 生成 -> 片段审核/局部重试 -> 拼接导出
```

电影分镜与模型执行片段是两个不同层级：

- `Shot` 表示电影语义上的完整镜头，可以超过 15 秒。
- `GenerationSegment` 表示一次 H3 生成，必须不超过 15 秒。
- 长镜头会按动作和运镜语义拆分，并通过 Motion Context 连接相邻片段。
- 项目公共素材（例如主角参考图）可以自动绑定到所有相关片段。

## 文档

- [项目计划](docs/PROJECT_PLAN.md)
- [系统架构](docs/ARCHITECTURE.md)
- [职责与协作](docs/RESPONSIBILITIES.md)

## MVP 技术方向

- 本地 Web UI，后续再评估 Tauri 桌面封装
- Python、FastAPI、Pydantic、SQLAlchemy、Alembic
- SQLite WAL 保存结构化状态，媒体产物保存为文件
- 通过稳定的 Worker Adapter 接入本地 ComfyUI
- fork Motion Director 作为唯一 H3 执行引擎候选，先经 P0 实测
- OpenAI-compatible LLM 接口，首版只验证一个实际模型
- FFmpeg 完成基础拼接、转码和 MP4 导出

具体依赖和启动方式将在 P0 技术验证后补充。
