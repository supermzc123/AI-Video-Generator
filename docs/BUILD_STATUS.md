# 构建状态

## 当前阶段

当前完成 P0 Foundation，只构建控制平面基础，不执行视频生成。

已经具备：

- Python 3.11/3.12 项目与锁定依赖。
- FastAPI 健康检查和本地 Worker 能力探测。
- 对 ComfyUI 目录、版本、commit、自定义节点和 H3 模型文件的只读盘点。
- `Shot` 下技术片段链的不可变 `ChainSpec`。
- 15 秒、`17k+5` 帧网格、32 像素画布倍数和 Motion Context 邻接校验。
- 包含 prompt、素材顺序、模型哈希、VAE、LoRA、节点版本的条件缓存指纹。
- 非生成单元测试。

当前明确不具备：

- `/prompt` 提交或任何视频生成接口。
- Motion Director fork 和运行时 patch。
- 条件张量的实际保存/加载节点。
- 数据库、调度器、素材上传、LLM 和 Web 工作台。

## 本机只读盘点

盘点日期：2026-08-14。

- ComfyUI：`0.30.0`
- ComfyUI commit：`344b43989e8c56b5bb4a66cf028c834192ab59dd`
- ComfyUI Python：`3.12.9`
- PyTorch：`2.11.0+cu130`
- GPU：NVIDIA GeForce RTX 3080 Ti
- H3 基模：FL2VA pruned INT8 convrot
- 文本/多模态编码器：Qwen3-VL-32B INT8 convrot
- VAE：MiniMax H3 video FP16、audio FP32
- 已安装相关节点：H3 Audio T8 `1.3.2`、H3 Turbo `1.2.2`、Spectrum H3
- Motion Director：未安装
- 独立 Motion Context/Contex Loop 冲突：未检测到
- ComfyUI 服务：盘点时未运行

现有中文启动批处理中的 Python 路径指向另一套 `H:` 盘 ComfyUI，而不是当前配置目录。为了不改动用户环境，本阶段只记录问题，不修改启动脚本。

## 本地命令

```powershell
.\scripts\setup.ps1
.\scripts\run-api.ps1
```

API：

```text
GET http://127.0.0.1:8000/api/v1/health
GET http://127.0.0.1:8000/api/v1/workers/local/capabilities
GET http://127.0.0.1:8000/docs
```

能力探测只对 ComfyUI 调用 `GET /system_stats` 和 `GET /object_info`。当前代码没有任务提交方法。

## 下一构建步骤

1. 引入并固定 Motion Director fork，先隔离执行核心与其内置 UI/素材库。
2. 冻结 ConditioningArtifact 格式和安全加载方式。
3. 实现“整批预编码、持久化、卸载编码器、再加载扩散模型”的两阶段 Worker workflow。
4. 加入 SQLite/Alembic 的 revision、任务和 artifact 表。
5. 在不采样的 dry-run 模式下验证 workflow 编译、节点能力和缓存指纹。

只有以上 dry-run 验收通过并得到用户允许后，才提交真实 H3 生成任务。
