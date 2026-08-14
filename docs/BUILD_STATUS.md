# 构建状态

## 当前阶段

当前完成 P0 dry-run，只构建控制平面与不可提交的执行计划，不执行视频生成。

已经具备：

- Python 3.11/3.12 项目与锁定依赖。
- FastAPI 健康检查和本地 Worker 能力探测。
- 对 ComfyUI 目录、版本、commit、自定义节点和 H3 模型文件的只读盘点。
- `Shot` 下技术片段链的不可变 `ChainSpec`。
- 15 秒、`17k+5` 帧网格、32 像素画布倍数和 Motion Context 邻接校验。
- 包含 prompt、素材顺序、模型哈希、VAE、LoRA、节点版本的条件缓存指纹。
- 固定 Motion Director `a58f282` 源码、NOTICE、许可证和上游单元测试。
- `ConditioningArtifact` 的 `safetensors+json-v1` 内容寻址契约。
- 批量预编码、卸载编码器、加载扩散模型、按 Motion Context 生成的任务 DAG。
- `POST /api/v1/plans/dry-run`，输出在类型上固定为不可提交。
- 非生成单元测试。

当前明确不具备：

- `/prompt` 提交或任何可执行的视频生成接口。
- 将 Motion Director vendor 快照安装到本机 ComfyUI。
- 对 Motion Director 编码路径的本地改造和运行时 patch 验证。
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
POST http://127.0.0.1:8000/api/v1/plans/dry-run
GET http://127.0.0.1:8000/docs
```

能力探测只对 ComfyUI 调用 `GET /system_stats` 和 `GET /object_info`。dry-run 只在控制平面内编译任务 DAG；当前代码没有 ComfyUI 任务提交方法。

## 下一构建步骤

1. 在 vendor fork 中增加独立的条件保存/加载节点，不改其内置 UI 和素材库。
2. 使用 safetensors 张量文件和 JSON manifest 实现实际 ConditioningArtifact I/O。
3. 把 Motion Director 的逐段内部编码拆为预编码与扩散执行两条 workflow。
4. 加入 SQLite/Alembic 的 revision、任务和 artifact 表。
5. 安装 fork 前执行重复 patch owner、节点注册和 workflow JSON 的离线预检。

只有以上 dry-run 验收通过并得到用户允许后，才提交真实 H3 生成任务。

## Vendor 基线

`vendor/motion-director` 是上游 commit `a58f28271a9db8af2a802533a1078b5890c9c4b9` 的固定快照。保留源码、README、NOTICE、LICENSES 和单元测试，排除了演示视频、截图、live UI 测试及上游 Git 元数据。

基线验证：

- 主项目：14 passed。
- Motion Director Python：170 passed、12 skipped。
- Motion Director JavaScript：129 passed。

上游当前会在每段 conditioning 内部调用 Qwen3-VL 编码，尚未实现整批预编码。现有 `.pt` 缓存使用 `weights_only=True` 加载，但缓存身份和目录仍以 ComfyUI `node_id` 为中心，因此不能直接替代产品级 ConditioningArtifact。
