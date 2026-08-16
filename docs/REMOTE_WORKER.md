# Ubuntu远程Worker基础

远程Worker只主动连接控制平面，不监听公网端口，也不把ComfyUI暴露到公网。生产拓扑必须由反向代理终止TLS，Worker使用HTTPS或WSS访问控制平面。

## 已实现边界

- Bearer Token注册及能力上报。
- HTTP心跳/任务认领，以及WSS事件轮询和断线退避。
- 单GPU单活动租约、定时续租和租约丢失中止。
- SHA-256内容寻址的分块上传、分块下载和落盘前校验。
- 稳定`report_id`结果上报、SQLite持久化收据和重复结果去重。
- 网络超时、HTTP 408/429/5xx指数退避。
- 控制平面重启后重新注册；响应丢失时返回原活动租约。
- 远程认领严格过滤`execution_target=remote`。
- 内容寻址的`avg-comfyui-workload+json-v1`执行manifest及Worker端二次校验。

安装Worker依赖：

```bash
uv sync --extra worker
```

wheel提供实验性命令`aivideo-worker`。systemd示例和安装脚本位于`packaging/linux/`；Token优先从`LoadCredential=worker-token:...`读取。该入口尚未在真实Ubuntu GPU主机完成声明式ComfyUI执行验收，不应描述为生产支持。

客户端入口是`ai_video_generator.workers.remote_worker.RemoteWorkerClient`，运行循环是`RemoteWorkerRuntime`。生产地址默认必须是`https://`；明文HTTP只可通过`allow_insecure_http=True`显式用于回环测试。

## 执行器接口

运行时保留异步`executor(TaskSpec) -> WorkerExecutionOutcome`作为无manifest旧任务的兼容接口。带`workload_manifest_sha256`的新任务由运行时从控制平面下载manifest，校验SHA-256、Schema、任务类型及Worker能力后，交给`workload_executor(TaskSpec, TaskWorkloadManifest)`。执行器负责物化manifest列出的输入blob、把API prompt提交到Worker本机的ComfyUI并对账history；成功产物通过`ProducedArtifact`返回，运行时会先上传所有产物，再提交结果。执行器异常会转换为`worker_executor_error`失败结果。

## Workload manifest契约

控制平面通过`POST /api/v1/workload-manifests`登记规范化JSON，并返回其SHA-256。任务通过可空的`TaskSpec.workload_manifest_sha256`引用它；可空设计保证旧任务和旧数据库兼容。远程Worker通过需要认证的`GET /api/v1/workload-manifests/{sha256}`读取内容。

首版manifest只允许声明式ComfyUI API prompt，格式固定为`avg-comfyui-workload+json-v1`。它包含任务类型、工作流和节点Schema哈希、prompt、内容寻址的输入blob、输出节点以及所需节点/模型/模板能力。manifest不接受命令、Python对象、pickle或脚本入口；输入blob的挂载路径必须是无`..`的相对POSIX路径。

哈希输入是UTF-8 JSON，键排序、无空白、禁止NaN。登记时控制平面验证引用完整性和任务类型；Worker下载后重新计算并验证记录，再核对`node_schema_sha256`以及所需节点、模型和模板。能力不匹配、缺少新执行器或任务类型不符会产生明确失败码，不会退回旧执行器执行。

## 凭据与部署

Token不得写入Git、项目数据库、日志或命令行历史。Linux部署应从systemd credential、内核Keyring或权限受限的环境文件注入。SSH只用于部署、维护或隧道，不参与任务协议。
