# LLM Provider 联调基线

联调日期：2026-08-15。服务使用OpenAI-compatible协议，凭据只从本地忽略文件或环境变量读取，不能写入数据库、Git、日志或错误响应。

## 已验证配置

```text
Base URL: https://hub.linux.do/v1
Model ID: gemini-3.7-flash
Protocol: POST /chat/completions
Structured output: response_format={"type":"json_object"}
Image input: OpenAI image_url content part (HTTPS URL或data URI)
```

`GET /models`返回200，并明确列出`gemini-3.7-flash`。不要使用自然语言描述“Gemini 3.7 Flash”代替模型ID，也不要自行改成`models/gemini-3.7-flash`，两者在聚合网关中可能代表不同路由。

## 实测结果

| 请求 | 状态 | 耗时 | 结果 |
|---|---:|---:|---|
| `GET /models` | 200 | 约2.1秒 | 2502个条目，包含精确模型ID |
| 32-token JSON | 200 | 约31.9秒 | `finish_reason=length`，正文为空 |
| 256-token JSON | 200 | 约10.1秒 | 标准`chat.completion`，有效JSON |
| 64×64 JPEG视觉请求 | 200 | 约4.8秒 | 正确识别左右红色/蓝色 |
| Harness工作流映射 | 200 | 约10.6秒 | 一次调用通过Schema、节点和字段校验 |

该模型会返回`reasoning`和`reasoning_content`扩展字段，但应用只信任`choices[0].message.content`。极低输出预算可能全部消耗在推理上，因此即使HTTP 200也必须拒绝空正文并记录`finish_reason`。实际工作流映射的Schema远大于联调样例，不能照搬32或256 token上限。

1×1 PNG data URI曾收到HTTP 400 `invalid_request_error`。有效64×64 JPEG通过，因此图片输入应在调用前完成格式、尺寸和解码校验；上游拒绝图片时不得自动降级为纯文本并假装完成视觉分析。

## 运行配置

```powershell
$env:AIVIDEO_LLM_BASE_URL = 'https://hub.linux.do/v1'
$env:AIVIDEO_LLM_MODEL = 'gemini-3.7-flash'
$env:AIVIDEO_LLM_API_KEY = '<从安全存储读取>'
$env:AIVIDEO_LLM_TIMEOUT_SECONDS = '120'
```

生产桌面端应把密钥迁移到Windows Credential Manager。仓库根目录的`api-key.txt`仅用于当前本地联调且必须保持Git忽略；任何诊断只能记录文件存在/非空，不能输出内容、Authorization header或请求转储。
