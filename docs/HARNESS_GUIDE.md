# Harness 文件与生效范围

内置 Harness 使用 UTF-8 Markdown 保存，位于：

```text
src/ai_video_generator/resources/harnesses/
```

这些文件在每次对应的 LLM 请求开始时读取。修改 Markdown 后无需重新编译前端，下一次请求即使用新内容；已经创建并锁定的任务不会被追溯修改。

## 项目最高指令

每个项目的“项目配置”中都有“最高指令”文本框。内容保存在项目 workspace 的
`highestInstruction` 字段中，并在该项目的 LLM 请求中先于对应 Harness 发送。生效
顺序是：项目最高指令、阶段 Harness、当前任务数据。

最高指令用于声明贯穿项目的创作要求和限制，但不会改变阶段输出契约：规划类任务
仍返回约定的 JSON，图片与 H3 视频提示词仍只返回提示词正文。修改后需要保存项目并
重新生成相应内容；已经创建并锁定的任务不会被追溯修改。

## 内置 Harness

| 文件 | 生效位置 | 何时调用 |
| --- | --- | --- |
| `project-lead.md` | 项目负责人 | 创意、大纲、分镜、素材规划和项目负责人对话的结构化修改。Motion Context 分段与一次性场景/物品规则在这里。 |
| `workflow-mapping.md` | 工作流 LLM 标定 | 用户上传图片、视频后处理或 H3 编码/扩散工作流后，将节点输入映射到标准字段。它不参与内容创作。 |
| `image-default.md` | 图片提示词默认 Harness | 图片工作流没有登记专用 Harness 时使用。已绑定 Harness revision 时，revision Markdown 替代此文件。 |
| `image-writer.md` | 图片提示词输出规则 | 每次 AI 编写图片提示词时追加，限定只输出进入文本框的提示词正文。 |
| `h3-direct-writer.md` | 当前 H3 视频提示词写入 | 追加在官方规范和对应模式 Writer 之后，要求只输出最终提示词并限制素材引用范围。 |
| `h3-staged-system.md` | H3 分阶段运行器 | Director、Planner、Writer、Reviewer 分阶段链的公共规则。当前直接写入路径不调用此文件。 |
| `h3-reviewer-visual.md` | H3 分阶段 Reviewer | 只在分阶段 Reviewer 中追加，不进入直接 H3 提示词请求。 |
| `h3-translation.md` | H3 中文对照 | 用户生成中文对照时使用；翻译不参与视频执行。 |

## 官方与社区 H3 文档

H3 模式规范没有硬编码在 Python 中。安装后保存在运行数据目录：

```text
<data_root>/harness-sources/official/<commit>/
<data_root>/harness-sources/community/<commit>/
```

| 来源文件 | 生效模式或阶段 |
| --- | --- |
| `official/skills/h3-prompt-writing/SKILL.md` | 所有 H3 请求的公共官方规范。 |
| `official/.../references/base-en.txt` | T2VA、I2VA、FL2VA、L2VA。 |
| `official/.../references/ref-en.txt` | Ref2VA。 |
| `community/.../minimax-h3-text-video-prompt/SKILL.md` | T2VA Writer。 |
| `community/.../minimax-h3-keyframe-video-prompt/SKILL.md` | I2VA、FL2VA、L2VA Writer。 |
| `community/.../minimax-h3-reference-video-prompt/SKILL.md` | Ref2VA Writer。 |
| `community/.../minimax-h3-creative-director/SKILL.md` | 分阶段 Director。 |
| `community/.../minimax-h3-multishot-planner/SKILL.md` | 多镜头 Planner。 |
| `community/.../minimax-h3-prompt-reviewer/SKILL.md` | 分阶段 Reviewer。 |

组装 H3 Harness 时，这些来源文件会复制进数据库中的不可变 revision 快照。项目生成 H3 提示词时读取已批准 revision，而不是重新读取安装目录。因此：

1. 修改 `harness-sources` 下的文件不会改变现有批准 revision。
2. 修改后需要在 Harness 页面重新组装并批准，才会生成新的活动 revision。
3. 已保存提示词和已编译任务仍保持原 revision；重新编写提示词并重新创建计划后才使用新 revision。

## 修改约束

- 只修改自然语言规则，不修改 JSON Schema 字段名、workspace 字段名或 ComfyUI binding semantic。
- 保持 `project-lead.md` 中 `motionSegments`、`shotIds` 等正式字段拼写不变。
- H3 模式规则优先修改对应官方/社区来源文件并重新组装，不要重复写入多个内置文件。
- Markdown 文件不得为空；无法读取或内容为空时，请求会报告具体文件路径。
