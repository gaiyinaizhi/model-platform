# 训练实验可信比较

新增鉴权只读 `GET /training-model-comparisons?model_a_id=model_…&model_b_id=model_…`，支持两个不同、当前租户成功登记的 `yolox-nano-dataset/v1` 模型。查询须恰含两个键各一次，不接受请求体、Range、编码路径、重复/未知查询参数。返回 `training-model-comparison/v1`，不创建训练、评估、部署、审计或其他记录。

## 证据与兼容性

在一个数据库读事务中先解析两侧归属，再复用训练报告的任务/attempt/模型/评估/数据绑定与语义验证。响应包含两份原 `training-evaluation-report/v1` 安全投影及其摘要，维持原报告格式。它核对持久化记录内部一致性，不读取原始数据/模型文件，不提供签名或真实性证明。

兼容性按以下稳定原因代码顺序判断：

- `evaluation_content`：有序评估图像/标签内容摘要不同。
- `evaluation_split`、`evaluation_role`：val/test 与 validation-only/held-out-test 不同。
- `classes`：有序类别ID/名称映射不同。
- `evaluation_protocol`：输入/后处理、阈值、NMS、检测上限、COCO参数/实现版本不同。
- `execution_code`、`execution_python`、`execution_platform`、`dependencies`：实际执行条件不同。

代码条件使用现有完整源码摘要，尚无独立评估器源码摘要，因此它是保守限制：仅训练实现变更也可能不可比。提交机Python/platform不作为比较条件，执行环境才是实际评估条件。相同评估内容不因数据集版本ID不同而拒绝；训练集、完整数据集摘要、seed、epochs、learning rate和模型产物可以不同。

兼容返回总体 `map_50`、`map_50_95`、`ar_100` 及逐类 AP50/AP50:95/AR@100 的 B−A 原始数值差；展示层转换为百分点。任一值为 null 时差值仍为 null，真实0保留。不兼容仍返回200、稳定 `differences` 及双方证据，但 `deltas=null`，不生成误导性的数值比较。类别保持原顺序，不用名称猜测重排。相同条件不代表统计显著、质量优胜、官方COCO基准或生产许可。

未知/跨租户任一模型404（在另一侧完整性错误前）；格式错误400；损坏、不支持、非成功证据分别409 `training_comparison_integrity`、`training_comparison_unsupported`、`training_comparison_not_succeeded`。响应最多两倍512KiB加16KiB，规范JSON、no-store、nosniff，重复查询逐字一致。

## 兼容旧客户端

原 `POST /model-comparisons` 保持原响应/错误契约，它仅比较旧评估数据、类别、协议、split，不提供本接口的跨表完整性或执行环境保证。新页面不得回退旧POST；旧接口不是本可信比较的别名。独立COCO比较与训练报告schema不变，无数据库迁移。

## 验证

`python -m unittest discover -s tests -p 'test_training_comparisons.py'` 使用明确合成持久化记录，覆盖相同口径/训练输入可变、执行差异、split/content不兼容、null/0、B−A、双归属优先、证据篡改、单快照只读、严格HTTP与预算。原训练报告回归必须保持通过。真实CPU训练及页面结果另见本轮验收证据，不以单元fixture证明质量。

同一有序评估内容与类别必须具有一致的真值统计（图像数、有/无真值图像数、真值框数、类别数、有真值类别数，以及各有序类别的真值框数），即使执行环境不同也不能矛盾。矛盾返回 `409 training_comparison_integrity`；预测数量允许不同。
