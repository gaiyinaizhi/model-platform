# 失败训练有条件重试

2026-10-20。此增量按已评审契约实施，仅扩展现有训练 retry；不涉及独立评估重试、运行中取消、自动重试、检查点恢复、依赖安装、部署或真实运行目录变更。

## 请求与兼容

鉴权 `POST /runs/{run_id}/retry` 接受：

```json
{"expected_attempt_id":"attempt_0123456789abcdef0123456789abcdef"}
```

ID 必须是 `attempt_` 加 32 位小写十六进制。只接受这一字段；null、非对象、重复字段、额外字段、错误类型和格式均拒绝。用户身份先于对象归属；无权或缺失对象返回 404，不通过错误暴露他人 attempt。

旧请求体 `{}` 及内部 `Store.retry(tenant, run_id)` 保留原有状态、容量、三次上限和响应语义，但没有失败代际防重或新源码前置检查。新交互不能回退为 `{}`。`GET /health` 增加 `conditional_training_retry_supported: true`；缺失表示客户端不能使用新能力，非布尔值为无效能力响应。`cancel_supported=false` 和 `queued_cancel_supported=true` 不变。

## 单一写事务

有条件路径在现有 `BEGIN IMMEDIATE` 事务内依次：

1. 取得当前租户的 run。
2. 要求 failed、无 model、非空且全部 failed 的 attempt 历史，序号严格为 1..N，N < 3，并且最新 attempt ID 等于请求值。不满足则 `409 retry_conflict`，无任何写入。
3. 比较 run 的不可变 `environment.code_sha256` 与当前源代码摘要。不符则 `409 retry_code_changed`，不排队、不消费 attempt、不重写环境或任何血缘；需要新代码时由操作者显式新建 run。
4. 检查现有训练与独立评估共享的 32 个 queued/running 逻辑槽位。满载则 `429 run_capacity`，无任何写入。
5. 只更新原 run 的 status=queued、updated_at，并追加一条现有 retry audit。Worker 领取时才创建下一个 attempt。

原始 run ID、请求摘要、数据/配置、环境、数据集版本绑定、全部失败 attempt/日志及原始队列创建顺序均保留。没有新表、schema 版本、迁移、幂等键缓存或新的 run。并发 retry、claim、cancel、recovery 通过现有 SQLite 写事务串行化。

源码检查只是一道提前拒绝，不替代 Worker 对启动时已加载代码和执行时磁盘代码的核验，也不保证任务成功。若旧失败原因为 `execution_code_changed`，恢复正确原始源码并重启相关进程后仍可显式重试；本操作本身不会修复或重启陈旧 Worker。

## 响应、重放与不确定结果

受理仍返回 `202` 与正常 run 详情。事务提交后的详情读取可能看到 queued、running、failed、succeeded 或 cancelled；客户端不能硬性要求 queued，也不能覆盖真实终态。

这是一条失败代际前置条件，不是相同响应重放缓存。保证的是同一已确认失败最多受理一次排队，不是网络层恰好发送一次：浏览器或传输层可能重传同一 POST，但旧 attempt 前置条件不能因此获得额外执行。

- queued/running 时重发同一请求为 409。
- 新 attempt 快速失败后，旧 `expected_attempt_id` 仍为 409，不会误排第三次执行。
- 新失败必须明确展示给操作者，再确认其新 attempt ID。
- 接受 retry 后又被取消，旧请求不能撤销 cancelled 终态。
- 网络中断、超时、无效响应或 409 后，应用只读刷新同一 run；不得声称“没有受理”，不得由应用自动重发，更不得刷新后自动使用新 ID 再 POST。

新 BFF/UI 必须绑定原 run/recipe/attempt 身份、校验连贯的 attempt 和状态/模型关系，并隔离导航或过时响应。受理响应须保留已确认失败的 ID、序号、failed 状态、错误码和完成时间；历史最终日志可能因失败日志的稍后追加而增加，不能要求日志逐字节冻结。独立评估、模型来源和既有 serving 不受影响。

## 范围与验证

仅可信本地 CPU 开发和现有逻辑预算，不是生产多租户沙箱或质量许可。测试覆盖实际 SQLite/loopback HTTP、丢失已受理响应、快速再失败、并发竞争、schema/不可变历史保持和旧客户端兼容，具体实际结果见[专项验收](conditional-training-retry-verification.md)。没有以合成状态/图像测试声称视觉训练质量或真实数据端到端验收。
