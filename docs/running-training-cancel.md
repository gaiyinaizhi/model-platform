# 可确认的运行中训练取消

此增量适用于所有已有静态训练配方（含固定 Cat 和自定义数据集训练），不适用于独立评估或 serving。保留排队撤销、原始输入、失败历史和产物；不增加自动重试、检查点或删除操作。未发布或部署本增量不代表已修改任何运行数据库。

## API 与状态

- `POST /runs/{id}/cancel` 仍只接收 `{}`，只撤销 queued 请求，不创建 attempt。
- `POST /runs/{id}/cancel-running` 必须精确接收 `{"expected_attempt_id":"attempt_<32 位小写十六进制>"}`，无额外字段。先校验归属再校验 body。不存在/跨租户返回 404，非法 body 返回 400 invalid_input；状态、当前 attempt 或历史不匹配返回 409 cancel_conflict。
- `GET /runs/{id}` 和新取消响应新增 `cancellation: null`，或 `{attempt_id,status,requested_at,confirmed_at}`；时间为 epoch 秒，status 为 requested/confirmed。不暴露 PID 或内部确认依据。
- 新请求/同 attempt 待确认重放返回 202：run 和 attempt 仍为 running，model_version_id 与 confirmed_at 为 null。同 attempt 已确认重放返回 200：run/attempt 为 cancelled，confirmed_at 非空，error_code 为 null。两种成功响应在修改事务内捕获完整 run 投影，不用事务后 GET 猜测结果。
- `/health` 新增 `running_cancel_supported:true`，保留 `cancel_supported:false` 和 `queued_cancel_supported:true`。客户端只能在明确支持时提供操作，不能把 202 乐观显示为 cancelled，也不能在网络结果不明时自动重发。
- pending 仍占用 active 预算；cancelled 不可 retry。先前失败 attempt 原样保留，取消不创建模型或评估。

## 事务与进程证明

请求、登记和失败以 SQLite BEGIN IMMEDIATE 仲裁。登记或失败先提交时取消为 409；意图先提交时禁止发布。失败提交必须绑定 run 的当前 running attempt，旧 attempt 不能修改新 attempt 对应 run。意图、确认、run/attempt 终态、确认日志和审计在各自事务内提交。

监督者在 spawn 前、读循环、EOF 后等待退出时检查意图，登记事务再次拒绝待取消记录。只向本调用实际持有的 Popen 子进程发送 kill，然后有界 wait/reap；不会用持久化 PID 发信号。无子进程的确认来自本次执行确实尚未启动的生命周期；不能因没有 PID 就假定 never_started。

每次清理等待上限为 5 秒，这是调用等待预算而非 OS 终止保证。无法确认 kill/reap，或 Popen 构造结果不明，抛出绕过普通失败处理的专用 BaseException：Worker 退出、run 保持 pending、不发布、不继续领取。继承的 close-only Worker 租约在旧子进程仍持有时不会释放。父 SIGKILL 后，新 Worker 只有成功取得租约才可恢复：有意图的旧 attempt 确认为 cancelled/lease_recovered，其余 interrupted/failed。独立评估的恢复保证未扩展。

## 显式离线迁移

新空数据库直接初始化 `running-training-cancel/v1`。旧数据库启动前先核验完整 schema、索引、触发器与 marker；不自动修复/迁移部分状态。既有 queued-training-cancel/v1 是前置要求。

1. 停止 API、Worker、所有 serving。核实正确 MP_HOME；先完成已有 imported-origin 与 queued-cancel 迁移（若需要）。
2. 使用新版本执行 `MP_HOME=/绝对路径 python -m model_platform.migrate_running_cancel`。
3. 迁移取得 API/Worker/model-import 排他锁，探测 serving 已停止，不发送终止信号；用 O_EXCL/NOFOLLOW 建立 `before-running-training-cancel.sqlite3`，已存在则拒绝覆盖。事务重建 attempts 仅放宽 CHECK 为 cancelled，保留 rowid、原列、唯一约束和所有历史/外键/自定义索引触发器/其他 schema。
4. 新增不可删除的 training_cancellations：attempt PK、run UNIQUE、时间及内部 child_reaped/never_started/lease_recovered 依据。插入必须匹配 run/attempt；身份和请求时间不可更改，确认只能从未确认转为确认一次。完整性/FK 验证失败事务回滚并保留备份；已完成迁移为 no-op。
5. 重新启动新版本服务。迁移不会重写历史代码指纹；旧排队/失败请求遇到代码变化仍按现有规则处理。

回滚必须先停止全部服务，只能在接受丢弃备份后所有写入的前提下恢复迁移前备份。不要用旧二进制打开已迁移数据库，不提供能保留新增取消历史的伪降级。

## 验证边界

`python -m unittest discover -s tests -p 'test_running_cancel.py' -v` 覆盖事务顺序/并发、归属/重放/严格 body、旧 attempt fencing、无子进程、真实 spawn 空档/读循环/结果先于退出/EOF、kill/reap 不确定、父 SIGKILL/继承租约与下一项真实数值 CPU 训练。`test_running_cancel_migration.py` 覆盖隔离临时 DB 迁移。不把 fake child 或合成数据当成识别质量证据；真实 CPU 检测训练与产品 UI 验收独立记录。完整回归和可选 ONNX 结果仅在实际执行后记录。
