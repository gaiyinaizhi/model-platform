# 取消尚未领取的训练任务

2026-10-20，已评审契约后的本地增量；提交、合并与生产部署另行处理。

本功能仅撤销 TrainingRun 当前排队的执行，不中断运行中的训练，不取消独立评估，不删除数据或历史，不创建 checkpoint，也不扩大重试次数。

## HTTP 契约

使用原有 Bearer 鉴权，`POST /runs/{id}/cancel` 的 JSON 请求体必须为 `{}`。租户由身份决定，先检查归属；不存在与其他租户的记录均返回404。空对象以外的字段、重复 JSON 字段与原有传输限制继续拒绝。

- queued → cancelled：200，返回普通 run 详情。只改变状态、updated_at，增加一条 cancel_queued 审计。
- 已 cancelled：200，无时间戳或审计重复写入。
- running：409 / cancel_conflict，明确不支持中断已运行的训练，不发送进程信号。
- succeeded、failed：409 / cancel_conflict，不改变状态。

取消与 Worker 的领取、原有 retry 都在 BEGIN IMMEDIATE 事务内串行判定。Worker 先领取时取消失败；取消先提交时 Worker 不会再领取此 run。不在数据库提交前报告成功。

cancelled 是新的训练终态，不是 attempt 状态。未领取过的 run 没有 attempt；失败后已经重排的 run 可以取消当前排队，但原失败 attempt、日志、输入摘要、数据绑定与代码环境不变。取消不产生模型、评估或产物。

取消后的记录仍在原租户历史中，queue.position=null、model_version_id=null，不计入排队/运行容量。原提交幂等键重放返回同一个 cancelled 记录；retry 拒绝 cancelled。重新运行必须由操作者显式使用新的提交键，不能因超时或不确定结果自动提交另一条任务。

GET /health 保留 cancel_supported=false，另增 queued_cancel_supported=true。前者表示不支持通用/运行中取消，后者只表示排队撤销。旧后端缺少新字段时客户端必须按不可用处理。训练状态增加了 cancelled，严格枚举旧客户端需要同步升级；独立评估状态不变。

## 旧运行目录必须显式离线迁移

新运行目录直接创建第五状态及 queued-training-cancel/v1 标记。旧目录启动 API/Worker 会在写入 cursor 密钥或执行 schema 脚本前，明确要求迁移；GET 不会偷偷升级数据库。

1. 停止原 API、Worker、模型导入操作及其 serving 进程。不要在仍运行的进程下替换代码或运行迁移。
2. 如目录仍是早期训练型 models schema，先使用原 `python -m model_platform.migrate_imports` 完成 imported-origin 迁移。
3. 保持指向现有运行目录的 MP_HOME，从新源码根运行：

```sh
python -m model_platform.migrate_queued_cancel
```

命令尝试持有 api、worker、model-import 三把现有锁；任一锁被占用则失败，不停服务。对部署记录中的 PID 只做保守存活探测，不杀进程；活着或无法确认的 serving 记录会阻止迁移。

先用 SQLite backup API 独占创建 before-queued-training-cancel.sqlite3，权限0600，不覆盖旧备份、不跟随同名符号链接。然后在一个事务内仅重建 runs CHECK，保留原 rowid、全部字段值、唯一约束、索引、触发器、外键、数据绑定、attempt、日志、模型与独立评估。检查全行/其他 schema 一致性、foreign_key_check、integrity_check，最后原子写入迁移标记。异常回滚，备份保留。失败后同名备份存在会阻止盲目重试，需要操作者先检查失败与备份，不能由脚本自动覆盖。

迁移不会把崩溃遗留的 running 行改成 cancelled，也不会伪造完成。取得全部排他锁且排除 live serving 后，遗留 running 记录原样保留，随后正常 Worker 启动按既有规则恢复为 interrupted/failed，再领取工作。不要把迁移命令当作恢复/执行命令。

重复迁移校验完整当前 schema、必需索引/触发器、标记与数据库完整性后返回 already current。部分升级、缺少标记、被篡改的约束/必需触发器均拒绝，不猜测修复。

### 旧排队任务与代码指纹

本平台把全部后端 Python 源码纳入提交代码指纹。迁移不重写已排队任务的历史环境；升级前排队的任务被新 Worker 领取时仍可能失败为 execution_code_changed。可在启动 Worker 前取消这些旧排队记录，再由操作者显式新建任务。失败/运行历史必须保留，不允许为了执行而改写其代码指纹。

### 回退边界

回退是停服务后由操作者恢复迁移前备份，不是删除 cancelled 历史的反向 schema 改写。恢复会丢失备份之后的全部更改，不自动执行，不宣称新工作后仍可无损回退。应保留当前目录的安全副本并评估所有新增记录后再操作。代码与数据库必须配套，旧严格状态客户端不能直接使用新 cancelled 记录。

## 验证范围

新增测试在隔离临时 SQLite 和 loopback HTTP 中验证鉴权、状态、幂等、审计一次性、队列容量、多进程领取/取消/重试竞争、旧 schema 备份/迁移/回退副本、完整性和中断恢复。使用内存固定测试身份与 cursor 材料，不新建凭据文件，不迁移真实保留的运行目录，不训练或下载模型。最终命令和计数另记验收报告；不把 fixture 验证称为生产升级或真实模型质量验证。
