# 独立 Linux CPU Worker 部署与排障

本指南针对可信本地操作者、Linux x86_64 / **CPython 3.12.14**。这是单机 CPU 开发运行方式，不是公网服务或生产多租户部署方案。API、训练 Worker 与推理子进程分离；API 接受训练任务并不表示已经有 Worker 执行。无需 Sluggard、业务数据库、GPU 或 CUDA。

## 1. 固定代码、解释器与依赖

从仓库根目录运行，先确认 `python3 --version` 为 3.12.14；不匹配时先由操作者准备匹配的解释器，不要将其他版本的执行结果称为本平台锁定环境的验证。

```sh
python3 --version
python3 -m venv .venv-cpu
.venv-cpu/bin/python --version
```

按配方选择依赖：

- 二分类与 UCI Digits：`requirements.lock` 声明标准库核心，无第三方依赖，不需要 pip 安装
- Fashion CNN：`.venv-cpu/bin/python -m pip install -r requirements-fashion.lock`
- YOLOX Cat 或自定义 YOLO 数据集训练：使用下面的完整检测 CPU 栈；它已递归包含 Fashion 及数据集校验依赖，不需要逐个手工安装 torch 等包
- 仅导入/校验数据集：`requirements-datasets.lock` 只提供 Pillow/PyYAML，不足以训练检测模型

完整检测 CPU 环境的安装命令：

```sh
.venv-cpu/bin/python -m pip install -r requirements-detection.lock
.venv-cpu/bin/python -m pip check
```

锁文件限定 Linux x86_64 / CPython 3.12 的已选 wheel 及 SHA-256，CPU PyTorch 来自官方 CPU wheel，其他依赖来自 PyPI；不等于 ARM、Windows 或所有 Python 版本兼容。不要取消哈希校验、替换 CUDA wheel 或自行升级单个依赖以规避安装失败。安装命令供操作者使用，本指南存在不代表已在目标主机执行或安装成功。

API、Worker 和 API 启动的 serving 子进程应使用**同一个虚拟环境、同一份冻结源码**。可选依赖实际在 Worker 进程中检查，API 进程能导入模块不能代替 Worker 就绪证据。运行或排队期间不要修改代码、安装/升级依赖；改变环境前安排停止，随后使用匹配环境重新启动。提交时源码与执行时源码不同会导致 `execution_code_changed`，不能通过编辑历史 run 或摘要修复。

## 2. 运行目录与本地身份

所有终端先从**同一仓库根目录**设置同一个绝对路径：

```sh
export MP_HOME="$PWD/.runtime/cpu-local"
```

`MP_HOME` 是共享 SQLite 队列、数据快照、模型产物和身份所在的私有目录。不要让 API 和 Worker 分别使用默认目录与另一个自定义目录；相同相对字符串在不同工作目录下也会指向不同数据库。

**下面仅是操作者自行执行的一次性初始化步骤。它会生成本地访问凭据，不能作为普通诊断命令自动执行：**

```sh
.venv-cpu/bin/python -m model_platform init --tenant demo --tenant other
```

已有运行目录应保留原身份，不要重新初始化、覆盖或复制凭据。`init` 不覆盖已存在的 `credentials.json`。新运行目录权限为 0700、凭据为 0600；不要使用共享/公开目录。自定义路径只有位于 `.runtime/` 内才自动获得本仓库的 Git 忽略保护；不要提交数据库、数据、模型或凭据。不要把 token 粘贴进聊天、文档、URL、shell 参数或日志。

## 3. 启动独立进程

终端 A，设置上述相同 `MP_HOME` 后：

```sh
.venv-cpu/bin/python -m model_platform api --port 8080
```

终端 B，设置上述相同 `MP_HOME` 后：

```sh
.venv-cpu/bin/python -m model_platform worker
```

API 固定监听 `127.0.0.1`。**不要修改绑定地址，也不要把开发 HTTP API 反向代理到公网。** 一份运行目录限一个 API 和一个 Worker；不要启动第二个 Worker 来绕过锁或争抢同一队列。Worker 顺序领取任务，训练期间其他任务排队属于正常现象。

`worker --once` 只做启动恢复并**至多领取一个任务**，之后退出；它不是守护进程，也不会持续消费队列。持续工作应使用上面的无 `--once` 命令。

`Ctrl+C` 停止各自终端进程。中断的 attempt 在下次 Worker 启动后登记为 failed，保留历史；不会自动恢复 checkpoint 或自动重训。排队中的其他任务仍可领取。API 正常退出结束自己的 serving 子进程，但未 stop 的 deployment 保留 `desired=running`，重启 API 后会按原版本重新加载并实际探测。

## 4. 就绪检查与任务状态

鉴权 `GET /health` 只说明 API 可以响应。`GET /recipes` 是静态配方发现，不承诺安装了可选依赖。鉴权 `GET /worker-readiness` 用于观察同一运行目录中 Worker 的心跳及实际依赖检查、忙闲状态和调用者自己的排队/运行计数；它不返回其他租户的 run ID 或任务明细。

响应字段：

- `worker.state`：`absent`（本运行目录无记录）、`idle`、`busy`、`stopped` 或 `stale`
- `worker.fresh`、`checked_at`、`stale_after_seconds`：Worker 每 2 秒更新心跳，超过 15 秒视为陈旧；时间字段为 Unix 秒
- `worker.ready`：新鲜且状态为 idle/busy 的 Worker 存活观察，**不包含每个配方的依赖检查结论**；`accepting_immediately` 仅在 idle 时为 true
- `recipes[]`：每项含 `id`、`ready`、`dependencies`（name/available/version/reason）、`reasons` 和 `data_status: not_checked`。配方 ready 同时要求 Worker 新鲜存活以及所需依赖的启动时真实导入成功
- `queue`：调用者自己任务的 `queued`、`running` 计数；不会返回其他租户数量、任务 ID 或运行路径

`--once`、正常返回和 Ctrl+C 会写入 stopped；SIGTERM/SIGKILL 等突然终止可能保留最后一次记录，随后过期为 stale，不能宣称立即检测到退出。依赖只在当前 Worker 启动时探测；环境变更后应重启，不把旧探测当作新环境事实。

就绪信息是时间点观察，不保证下一次任务一定成功，也不证明依赖版本互相兼容、数据缓存、模型质量、设备兼容或生产准入。CPU 依赖就绪不等于 GPU 支持；空闲 Worker 不等于 serving ready。

排障时先读取 readiness，再查询自己 `GET /runs/{id}` 的 status、attempt 和稳定错误码：

| 现象 | 含义与操作 |
| --- | --- |
| API 正常但任务持续 queued；没有 Worker 心跳 | 确认终端 B 启动了持续 Worker；检查相同绝对 `MP_HOME`、同一 checkout 和虚拟环境。另一目录中运行的 Worker 无法领取这里的任务 |
| Worker 心跳陈旧或停止 | 检查该 Worker 进程是否退出/中断。不要只根据旧的就绪记录认定存活；恢复环境后重新启动同一运行目录的 Worker |
| Worker 忙，自己的任务 queued | 单 Worker 串行执行；等待已有任务完成。共享机器上忙态可能来自其他租户，但接口不披露对方任务标识 |
| 已领取任务，随后 failed / `optional_dependencies_missing` | Worker 已经消费队列，但执行环境缺少 torch、ONNX 或其他配方依赖；这与“没有 Worker 所以一直 queued”不同。按对应锁文件修复并重启，检查原 attempt 历史 |
| `execution_code_changed` | 提交与执行的源码不一致。冻结代码、重启 API/Worker，使用新幂等键提交新 run；不要覆盖旧记录 |
| `dataset_missing` / `dataset_corrupt` | 依赖可用不保证数据可用。按对应数据指南准备/校验合法缓存或不可变数据集版本；Worker 不会自动联网补下载或静默绕过校验 |
| `GET /runs/{id}` 返回 404 | 确认使用该对象所属租户的本地身份、正确 API 和运行目录；404 也可能是不存在，不能据此探测其他租户 |

修复失败原因后可由操作者显式调用 `POST /runs/{id}/retry`、请求体 `{"expected_attempt_id":"attempt_<最新失败记录的32位小写hex>"}`；仅完整失败历史且未达 3 次上限可重排，保留原 run、输入、环境和历史。接口能力由 `GET /health` 的 `conditional_training_retry_supported=true` 声明。当前源码摘要不符返回 `409 retry_code_changed`；需要改变代码或输入时应显式创建新 run，不能覆盖旧血缘。若恢复了原始源码且停止/重启 API 和 Worker，过去的 `execution_code_changed` 错误本身不永久禁止重试。

旧请求体 `{}` 继续兼容，但不防止响应丢失后、快速再失败时重复重排。新客户端必须使用失败 attempt 前置条件；网络错误或 `409` 后只读刷新同一 run，不自动重发或偷偷采用新的失败 ID。受理 `202` 的详情可能已进入 running 或终态，不保证仍为 queued。详见[契约](conditional-training-retry.md)。只读 readiness 不会重排、领取或执行任何任务。

## 5. 验证范围

可先用现有标准库合成示例验证本地生命周期：

```sh
.venv-cpu/bin/python scripts/demo.py
```

脚本从 `MP_HOME/credentials.json` 读取 demo 身份，不打印 token。默认 API 为 `http://127.0.0.1:8080`；其他本地端口可设置 `MP_API`。该脚本会提交真实合成训练、部署、预测并 stop，不是只读健康检查；其成功不能证明 CNN/检测依赖就绪或业务数据质量。

可选真实配方的数据准备与演示分别见 [Fashion](fashion.md)、[Cat 检测](detection.md) 和 [自定义数据集训练](dataset-training.md)。ONNX 下载交付见 [端侧交付](edge-delivery.md)。目标主机验收应另记录实际命令、环境、通过/失败/跳过；本文不代替执行证据。

当前没有 cgroup、CPU/RSS 硬配额、独立 OS 用户隔离、分布式 Worker、自动重试或训练取消。进程分离、队列逻辑预算与 tenant 对象隔离不能称为生产多租户安全沙箱。

## 6. 忙态原因与排队位置（2026-10-15 增补）

`worker.state=busy` 时，`worker.ready=true`、`accepting_immediately=false`，
`worker.reasons=["worker_busy"]`。这是单 Worker 正在执行的提示，不是故障；
`recipes[].ready` 仍独立按存活及依赖检查，不会因忙态变为 false。
过期/停止/缺席分别使用 `worker_stale` / `worker_stopped` / `worker_absent`，
不沿用过期的 busy 作为实时结论。检查配方依赖失败请看该配方的 reasons。

`GET /runs/{id}` 新增 `queue: {"scope":"tenant", "position":1}`。
position 从 1 开始，只表示当前租户的 queued 任务中的相对顺序；
非 queued（包括 running、failed、succeeded）为 null。
运行中任务不计入位置，其他租户的任务不计入位置，也不披露它们的 ID、数量或时间。
因此 position=1 不代表全局下一项，更不是等待时长或完成承诺。
界面建议显示“本租户排队第 1；Worker 忙”，不要显示“全局第 1”。
原 readiness.queue 的 queued/running 计数及租户范围保持不变。

领取顺序明确为原始 `created_at` 升序，相同时间按 `id` 升序；位置与领取使用同一排序。
显式 retry 保留原始创建优先级，不按重试时间移至队尾。位置只是一次数据库快照，
后续领取、提交或重试可能改变位置；没有 ETA。读取 run 的状态和位置使用同一快照。
