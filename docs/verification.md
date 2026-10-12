# 首版 CPU 闭环验收记录

日期：2026-10-12。范围：独立 Python 平台的可信本地、合成数值二分类 CPU 切片。数据、权重、运行数据库和临时测试凭据均位于忽略的 `.runtime/` 下，不进入 Git。没有安装第三方包、使用外部训练服务或执行设备验收。

## 实测环境与版本

- Linux x86_64，`Linux-6.18.44-x86_64-with-glibc2.41`
- Python **3.12.14**，SQLite **3.53.1**；`.python-version` 与 `requirements.lock` 已改为本次实际版本
- `python3 -m venv .venv` 成功；`.venv/bin/python --version` 输出 `Python 3.12.14`
- `fcntl.flock`、`pass_fds`、loopback 端口、独立进程与进程信号均通过实际运行路径验证
- 之前交接文档提及的 Python 3.14.3 是另一环境的历史记录，**没有在本次云端执行**；不把它写作本次通过版本，也不声称完整跨平台兼容

## 执行记录

1. 原始交接代码：`python3 -m unittest discover -s tests -v`，11 项通过，6.311 秒。这只是修复前基线，不代替最终结果。
2. 修复后：`.venv/bin/python -m compileall -q model_platform scripts tests`，退出码 0。
3. 最终：`.venv/bin/python -m unittest discover -s tests -v`，结果见下方最终检查记录。
4. 独立初始化测试运行目录，启动 `.venv/bin/python -m model_platform api --port 0` 和独立 `.venv/bin/python -m model_platform worker`，设置同一个 `MP_HOME` 与实际 loopback 地址 `MP_API`，执行 `.venv/bin/python scripts/demo.py`；退出码 0。
5. 所有演示和测试只绑定当前机器 loopback。演示结束已停止其 deployment，并结束该次 API / Worker，未保留对外服务。

完整演示的真实计算结果（不是测试桩或预设指标）：

```text
training samples: 8
evaluation samples: 4
accuracy: 1.0
log_loss: 0.10742554536851148
prediction [-2.5]: class 0, probability 0.06831385274539217
prediction [ 2.5]: class 1, probability 0.9316861472546077
stop verified: subsequent inference rejected (503)
```

这是易分的合成二分类样本，只证明训练→产物→留出评估→登记→持续推理→停用闭环真实发生。它不证明泛化能力、视觉效果、设备兼容性、低延迟、业务质量或安全控制能力。随机技术 ID 每次不同，指标来自实际代码计算。

## 自动化覆盖

以下测试均使用真实 HTTP、SQLite、训练、文件及独立进程；没有用 mock 训练或 mock serving 替代结果：

- 同租户并发幂等提交、不同输入冲突、不同租户相同 key 独立作用域、缺失鉴权/幂等键
- 未知字段、数据使用确认、两类标签、数值/维数/样本边界、训练与评估特征重叠、重复 JSON 字段、超大 Content-Length 早拒绝
- 实际训练参数变化、从落盘 JSON 重算评估、产物/数据摘要、版本唯一、SQLite 技术记录和 run 输入不可变
- 最大 1,000 行总预算下的 4 行训练 + 996 行留出评估；在线预测 128 行成功、129 行拒绝
- 极小但非恒定特征（约 `1e-200`）不因方差平方下溢被误判为常量
- 全常量训练数据真实失败，保留 attempt / 日志，显式重试最多 3 次，无模型误登记
- 真实 serving 就绪探测、带版本实际预测、维数拒绝、跨租户拒绝、终结实际 PID、停后 503、重复停止
- 4 个部署预算、第五个返回 429、停止后预算可再使用
- API 被 SIGKILL 后重新启动，恢复运行期望、保留已停止状态、运行/模型/队列可恢复、旧孤儿 serving 自行退出
- Worker 被 SIGKILL 后，下次启动标记 interrupted，显式重试新增 attempt 后成功；活跃 Worker 所有权不被第二个 Worker 抢占
- 产物损坏不能 ready；serving 被 SIGKILL 后撤销 ready、拒绝推理
- 32 个活动 run 预算，已存在幂等请求可重放；每 tenant 每分钟速率限制
- 单 API 文件锁；重复 init 不覆盖凭据；临时运行目录 0700、凭据 0600
- 已停止 deployment 历史不被后台反复改写 checked_at
- 在隔离源码副本中先启动 Worker，再修改源码并启动 API；新提交 run 被旧 Worker 拒绝为 `execution_code_changed`，不会将旧导入代码产物误记为新源码

测试清理在 setup 阶段就注册，启动失败时仍清理自己创建的进程、日志句柄和临时目录。

## 审查与修复

人工阅读了所有实现模块、测试、CLI、演示及架构文档，并进行独立只读代码审查，确认本切片没有发现可由 API 越权访问其他 tenant 的路径，也没有暴露任意文件、代码、URL、pickle 或外部模型加载入口。这是当前范围的代码检查，**不是渗透测试或生产安全认证**。

本轮修复了四个具体问题：

1. 合法留出集大于 128 行时，曾错误触发在线推理上限；改为受控分批评估，保留在线 128 行上限。
2. 已停止部署原先每 0.5 秒写一次 SQLite；对账查询排除已完成停止的历史记录，避免不断产生无意义写事务。
3. Worker 原先按当前磁盘文件标记代码摘要，可能与已经导入的旧函数不一致；固定启动时执行环境，并在执行前、登记前核对 run 与当前磁盘摘要。
4. 极小非恒定特征直接平方计算方差会下溢；用稳定欧氏范数计算尺度，并增加真实训练回归。

`README.md` 已替换旧的“只有文档、无实现”和“业务系统主写技术登记”说明，保持与独立技术平台边界一致。接口、终态、恢复方式及未实现能力均按实际代码描述。

## 最终检查记录

- `.venv/bin/python -m compileall -q model_platform scripts tests`：通过，退出码 0
- `.venv/bin/python -m unittest discover -s tests -v`：**16 项通过，Ran 16 tests in 10.849s / OK**
- 全部代码修复后的 `scripts/demo.py` 再次运行：退出码 0，以上指标与预测完全一致，停用后 503，API / Worker 均已结束
- `git diff --check` 及全部 21 个拟提交文本文件的 whitespace 检查通过；新增文件也纳入检查
- AST 导入检查确认所有 Python 模块依赖均来自标准库或本仓库；Markdown 相对文件链接目标全部存在
- 拟提交文件逐一检查，未包含运行数据、权重、数据库、虚拟环境或生成的真实测试凭据值

一次临时 whitespace 检查脚本最初把 `git diff --no-index` 的正常“存在差异”退出码 1 误判为失败；修正脚本的退出码判断后复跑通过，无需修改源文件空白。该辅助脚本检查不冒充第三方静态分析。

## 尚未验证或不支持

- 没有跑 Windows 或多 Python 版本 CI，没有安装/运行第三方 linter、类型检查器、安全扫描器或负载测试器；语法编译与人工审查不能替代它们
- 没有公网/TLS、生产安全、多主机故障、断电/磁盘耗尽、长时间容量/保留、精确硬 CPU/RSS 限额或 GPU 验收
- 请求预算与独立进程是逻辑限制，同一 OS 用户仍可读写文件；不同租户共用单机 Worker 和全局队列预算
- 浮点算法仍受机器精度和支持的模型尺度范围影响；没有承诺任意数量级的数值稳定性
- 取消仍返回 501；没有自动训练重试、检查点、版本热切流、自动回滚或质量发布门槛
- 数据集由调用方提供；样本无特征重叠不等同于已经证明无数据泄漏，评估也不是外部审计
- 没有视觉、无人机、机器人、真实设备、Sluggard 或外部厂商集成验收
- 本记录只描述云端工作区的实测状态，不声明远端提交、PR 或 CI 已完成；远端状态须另行核实
