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

## 真实 UCI 数字十分类增量验收（2026-10-12）

在相同 Linux / Python 3.12.14 标准库环境实际运行；旧合成二分类配方、JSON 权重格式与回归测试保留。数据出处、CC BY 4.0、下载摘要、固定配置、完整命令及报告数值见 [digits.md](digits.md)。

- `.venv/bin/python -m compileall -q model_platform scripts tests` 通过
- `.venv/bin/python -m unittest discover -s tests -v` 实际 **42 tests，OK**：16 个原有回归、22 个十分类契约/训练/HTTP/生命周期测试、4 个离线数据读取/校验/审计/重定向测试
- 新增覆盖64维整数像素、十类全覆盖、各split/epoch/seed/学习率预算、未知配方、非法来源声明、模型格式及数值检查、重载评估数学重算、确定性产物、旧二分类兼容、失败重试、worker中断/API恢复、服务崩溃/坏产物/停用；JSON数值溢出±1e999均为400，不泄漏500
- 官方全量 **3823训练 + 1797最终测试** 实际运行两次，固定40epoch/0.03/seed1729；无超参数搜索、test早停或test驱动选择。两次进程启动、run/model ID不同，JSON模型SHA256和accuracy/macro F1/log loss/confusion完全一致
- 主指标：accuracy **0.9499165275459098（1707/1797）**，macro F1 **0.9501111995181309**，log loss **0.15636690333762782**；完整10×10混淆矩阵见digits文档和本地报告
- 官方数据内部、跨train/test的精确特征重复均0；**没有删除、移动、重抽样任何记录**。API不会独立认证调用方来源，完整演示通过固定官方字节摘要认证下载
- 实际登记、ready概率探测、每轮10次连续HTTP调用（每批10行）、版本/概率/重复结果一致性、显式stop后HTTP503均通过；两轮演示创建的API、Worker、serving进程均已停止
- 训练第一次9.9094秒、重复10.9199秒；HTTP每10行批次中位数第一次5.8853毫秒、重复6.3808毫秒。这是共享CPU的本次观察，包含HTTP鉴权/探测/序列化开销，不是独立性能基准

测试小数据明确是合成契约夹具；真实数据质量证据仅来自以上官方全量演示。无公网部署、真实业务接入、生产隔离、原图视觉网络、ONNX、GPU或设备验收。数据、权重、原始日志、数据库、凭据和JSON运行报告均位于被Git忽略的`.runtime/`，不提交。

## Fashion-MNIST / CNN / ONNX 增量验收（2026-10-12）

以上“无ONNX/视觉网络”描述仅对应之前数值配方阶段。新增固定公开Fashion-MNIST灰度小CNN的真实闭环已经通过，完整数据划分、实际指标、依赖、预算和限制见[fashion.md](fashion.md)。可提交的[fashion-verification.json](fashion-verification.json)仅含脱敏技术元数据、聚合评估/校验和，不含图像、权重、凭据或原始日志。

- `.venv-fashion/bin/python -m compileall -q model_platform scripts tests` 通过
- `.venv-fashion/bin/python -m unittest discover -s tests -v`：75 tests，全部通过，22.922秒；原42项回归+17项图像/ONNX/真实HTTP生命周期+16项数据来源/解压/预算/目录安全
- `.venv/bin/python -m unittest discover -s tests -v`（原零依赖venv）：75 tests，64通过/11可选依赖跳过，16.066秒；`python -m unittest discover -s tests -v`系统环境：65通过/10跳过，17.438秒
- 官方6000分层训练、完整10000 test；accuracy0.8076，macro-F1 0.8048124060，log loss0.5301679883；全test双后端类别一致100%，最大绝对logit差1.335144e-5
- 真实模型持续HTTP、版本关联、停止503、API崩溃恢复和孤儿退出、隔离副本权重损坏拒绝均通过。只有合成测试在构造模型权重，真实指标来自官方数据实跑
- 同次回归完整UCI3823/1797及旧二分类闭环均通过
- 全21个实际安装依赖版本与hash lock一致，21个官方wheel共252,708,049字节下载hash验证通过；pip check通过，torch cuda=None
- 核心源码SHA256 `dd74a334d6d165edf4c5f4e8eb0f4e287a09f866acb6deea4392d32156da193f` 与真实训练报告一致；测试/文档后续编辑不改模型核心源码
- 没有测试外部部署、GPU/设备兼容性、硬资源沙箱或生产多租户；该阶段不新增这些承诺

## 静态配方注册与 CPU CI 回归（2026-10-12）

协议见 [recipes.md](recipes.md)，工作流与本地逐步验证见 [ci.md](ci.md)，精简可机器读取证据见 [registry-verification.json](registry-verification.json)。没有新增公开发现/插件上传接口；三套现有 HTTP 契约不变。

最终模型执行源码摘要为 `e77573939e0850d00a34892d9f72b752c63f6d16b419ff8a1999f94c3d9f1515`；下列最终测试与三条真实 demo 均在该摘要冻结后执行，未在 Worker 运行时修改源码。

- `.venv/bin/python -m compileall -q model_platform scripts tests`：通过
- `.venv/bin/python -m unittest discover -s tests -v`：91 项，80 通过、11 预期 skip（10 个可选 ONNX、1 个需 NumPy/官方缓存），无失败，18.323 秒
- `.venv-fashion/bin/python -m unittest discover -s tests -v`：91/91 通过、零 skip，24.001 秒；此完整本地执行具有已核验官方数据缓存
- 工作流的可选 ONNX 无 skip runner：17/17 通过，5.635 秒，不下载数据、不训练 CNN
- 新增 `test_recipes.py` 16 项：静态注册唯一性/非法注册、冻结对象、目录副本、未知/非字符串/path recipe HTTP 400、旧请求原始摘要与幂等、各配方预算、无 ML 导入、缺依赖失败 → 显式 retry → 中断恢复 → attempt 上限；缺可选库后基础配方仍真实训练成功
- 工作流全部 shell 步骤还在新建隔离 venv 中执行；现有 21 项可选依赖重新下载并通过 hash 安装、`pip check` 和 CPU import 检查。负向验证确保无库/skip不会变为可选 job 的假通过。Action SHA 已核对官方 release commit，YAML/shell/权限结构检查通过

### 真实数据与产物兼容

使用新私有运行目录，复制已有公开缓存并按原脚本重新校验固定官方字节摘要；没有新外部数据下载。启动 API 和独立 Worker，实际依次运行 `scripts/demo.py`、`scripts/demo_digits.py`、`scripts/demo_fashion.py`，三条均成功。Fashion 本次确实从零重训并导出、评估，不是复用上次结果：

| 配方 | 实际训练/评估范围 | 本次质量 | 与变更前权重字节 |
| --- | --- | --- | --- |
| 二分类 | 无敏感合成数值 demo，4 行独立留出 | accuracy 1.0；log loss 0.1074255454 | SHA-256 完全一致 |
| UCI digits | 官方完整3823训练 / 1797最终测试 | accuracy 0.9499165275；macro-F1 0.9501111995 | SHA-256 完全一致 |
| Fashion CNN | 官方固定分层6000训练 / 完整10000最终测试 | accuracy 0.8076；macro-F1 0.8048124060 | SHA-256 完全一致 |

Fashion ONNX/PyTorch 全10000 test 的 argmax agreement 为 1.0，最大绝对 logit 误差 `1.33514404296875e-05`，低于 `1e-4`。数字/Fashion各实际持续调用10次且预测稳定，版本与产物绑定正确；三配方显式停止后预测均返回503。质量不是生产或设备验收，时延不是性能基准。

另外用新 API/serving 直接加载变更前持久化的三套已登记模型；不重写旧 manifest，实际预测、停止及停止后503都通过，旧模型的代码血缘 `dd74a334d6d165edf4c5f4e8eb0f4e287a09f866acb6deea4392d32156da193f` 保持不变。临时 API、Worker、serving 均已终结，验证创建的部署均 stopped。

未在本阶段执行 GitHub 托管 Actions；本地通过不等于远端 CI 已通过。没有 commit、push、merge、部署或修改远端配置。本阶段不实现 vision-lab 接入。

## YOLOX 检测与 mAP 最终增量验收

详见 [检测验收/质量缺陷](detection.md) 和 [结构化证据](detection-verification.json)。177可选测试全过，标准库134通过43跳过；真实YOLOX训练、32图独立test AP50=0.5151476295/AP50:95=0.1721211584、双后端框及AP一致性、持续服务/停止/重启和三分类真实demo通过。原始候选存在高置信极端坐标，模型质量不合格，禁止生产使用；技术生命周期通过不能替代模型质量验收。所有测试进程/演示部署已清理，无数据/权重提交。

## 2026-10-13：业务接入 API 增量

基线为 `feat/standalone-cpu-loop` 的 `07c3bc1786f28556356ec204fcd1bb15a8b3cf6f`，不从文档 main 分支恢复。新增范围及兼容策略见 [integration-api.md](integration-api.md)，机器可读增量证据见 [integration-verification.json](integration-verification.json)。没有改写先前模型、执行源码摘要或检测质量记录，没有调整训练参数/阈值。

- 修改前系统解释器基线：177项，19.513秒，139通过/38可选依赖skip（系统已有NumPy/Pillow）
- 修改后纯净 `.venv` Python3.12.14：`python -m unittest discover -s tests -v`，189项，22.124秒，146通过/43可选skip
- 新增 `test_integration_api.py` 12项全部执行通过，包括真实训练/持续服务HTTP、8并发同key、响应丢失重试、tenant隔离、满配额replay、8不同key竞争4配额、停止与重启、107条历史、同时间排序、后续/backdated插入排除、cursor鉴权/边界/篡改/跨集合、0600签名key并发初始化及旧schema迁移
- `.venv-fashion` 按原 `requirements-detection.lock` 以 `--require-hashes --only-binary=:all:` 安装成功，`pip check` 无冲突；确认PyTorch CPU和ONNX CPUExecutionProvider
- 可选CPU测试按现有CI完全相同的筛选逻辑，明确只排除 `test_fashion_data.OptionalOfficialCacheTests.test_full_source_is_reverified_offline`（未恢复Fashion官方缓存）：其余188项全部通过，30.838秒，0skip
- `compileall`、`git diff --check` 通过。源码不引入新依赖；stdlib与可选库两条路径均覆盖
- 独立只读安全审查发现合法长Unicode tenant可令cursor超过长度上限；已用固定长tenant SHA256摘要绑定修复并新增160个非BMP字符回归。复核额外8进程64次完整性key初始化同钥/0600通过，事务/并发/配额/停止重放无新增阻塞

本轮平台测试不下载官方数据、不重新训练YOLOX、不声称检测质量改善。业务侧最小真实检测接入另行验收，不能用这里的二分类真实HTTP或合成ONNX回归代替业务端到端证据。所有测试创建的API/Worker/serving均由测试清理；共享可选venv仅供后续受控真实检测接入验收使用。
