# 架构与首版契约

2026-10-12：用户已确认独立 Python 核心和下述权威边界。本次先记录可审阅契约，再实施首个 CPU 切片。实际执行证据见 [verification.md](verification.md)。

## 权威边界

model-platform 独立管理算法配方、数据快照及技术引用、TrainingRun/attempt、独立留出集 Evaluation、不可变 ModelVersion/产物、部署期望及实际运行状态，并提供自身基本鉴权与技术资源隔离。无需 Sluggard、业务数据库、外部资产系统或 GPU 即可运行。

Sluggard 是可选业务接入：管理课程、业务任务、业务审批、业务权限、界面和关联，通过稳定技术 ID 和接口引用本平台事实。业务审批可决定是否调用技术接口，但不直接修改技术模型登记、执行或部署状态。不假定已有真实 Sluggard 接口；本次不实现对接。

visionlab、教学、机器人和无人机产品独立。本平台不承担课程成绩、机器人控制、飞控或设备安全。未来端侧包与外部 endpoint 适配属于规划；本次数值分类示例不是视觉或无人机能力验收。

## 选型与运行边界

- Python 3.12.14（本次 Linux 云端固定并实测），标准库 HTTP/JSON、SQLite WAL、Unix 文件锁；零第三方运行或测试包，依赖锁见 requirements.lock。选择成本是自行维护小型 HTTP 契约、输入校验与错误处理；后续公网服务应迁移成熟网关/框架。
- 模块化单体 API、一个独立训练 Worker、每个部署一个独立受控推理子进程。HTTP 不训练；SQLite 事务队列防止提交与领取之间丢任务。
- 一个运行目录限一个 API、一个 Worker；文件锁在进程死亡时释放。Worker 启动将未终结 attempt 标记 interrupted/failed，保留日志；不自动重训。显式 retry 新增 attempt，最多 3 次。queued 请求继续领取。
- API 监督子进程；推理端口仅 localhost，随机内部 capability 通过 stdin 传递。对外调用必须经过鉴权 API，不允许任意路径、URL、代码配方或 pickle。部署只加载已登记版本并校验 SHA-256。
- API 重启清除旧观察状态，按持久化 desired 状态重建实例，再实际探测。子进程检测父进程死亡后退出；不会基于旧 PID 杀任意进程。停止先保存 stopped 期望并阻断新调用，再等待子进程终结。已受理调用可能完成，停止返回时子进程已终结。
- 资源预算：全局 32 个排队/运行 run，单 Worker，最多 4 个活动部署，API 最多 16 个连接、每 tenant 每分钟 120 请求、每请求 2 MiB（原二分类规范化 JSON ≤1 MiB）；二分类单次训练最多 1,000 行、16 特征、200 epoch、60 秒；推理最多 128 行。不同 token 的 tenant 对所有对象强制隔离，客户端不能选择租户。
- 这些是逻辑预算和进程分离，**没有 cgroup、容器、CPU 配额、RSS 硬上限、独立 OS 用户或 GPU 隔离**。同一 OS 用户可读本地数据，训练仍可能争用 CPU。仅支持可信本地操作者、小型 CPU 数据，不能宣称生产多租户安全或公网就绪。

## 数据、算法与血缘

兼容配方 `logistic-regression/v1`：纯 Python 批梯度下降二分类，标准化参数只从训练集拟合；学习率和 epoch 固定输入，确定性零初始化。模型是带契约版本的 JSON 浮点数组。

提交必须包含 `rights_confirmed: true`、数据 source/version、分开的 train/evaluation 集（两者均含 0/1，样本特征不可跨集合重叠）。数值有限、有界、维数一致。合成数据仅用于可复核示例，但执行的是实际参数优化和实际留出集预测，不伪造指标。全常量训练特征是合法输入但算法不可训练，会产生真实失败记录。

run 固定输入快照、摘要、配方、代码源码摘要、Python/平台环境；Worker 固定启动时的执行环境，并在执行前、登记前校验源码摘要，拒绝代码变化；attempt 单独保留开始/结束、序号、错误及结构化日志。评估从已写出的 JSON 产物重新加载，在独立留出集分批计算（每批不超过 128 行，合计受 1,000 行数据预算约束）accuracy、log_loss、confusion、样本数，记录产物和评估数据摘要。评估阶段与训练实现分离，但由同一个 Worker 顺序执行，不代表外部独立审计。

成功登记使用唯一 model version ID，run 和 attempt 唯一约束；产物、评估和模型元数据不允许 UPDATE/DELETE（SQLite 触发器），API 无修改入口。文件摘要在评估及每次服务加载时核实。DB 事务提交模型、评估和成功状态；若文件已落盘而 DB 未提交，文件成为未登记孤儿，不允许部署，不自动删除。本地管理员仍可破坏磁盘文件；逻辑不可变不是 WORM 存储。

## 最小 API（v1）

所有路由含健康查询均要求 `Authorization: Bearer <token>`；凭据由本地 init 生成在忽略目录，权限 0600，运行目录 0700。身份绑定 tenant，tenant 不从请求体接受。对象无权访问返回 404。

| 方法 / 路径 | 行为 |
| --- | --- |
| GET /health | API 状态、配方、预算和取消能力，不承诺 Worker 存活 |
| POST /runs | 必须 Idempotency-Key；同 tenant 同 key 同摘要返回同 run，不同摘要 409；首次 202，重放 200 |
| GET /runs、GET /runs/{id} | 不返回原始数据；包含 attempt、日志、状态和 model_version_id |
| POST /runs/{id}/retry | 仅 failed 可重排，保留历史，最多 3 attempts；其他状态 409 |
| POST /runs/{id}/cancel | 501，明确不支持；不伪造 cancelled |
| GET /models、GET /models/{id} | 不可变技术清单、血缘、评估、资源 ID 和摘要，不暴露路径或权重 |
| POST /deployments | 指定 model_version_id；持久化 desired=running 后后台启动，202 |
| GET /deployments、GET /deployments/{id} | desired 与 observed 分开；ready 来源于带版本的真实零向量推理探测，含检查时间 |
| POST /deployments/{id}/predict | features 数组；实际 ready 才转发，返回版本、摘要、概率和分类 |
| POST /deployments/{id}/stop | 先拒绝新调用，再终结进程；失败如实返回 503，成功返回 observed=stopped |

错误为 JSON `{error: {code, message}}`：400 契约错误、401 鉴权、404 无权或不存在、409 冲突、413 超大输入、429 预算、501 取消不支持、503 未就绪/运行失败。未知字段被拒绝。无 artifact 上传/下载、模型导入、任意执行、任意外部访问或真实业务接口。

## 后续能力

更新/回滚需新部署 ID，首版没有流量原子切换与自动回滚；没有模型质量发布门槛、审批准入或生产发布。设备包、视觉算法、厂商来源、外部 endpoint、对象存储、分布式 Worker、取消、强资源隔离和成熟多用户权限体系另行设计与验收。

## 数字十分类增量契约

POST /runs 可选顶层 `recipe`，省略时仍为 logistic-regression/v1。新值 softmax-digits/v1 对应独立 digits-64x10/v1 / softmax-json/v1；旧 JSON 权重和二分类输出结构继续加载。新配方限定64个整数0..16特征、标签0..9、每split含十类、训练10..3823行/评估10..1797行、epochs1..60、learning_rate0.001..0.1、seed整数0..2^32-1。其固定像素缩放/16不学习test统计；标准库softmax SGD、确定性零初始化与固定种子洗牌，不引入pickle或新依赖。Worker限制训练循环120秒，产物最大32KiB。API只校验和排队，不训练。

数据须另含 provenance（archive_sha256/train_sha256/evaluation_sha256、official_split=true、license=CC-BY-4.0、固定UCI source_url）；API检查声明格式而不下载或独立认证来源，也不保证小型调用方样本就是完整官方数据。完整数据演示由下载脚本校验固定官方原字节SHA256，实际用3823训练、1797最终测试。数字配方保留量化后相同特征的不同记录；模型manifest记录跨split重合评估行数与来源声明，下载审计另有完整重复统计。原二分类重合拒绝保持不变。

新推理输出每行 `{class: 0..9, probabilities: [p0,...,p9]}`，响应contract、deployment.contract以及manifest均来自绑定模型；ready探测验证十项概率与和。仍限128行/请求。评估重载登记前产物，独立计算 held-out-multiclass/v1 的accuracy、macro_f1（固定十类）、log_loss、confusion（行真值/列预测）及class_order。训练与评估预测耗时是当前环境观察，不是性能基准。无test调参、早停、原图CNN、ONNX/GPU或设备扩展。数据归属、来源及运行见[digits.md](digits.md)。
