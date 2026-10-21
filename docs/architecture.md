# 架构与首版契约

2026-10-12：用户已确认独立 Python 核心和下述权威边界。本次先记录可审阅契约，再实施首个 CPU 切片。实际执行证据见 [verification.md](verification.md)。

## 权威边界

model-platform 独立管理算法配方、数据快照及技术引用、TrainingRun/attempt、独立留出集 Evaluation、不可变 ModelVersion/产物、部署期望及实际运行状态，并提供自身基本鉴权与技术资源隔离。无需 Sluggard、业务数据库、外部资产系统或 GPU 即可运行。

Sluggard 是可选业务接入：管理课程、业务任务、业务审批、业务权限、界面和关联，通过稳定技术 ID 和接口引用本平台事实。业务审批可决定是否调用技术接口，但不直接修改技术模型登记、执行或部署状态。不假定已有真实 Sluggard 接口；本次不实现对接。

visionlab、教学、机器人和无人机产品独立。本平台不承担课程成绩、机器人控制、飞控或设备安全。未来端侧包与外部 endpoint 适配属于规划；数值分类示例不代表视觉能力，新增Fashion灰度小CNN也不代表无人机或实机验收。

## 选型与运行边界

- Python 3.12.14（本次 Linux 云端固定并实测），标准库 HTTP/JSON、SQLite WAL、Unix 文件锁；基础二分类/UCI零第三方运行或测试包，依赖锁见 requirements.lock；可选Fashion深度学习依赖单独锁于requirements-fashion.lock。选择成本是自行维护小型 HTTP 契约、输入校验与错误处理；后续公网服务应迁移成熟网关/框架。
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
| GET /health | API 状态、配方、预算、取消和有条件训练重试能力，不承诺 Worker 存活 |
| POST /runs | 必须 Idempotency-Key；同 tenant 同 key 同摘要返回同 run，不同摘要 409；首次 202，重放 200 |
| GET /runs、GET /runs/{id} | 不返回原始数据；包含 attempt、日志、状态和 model_version_id |
| POST /runs/{id}/retry | 有条件请求绑定最新失败 attempt，保持原代码/输入并最多 3 attempts；旧 `{}` 兼容，详见[契约](conditional-training-retry.md) |
| POST /runs/{id}/cancel | `{}` 撤销 queued → cancelled；重复取消200；running/failed/succeeded为409，不中断进程 |
| GET /models、GET /models/{id} | 不可变技术清单、血缘、评估、资源 ID 和摘要，不暴露路径或权重 |
| POST /deployments | 指定 model_version_id；可选 Idempotency-Key，首次202/重放200；持久化 desired=running 后后台启动 |
| GET /deployments、GET /deployments/{id} | desired 与 observed 分开；ready 来源于带版本的真实零向量推理探测，含检查时间 |
| POST /deployments/{id}/predict | features 数组；实际 ready 才转发，返回版本、摘要、概率和分类 |
| POST /deployments/{id}/stop | 先拒绝新调用，再终结进程；失败如实返回 503，成功返回 observed=stopped |

错误为 JSON `{error: {code, message}}`：400 契约错误、401 鉴权、404 无权或不存在、409 冲突、413 超大输入、429 预算、503 未就绪/运行失败。未知字段被拒绝。已有鉴权 ONNX artifact 下载与交付清单，详见 [edge-delivery.md](edge-delivery.md)；另有精确hash的[预训练导入上传](pretrained-import.md)；无任意模型导入、任意执行、任意外部访问或真实业务接口。

## 后续能力

更新/回滚需新部署 ID，首版没有流量原子切换与自动回滚；没有模型质量发布门槛、审批准入或生产发布。设备包、更多视觉算法、厂商来源、外部 endpoint、对象存储、分布式 Worker、运行中取消、强资源隔离和成熟多用户权限体系另行设计与验收。

## 数字十分类增量契约

POST /runs 可选顶层 `recipe`，省略时仍为 logistic-regression/v1。新值 softmax-digits/v1 对应独立 digits-64x10/v1 / softmax-json/v1；旧 JSON 权重和二分类输出结构继续加载。新配方限定64个整数0..16特征、标签0..9、每split含十类、训练10..3823行/评估10..1797行、epochs1..60、learning_rate0.001..0.1、seed整数0..2^32-1。其固定像素缩放/16不学习test统计；标准库softmax SGD、确定性零初始化与固定种子洗牌，不引入pickle或新依赖。Worker限制训练循环120秒，产物最大32KiB。API只校验和排队，不训练。

数据须另含 provenance（archive_sha256/train_sha256/evaluation_sha256、official_split=true、license=CC-BY-4.0、固定UCI source_url）；API检查声明格式而不下载或独立认证来源，也不保证小型调用方样本就是完整官方数据。完整数据演示由下载脚本校验固定官方原字节SHA256，实际用3823训练、1797最终测试。数字配方保留量化后相同特征的不同记录；模型manifest记录跨split重合评估行数与来源声明，下载审计另有完整重复统计。原二分类重合拒绝保持不变。

新推理输出每行 `{class: 0..9, probabilities: [p0,...,p9]}`，响应contract、deployment.contract以及manifest均来自绑定模型；ready探测验证十项概率与和。仍限128行/请求。评估重载登记前产物，独立计算 held-out-multiclass/v1 的accuracy、macro_f1（固定十类）、log_loss、confusion（行真值/列预测）及class_order。训练与评估预测耗时是当前环境观察，不是性能基准。无test调参、早停、原图CNN、ONNX/GPU或设备扩展。数据归属、来源及运行见[digits.md](digits.md)。

## Fashion-MNIST CNN 增量契约

新增可选fashion-cnn/v1，固定公开官方缓存引用、6000分层训练/10000独立最终test，PyTorch CPU从零训练、ONNX CPU评估和持续服务。详细输入、版本、预处理及资源预算见[fashion.md](fashion.md)。标准库二分类/UCI依赖保持不变；Fashion额外库是显式可选安装。ONNX产物只允许已登记hash验证的自生成文件，无外部模型上传。

## 静态内部配方注册与 CI 增量

三套配方现由不可变静态白名单统一分派校验、执行、产物加载和预测，配方预算与元数据有单一目录；见 [recipes.md](recipes.md)。不开放运行时注册或外部代码导入；后续鉴权只读发现见[业务接入增量契约](integration-api.md)。旧请求/产物/预测/血缘兼容；可选库继续 lazy。Worker 与 supervisor 生命周期职责不变。

新增 [CPU CI](ci.md) 执行标准库全部测试和独立 hash-locked ONNX 无数据下载回归。远端 Actions 的实际状态以对应提交的 run 为准，本地执行不等同远端已通过。

## YOLOX 单类检测增量契约

检测是独立的 `rgb-cat-detection/v1`，输入受限单张RGB像素对象，输出原图绝对坐标框、class_id=0、confidence，无分类概率向量。训练固定许可来源的数据引用，CPU从零训练并ONNX导出；验证/测试独立，报告COCO-style AP50及AP50:95，不冒充Open Images官方benchmark。探测回调由静态配方定义，沿用原版本登记、独立服务、停止和期望状态恢复。详见 [detection.md](detection.md)。

## 业务接入 API 增量

鉴权 `GET /recipes`、部署可选幂等以及集合显式稳定cursor分页已加入；无query集合与无key部署保持旧客户端行为。详情、安全边界与质量字段见 [integration-api.md](integration-api.md)。它不改变训练/模型技术血缘、静态注册或YOLOX禁止生产状态。

## 自定义 YOLO 数据集版本增量

2026-10-15：独立 staged import → ZIP validation report → explicit immutable version publication。技术对象仍由 model-platform 主写，tenant 来自 Bearer；没有新增训练配方、任意执行或网络下载。数据集专用 ZIP 路由采用独立32 MiB传输/128 MiB展开/像素预算，原 JSON 和 Cat 流程不变。完整契约、事务/恢复边界和限制见 [dataset-imports.md](dataset-imports.md)。

## 数据集版本绑定训练增量

2026-10-15：新增 `yolox-nano-dataset/v1`。提交事务写入不可变 run→版本关联，Worker 多阶段校验快照；多类别 ONNX模型与配置、代码、环境、评估、模型技术版本相连。val/test 明确区分，比较要求相同评估数据/类别/协议。固定 Cat 兼容；完整请求、预算、错误和服务契约见 [dataset-training.md](dataset-training.md)。

## 自定义数据集模型服务身份

2026-10-15：复用显式 deployment 生命周期，新增模型/训练/数据集/评估身份及固定推理协议、研究范围输出；加载与每次响应做一致性校验，停止重放不重启，API恢复不替换模型。详见[dataset-serving.md](dataset-serving.md)。

## CPU ONNX 交付增量（2026-10-15）

`GET /models/{id}/delivery` 返回版本化交付协议，`GET /models/{id}/artifact` 返回完整、已校验摘要与长度的登记 ONNX；复用租户鉴权，跨租户 404。下载前生成有并发/字节预算的私有快照，不接受路径、范围、转换参数。该产物不是训练恢复 checkpoint；TensorRT/RKNN 与生产审批未实现。[完整契约](edge-delivery.md)。

`GET /worker-readiness` 读取实际 Worker 写入的心跳、启动依赖探测与当前租户队列计数，不在 API 进程导入 ML 库，不暴露其他租户任务。心跳不等于数据可用或训练必然成功；进程崩溃有最多 15 秒检测窗口。[独立部署与排障](cpu-worker-deployment.md)。

## 受限预训练导入增量（2026-10-18）

单一审查用户资产通过独立 ModelImport/ImportValidation 登记 origin=imported，模型互斥来源约束及离线备份迁移保持旧训练关联。独立 CPU 协议不加入可训练recipes，不创建训练或评估；权重授权未独立核实，下载再分发禁用。具体鉴权路由、字节准入、资源限制及本轮待完成验收见[契约与证据](pretrained-import.md)。


## 导入模型独立评估增量（2026-10-18）

已先完成独立契约评审，新增 evaluation-purpose val-only 数据集与独立 `model_evaluations` 队列/不可变终态结果。共享32项训练/评估预算和单 Worker，按全局创建顺序领取；评估在受限 CPU 子进程执行，超时与重启中断失败闭合。绑定实际模型/import/validation/hash、数据集/类别/原图与请求 RGB、固定推理/评估协议、代码及依赖版本、预测摘要。AP/AR 和单阈值 TP/FP/FN 语义分开，错误例有数量/响应字节上限。

导入 ModelVersion 原有 `evaluation:null` / `not_evaluated` 是登记时事实，保持不可变；新的独立评估通过单独列表/详情关联，不能伪造训练 attempt 或将新结果写回旧 manifest。旧训练 train+val 及模型服务协议不变。当前是可信本机、小规模逻辑预算，不是生产多租户沙箱、完整 COCO benchmark 或外部独立审计。详见[独立评估契约](independent-evaluation-contract.md)。


## 独立评估只读比较（2026-10-18）

新增鉴权 `GET /model-evaluation-comparisons?evaluation_a_id=...&evaluation_b_id=...`，只比较同租户两条不同 ID 的成功独立评估，不要求模型 ID 不同。两侧完整性校验后严格比较数据版本、内容、样本顺序、类别映射、协议、提交和执行代码/Python/平台/依赖；任何差异返回明确不兼容原因和空差值。兼容时返回 AP50/AP50:95/AR@100、precision/recall、TP/FP/FN 的 B−A 差值，保留 null。

只读请求不初始化数据集 schema、不写审计/结果/模型、不运行推理；既有训练比较、导入登记血缘、serving 和部署限制不变。详见[已评审契约](evaluation-comparison-contract.md)及[验收记录](evaluation-comparison-verification.md)。现有单一权重的重复记录仅说明固定条件下的重复性，不代表多模型排名、生产质量门槛或官方 COCO benchmark。

## 单条独立评估报告

`GET /model-evaluations/{id}/report` 复用租户归属优先、捕获行及原证据验证，导出安全的确定性 JSON 投影。共享标准库 schema 支持离线内部一致性校验，不读取或导出原预测/图像/权重；[契约与边界](evaluation-report-contract.md)。既有 detail/comparison 接口保持不变。


## 训练排队撤销

新增 queued → cancelled 单向终态，领取/撤销事务互斥，保留历史与原提交幂等绑定。健康能力分开表示 `cancel_supported=false` 与 `queued_cancel_supported=true`；旧运行目录需要显式离线备份迁移，详情见[排队撤销契约与升级步骤](queued-training-cancellation.md)。独立评估与运行中训练不支持取消。

## 失败训练有条件重试

`POST /runs/{id}/retry` 可显式携带最新失败的 `expected_attempt_id`。归属、完整失败历史、三次上限、记录与当前源码摘要、共享容量均在同一 `BEGIN IMMEDIATE` 事务内核验；旧失败 ID 不能在新 attempt 再失败后触发额外执行。只排队原 run，不重写输入/环境/绑定、不产生新 attempt 或 schema 迁移。`GET /health` 声明 `conditional_training_retry_supported=true`；旧 `{}` 行为仍保留，但不保证按失败代际防重。完整错误、响应竞争和客户端不确定性要求见[有条件重试契约](conditional-training-retry.md)。

## 数据集训练评估报告

鉴权 `GET /models/{model_id}/training-report` 只读导出 `yolox-nano-dataset/v1` 成功模型的记录事实，单数据库快照核验关联，不读原始样本/权重、不写状态、不重新评估。协议 `training-evaluation-report/v1` 与独立导入模型报告严格分开，字段和校验边界见[训练报告契约](training-report-contract.md)。

## 训练实验可信比较

新增[训练比较只读契约](training-comparison.md)：复用训练报告跨表证据验证，两侧同快照、归属优先；评估内容/协议/实际执行环境一致才计算总体及逐类B−A差值。原POST比较保留legacy语义，不改独立COCO比较或训练报告格式，不构成质量门禁。
