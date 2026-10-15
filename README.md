# model-platform

供 `visionlab`、教学平台、机器人和无人机等独立产品复用的 Python 模型训练、评估、技术登记与推理运行核心。

**状态：2026-10-12，首版可信本地 CPU 闭环已实现并在 Linux / Python 3.12.14 验证。** 保留首版小型数值二分类回归，并新增 UCI 官方全量手写数字十分类 CPU 配方（[数据、许可及运行指南](docs/digits.md)）：真实优化逻辑回归参数、从 JSON 产物重新加载并进行独立留出评估，再运行持续 CPU 推理子进程。新增可选 [Fashion-MNIST 小型 CNN / ONNX CPU 切片](docs/fashion.md)；另新增可选 [YOLOX-Nano Cat 检测 / COCO-style mAP CPU 切片](docs/detection.md)；该增量的实际验证状态单独记录，不代表无人机能力、实机安全或生产多租户验收。

## 独立运行与职责边界

model-platform 主写模型技术版本、产物、训练及 attempt、评估、部署期望和实际运行状态，提供自己的 Bearer 鉴权与租户技术对象隔离。无需 Sluggard、业务数据库、外部资产平台或 GPU。

Sluggard 仅为可选业务接入，管理课程、业务任务、业务审批、业务权限和关联，通过稳定技术 ID 引用本平台事实；本版没有臆造或实现 Sluggard 接口。`visionlab`、教学平台、机器人和无人机各自保持独立，本仓库不承担课程成绩、机器人控制、飞控或设备安全。

## 已实现

- 新增[自定义 YOLO 数据集导入/版本](docs/dataset-imports.md)：受限 ZIP、逐文件图像/框/跨 split 校验、显式负样本策略、租户隔离、不可变复制快照；可通过[版本绑定多类别训练](docs/dataset-training.md)执行受控 CPU 配方

- 标准库 HTTP API + SQLite WAL 事务队列；API 不执行训练，独立 Worker 顺序领取任务
- 兼容配方 `logistic-regression/v1`，真实梯度训练、训练集拟合标准化参数、独立留出集实际计算 accuracy / log loss / confusion
- 独立配方 `softmax-digits/v1`：64 个 0..16 整数特征、10 类，确定性 softmax SGD；官方 3823/1797 划分全量保留，accuracy / macro-F1 / 10×10 confusion / log loss
- JSON数值模型及可选ONNX图像模型、SHA-256、数据快照摘要、代码/环境/配置血缘；评估与模型技术记录逻辑不可变
- 新增`yolox-nano-dataset/v1`：绑定当前租户不可变 YOLO 版本、1..20类、最小超参数、独立 ONNX mAP/逐类别 AP 和受兼容性约束的实验比较；[契约与验收边界](docs/dataset-training.md)
- 可选`yolox-nano-cat/v1`：官方 Apache-2.0 YOLOX-Nano 模型、Open Images Cat 128/32/32 独立划分、CPU 从零训练、真实框预测和 pycocotools mAP（test AP50=51.51%、AP50:95=17.21%；存在高置信极端原始框，质量不合格、禁止生产使用）；[契约/验收状态](docs/detection.md)
- 可选`fashion-cnn/v1`：官方Fashion-MNIST固定分层6000训练/完整10000 test，小CNN从零训练、ONNX一致性验证与独立CPU持续服务，细节见[Fashion指南](docs/fashion.md)
- 幂等提交、输入/大小/速率/队列预算、失败日志、最多 3 次执行尝试，重试保留失败历史
- 每部署一个独立 CPU 推理进程，绑定明确模型版本；实际预测探测 ready，停止进程后拒绝预测；自定义数据集模型返回完整类别/血缘及研究范围，见[持续推理契约](docs/dataset-serving.md)
- API 崩溃后的持久化期望状态恢复；Worker 中断在下次启动登记为 failed，显式重试，不静默重训

## 环境要求

- Linux 或支持 `fcntl.flock`、Unix 文件描述符继承及进程信号的系统；本次云端验收为 Linux x86_64
- **本次固定并实测 Python 3.12.14**，见 `.python-version`；二分类/UCI核心标准库包含 SQLite，不需要第三方包。Fashion配方额外依赖见可选 `requirements-fashion.lock`，无需GPU或CUDA
- `requirements.lock` 声明基础配方零第三方依赖，不是 Python 解释器二进制锁；没有宣称完整 Python 版本矩阵或 Windows 支持
- 所有命令从仓库根目录运行。API 固定绑定 `127.0.0.1`，仅在当前机器可访问；不要将开发 HTTP 服务反向代理或暴露到公网

## 快速运行

准备一次；`init` 会生成本机测试身份，只写本地文件，不打印 token：

```sh
python3 --version
python3 -m venv .venv
.venv/bin/python -m model_platform init --tenant demo --tenant other
```

默认运行目录是仓库的 `.runtime/`，在 `.gitignore` 内，包含凭据、原始输入快照、SQLite 和模型权重。新建运行目录为 0700，凭据为 0600。可以用 `MP_HOME` 指定自己的私有目录，但应放在仓库外，或 `.runtime/` 内；自定义路径不自动得到 Git 忽略保护。不要复用共享或公开目录。`init` 不覆盖已有凭据，再运行会明确失败。

终端 A 启动 API：

```sh
.venv/bin/python -m model_platform api --port 8080
```

终端 B 启动独立 Worker：

```sh
.venv/bin/python -m model_platform worker
```

终端 C 执行真实示例：

```sh
.venv/bin/python scripts/demo.py
```

三个终端须使用同一仓库和同一 `MP_HOME`。演示默认访问 `http://127.0.0.1:8080`，自动读取 `demo` 的本地凭据，不要求将 token 复制到聊天、命令行参数或文档。若用其他端口，通过 `MP_API=http://127.0.0.1:端口` 指定；API 的 `--port 0` 会选择空闲端口并在启动时打印地址。

演示生成无敏感信息的合成数值数据，验证重复提交返回同一 run，等待真实训练和评估，部署登记版本，实际预测 `[-2.5]`、`[2.5]`，然后停用服务并验证后续调用返回 503。输出包含 run / model ID、计算指标、预测及停止确认；指标没有硬编码。该小例的高准确率不代表业务模型质量。每次演示使用新幂等键，因此会生成新的历史记录。

API / Worker 通过各自终端的 Ctrl+C 停止。API 正常退出会终结自己的推理子进程；未显式停止的部署仍保留 `desired=running`，下次 API 启动重新加载并验证。演示创建的部署会显式停止。Worker 中断的执行在下一次 Worker 启动后变为 failed，队列中的其他任务仍可领取。不要在进程运行或任务排队期间修改代码；代码摘要变化时任务会失败为 `execution_code_changed`，应停止并重启 API / Worker，再使用新幂等键提交新 run。

`worker --once` 用于检查/恢复中断状态并至多领取一个任务，适合验收或人工排查；它不会持续消费队列。内部 `serve` 命令仅供 API supervisor 调用。

## API 与预算

所有业务及健康路由均要求 `Authorization: Bearer <token>`。token 固定绑定 tenant，请求体不能选择 tenant；越权对象返回 404。请求 JSON 不允许未知字段、重复字段、非有限数值或任意外部 URL、文件路径、可执行配方。完整路由和错误码见 [架构契约](docs/architecture.md#最小-apiv1)。

- 训练：`POST /runs` 携带 `Idempotency-Key`；`GET /runs/{id}` 查询；失败后 `POST /runs/{id}/retry` 携带 `{}` 显式重试
- 登记：`GET /models`、`GET /models/{id}` 查询不可变技术记录、评估和摘要，不下载原始权重
- 配方发现：鉴权 `GET /recipes` 返回任务/数据/预算/输入输出契约与质量警告；声明能力不等于运行就绪
- 部署：`POST /deployments` 携带 `model_version_id`，可选 `Idempotency-Key` 支持安全重试；查询 deployment 的 `desired`、`observed` 和 `ready`
- 预测：`POST /deployments/{id}/predict` 携带 `features`（分类为二维数值数组，检测为单张受限RGB像素对象）；停止：`POST /deployments/{id}/stop` 携带 `{}`
- `GET /health` 只表示 API 可用，不保证独立 Worker 存活；集合无 query 保持最新100个 ID；`?limit=1..100` / `?cursor=...` 显式开启稳定历史分页，见[业务接入契约](docs/integration-api.md)

除[数据集 ZIP 专用路由](docs/dataset-imports.md)的 32 MiB 上传预算外，传输预算为每请求 2 MiB（二分类规范化 JSON 仍限 1 MiB）、每 tenant 每分钟 120 请求、API 最多 16 个并发连接；全局最多 32 个排队/运行任务和 4 个 `desired=running` 部署。失败部署仍占预算，显式 stop 可释放。二分类训练和留出合计不超过 1,000 行、每个集合至少 4 行且含两类、最多 16 特征、200 epoch、训练循环 60 秒；在线预测最多 128 行。提交数据须声明合法使用，并提供不与训练样本特征重叠的留出集；这是调用方提供的留出数据，不是外部独立审计。数字配方单独限制训练≤3823、评估≤1797、固定64维/10类、≤60 epoch、训练循环120秒；需官方来源及许可摘要声明，完整演示核验官方字节校验和。数字配方保留官方记录中可能重复的量化特征并记录重合审计，不改变二分类的跨集合拒绝规则。

## 验证

已加入 [静态内部配方扩展协议](docs/recipes.md) 和 [双层 CPU CI](docs/ci.md)：基础标准库全套回归与独立可选 ONNX 轻量回归，不在每个 PR 下载官方数据或重训 CNN。

```sh
.venv/bin/python -m compileall -q model_platform scripts tests
.venv/bin/python -m unittest discover -s tests -v
```

测试使用真实临时 loopback HTTP、独立进程、SQLite、参数训练和推理，不访问外部服务。测试身份、模型和数据均写入 `.runtime/tests/` 下的隔离临时目录，结束后清理；进程崩溃测试只针对该测试自己启动的进程。完整实测命令、结果及边界见 [验收记录](docs/verification.md)。

## 明确限制

- **只适合可信本地操作者的小型 CPU 开发示例**：没有 TLS、成熟运维权限、独立 OS 用户、cgroup、容器、硬 CPU/RSS/GPU 配额；不同租户共用队列和逻辑预算，同一 OS 用户可读写运行文件
- 进程分离和 API 技术对象隔离不是生产多租户安全沙箱；本机管理员可破坏磁盘文件，逻辑不可变不是 WORM 存储
- 取消返回 501；没有检查点恢复、自动训练重试、版本热切流、自动回滚或质量发布门槛；更新版本需新部署 ID
- serving 崩溃会撤销 ready，不在同一个 supervisor 生命周期自动重启；API 重启按 desired 状态重新验证启动
- SQLite 单机运行，没有分布式队列、自动数据保留/清理、备份恢复验收或断电耐久性保证；未登记的孤儿产物不会被部署，也不会自动删除
- 图像配方包括固定公开Fashion-MNIST灰度分类、Open Images Cat单类检测和有预算的 YOLO 数据集版本绑定多类别 CPU 检测；检测仅允许受限内联RGB预测；尚无设备转换包、机器人/无人机适配、外部模型导入、厂商 endpoint 或 Sluggard 接口；后续能力见 [路线图](docs/roadmap.md)

## 目录与贡献

```text
model_platform/       API、队列、算法、评估、Worker、推理与生命周期
scripts/demo.py       真实合成数值 CPU 闭环演示
scripts/fetch_digits.py 官方完整数字数据下载与校验
scripts/demo_digits.py 完整官方数字 CPU 闭环演示
scripts/fetch_fashion.py 官方固定Fashion-MNIST下载/验证
scripts/demo_fashion.py 小CNN/ONNX图像CPU闭环演示
scripts/fetch_detection.py 固定Open Images Cat来源/许可/校验
scripts/demo_detection.py YOLOX-Nano框预测/mAP CPU闭环演示
tests/test_platform.py  实际 HTTP / 进程 / SQLite 验收
docs/                架构契约、路线图、验收记录
```

贡献前阅读 [AGENTS.md](AGENTS.md)、[架构](docs/architecture.md) 和 [路线图](docs/roadmap.md)。不要向 Git 提交训练数据、权重、数据库、日志、密钥或可用凭据；`.gitignore` 不能替代提交前检查。
