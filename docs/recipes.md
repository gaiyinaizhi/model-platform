# 内部配方扩展协议

2026-10-12：本次把三套已有配方的分派集中到 [`model_platform/recipes.py`](../model_platform/recipes.py)，不新增业务集成或任意插件执行能力。外部 `POST /runs`、登记产物、预测响应及旧版本血缘继续使用已有契约。省略 `recipe` 仍选择 `logistic-regression/v1`，不会规范化改写请求体，因此既有幂等摘要语义不变。

## 静态、受审查的注册

- `RecipeSpec` 和 `Budgets` 是 frozen dataclass；`REGISTRY` 是只读 mapping，按源码中的显式白名单初始化。`build_registry(specs)` 校验重复 ID / contract / format、版本化 ID、预算、元数据与必需 callable，返回新的只读 mapping，不修改当前目录
- `get_recipe(id)` 严格查找；未知、非字符串、路径或模块名均拒绝，不回退为二分类。没有 HTTP 注册、环境变量注入、entry-point 扫描、用户可控模块路径或动态外部 import
- 增加配方意味着受代码审查的源码更改，并更新测试、文档和能力边界；不是上传 Python、pickle、权重或训练脚本
- `catalog()` / `describe()` 返回脱离原对象的 JSON-safe 元数据副本，不含函数、运行路径或凭据。它们是内部 Python 接口，**不是已发布的 HTTP API**

## 最小协议与职责

`RecipeSpec` 集中记录 ID、输入/输出 contract、artifact format、运行时、类别数、数据模式、可选依赖和逻辑预算，绑定下列本地 callback：

1. `validate_request(dataset, config)`：配方专属输入、配置、大小和形状校验。公共 `training_request` 保留顶层未知字段、合法使用声明、来源/版本和可选业务引用校验，使用该配方的 request byte 预算；返回原请求，不改变默认值或摘要
2. `execute(spec, runtime, body, path, log, execution_environment, submission_environment)`：返回 `(artifact_sha256, evaluation, manifest)`。数值配方共用 train → 排他写产物 → 从产物重载评估的适配器；Fashion 保留原有固定数据加载、CPU 训练、ONNX 导出、重载独立评估及双后端一致性流水线
3. `load(path, expected_hash)`、`predict(model, features)`、`validate_features(features, dimensions)`：按登记 manifest 选择 loader，校验 recipe/contract/format、产物摘要和维数绑定；服务监督进程与推理子进程使用相同白名单分派
4. JSON 数值配方额外提供 `train(rows, config, log)`、`validate_model(model)`。ONNX 不通过 JSON 模型入口；它没有伪造的 JSON validator

Worker 继续独占 attempt/队列生命周期、源码环境摘要核验、错误清洗和事务登记。配方适配器不领取任务、不变更重试策略、不自建服务进程、不写业务状态。训练/评估/登记失败依旧产生原错误及 attempt 历史，最多三次显式尝试；中断恢复仍标记 failed，不自动重训。

## 兼容与预算

| 配方 | contract / format | 数据与主要逻辑预算 |
| --- | --- | --- |
| logistic-regression/v1 | binary-numeric/v1 / logistic-json/v1 | 内联；合计 ≤1000 行、≤16 维、≤200 epochs、60 秒训练、128 行预测、32 KiB JSON 产物；请求 ≤1 MiB |
| softmax-digits/v1 | digits-64x10/v1 / softmax-json/v1 | 内联；训练 ≤3823、评估 ≤1797、固定64维、≤60 epochs、120 秒训练、128 行预测、32 KiB JSON 产物 |
| fashion-cnn/v1 | fashion-28x28x10/v1 / fashion-onnx/v1 | 固定公开引用；6000训练/10000评估、固定784像素/5 epochs、600秒训练、导出/评估各120秒、32行预测、2 MiB ONNX产物 |

所有传输仍受 2 MiB 全局预算。配方预算由不可变 registry 提供给实际校验、训练超时、JSON/ONNX读写及评估批次；固定 Fashion 配置、数据来源和原有常量别名继续兼容。不同配方细节仍在对应算法模块维护，不把所有领域逻辑挤成通用引擎。预算都是可信本地执行的逻辑限制，native 调用不能被协作式超时强行中断，不宣称硬 CPU/RSS 沙箱。

注册和目录查询不导入 torch、NumPy、ONNX、ONNX Runtime 或 onnxscript。只有选择并执行/加载 Fashion 才使用可选库；基础两配方在零第三方包环境继续运行。目录中的依赖说明不表示机器已经安装它们；执行时缺依赖仍产生 `optional_dependencies_missing`，不得假装训练成功。

旧 JSON/ONNX 产物、输出字段、登记格式、来源声明、权重摘要、评估协议、代码血缘及旧 manifest 不做迁移或覆写。新增源码文件自然改变新 run 的 `code_sha256`，旧版本仍保留原值；部署旧版本按其登记产物和契约加载。

## 未来只读发现接口：候选，尚未实施

现有鉴权 `GET /health` 的 `recipes` 仍为 ID 列表。未来可以单独评审一个鉴权只读 `GET /recipes`，提供 versioned discovery contract、ID、task kind、输入形状/范围、输出语义/类别顺序、指标名、数据模式、配置默认值和限制、依赖要求、可用能力。应区分“服务器支持该配方”与“运行时依赖已就绪”，不能将 catalog 列表当作 Worker 健康或硬隔离承诺。

当前 vision-lab 是 YOLO 检测界面，检测框/mAP 不可直接映射三套分类配方。未来接入需先确认业务任务与输出契约，不能只替换 base URL；本次没有调用、猜测或实现 vision-lab 的 API。

新增配方的验收至少覆盖：注册冲突/未知配方、输入和预算拒绝、无可选库启动、真实训练/导出/独立评估、旧产物兼容、预测/停止/崩溃恢复/版本绑定、失败重试和代码变更边界。CI 见 [ci.md](ci.md)，本次实际结果见 [verification.md](verification.md)。
