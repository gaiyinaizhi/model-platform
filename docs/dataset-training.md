# 不可变数据集版本训练（2026-10-15）

本增量将已发布的 YOLO 数据集版本绑定到一个新的、源码审阅的 `yolox-nano-dataset/v1` 配方。固定 Cat 配方 `yolox-nano-cat/v1` 保持兼容；不加载用户代码、训练 YAML、远程路径、外部权重或任意超参数。

## 请求和预算

`POST /runs` 仍要求 Bearer 和 `Idempotency-Key`：

```json
{
  "recipe": "yolox-nano-dataset/v1",
  "dataset": {"dataset_version_id": "dsv_0123456789abcdef0123456789abcdef"},
  "config": {"epochs": 2, "learning_rate": 0.001, "seed": 20261015}
}
```

版本必须来自当前 tenant 的已发布导入。使用权确认继承导入记录；不接受客户端再声明 tenant、类别或摘要。允许 epochs 1..20、learning_rate 0.0001..0.01、seed 0..2^32-1。其他字段拒绝。固定 Adam、batch4、160×160、CPU、随机初始化、最终 epoch；没有预训练权重、早停、超参数搜索或 val/test 选 checkpoint。

配方预算比导入预算更小：1..20 类、1..128 train、1..64 val、0..64 test、所有 split 共≤20M 原始像素、每图≤100框。每个声明类别必须在 train 有标注；允许在 val/test 缺失，AP 为 null。拒绝相同类别/坐标的重复标注和无法有效 letterbox 的宽高比。负样本策略继承导入。导入允许的 1e-9 框边界容差在像素转换时裁到图内，不静默改变其他框。保留 Pillow 原始解码方向，不做 EXIF 转置。

线程和阶段采用现有协作预算：PyTorch intra/inter2；训练600秒、导出120秒、验证+测试180秒、模型8MiB。没有 cgroup、OS 独立用户或硬 RSS/CPU 沙箱。

## 不可变关联

提交事务同时写入 TrainingRun 与不可变 run→dataset-version binding。原始请求和幂等摘要不被改写。绑定包括版本 ID、完整内容摘要、固定有序类别映射、train/val/test 的有序图像及标签摘要、计数和最终评估 split。同 key 不同配置冲突；改变配置或数据需新 run；失败重试保留原绑定及 attempt 历史。

Worker 在训练前、ONNX 独立评估前及技术登记前再次校验所有快照文件；实际解码的图像/标签字节也逐个匹配 SHA-256。变更文件产生 `dataset_integrity`，不登记模型。API 不训练。数据摘要只意味着可信本地操作者下的逻辑不可变，不能防止同 OS 用户恶意同时改写数据库、文件和程序。

ONNX 自描述元数据包括新配方/格式/契约、规范 JSON 类别映射和数据版本/完整/train/eval摘要及 split；loader 校验 `[1,525,5+C]` 输出、算子白名单、无外部 tensors 和文件摘要。serving 将类别和数据关联与登记模型逐一核对。新格式为 `yolox-nano-dataset-onnx/v1`，推理契约为 `rgb-dataset-detection/v1`；请求形状保持受限单张内联 RGB，class_id 范围按绑定模型决定。模型记录配置及摘要、代码摘要、Python/平台、执行环境、依赖版本、优化前后参数摘要、损失、ONNX摘要、run/attempt/evaluation/model ID。

## 指标与比较

只使用 train 优化；最终权重先在 val 报告，再在独立 test 报告。若没有 test，主指标明确 `split=val`、`role=validation-only`，绝不标为 test。有 test 时为 `split=test`、`role=held-out-test`。本平台不阻止操作者根据历次 test 结果手工选择实验；频繁参考 test 会污染其最终留出意义。

官方 pycocotools 的实际 ONNX 框预测计算 `map_50`、`map_50_95`（0..1）和 `per_class`：`category_id` 为零基；`name`、`ground_truth_boxes`、`predictions`、`status`、`ap_50`、`ap_50_95`。没有 GT 的类别 AP 为 null，全部无 GT 时总 AP 亦为 null。PyTorch/ONNX 在每张 val/test 图像比较原始输出、框、分数、服务阈值以及总体和逐类别 AP，不复用训练后端的指标充当交付模型指标。

`POST /model-comparisons` 的请求为 `{"model_version_ids":["model_A","model_B"]}`。必须为两个不同、当前 tenant 可读的本配方模型。成功返回 `compatible=true`、`compatibility_key`、`model_version_ids`、两个 `metrics`（总体/逐类别）和 `deltas`（B−A）。无 GT 不生成数字差值。

兼容条件：评估 split 内容/顺序摘要、split 角色、有序类别映射、COCO 参数和实现版本、预处理、阈值、class-aware NMS 及全图100框上限完全一致。完整训练集可不同；不能仅以完整数据集 hash 判断可比性。冲突409 `evaluation_incompatible`，message 指出 dataset/classes/protocol/split；越权404。该比较只表示评估口径一致，不表示统计显著性、业务质量或可生产发布。

## 运行与验收

复用已有官方 CPU/PyPI、逐 wheel hash-locked `requirements-detection.lock`，没有新增依赖。Python3.12.14。API 和 Worker 必须在同一冻结代码和同一私有 `MP_HOME`；运行中不修改 Python 源码。

```sh
.venv-training/bin/python -m model_platform init --tenant demo --tenant other
.venv-training/bin/python -m model_platform api --port 8080
# 独立终端
.venv-training/bin/python -m model_platform worker
# 独立终端
.venv-training/bin/python scripts/demo_dataset_training.py
```

演示完全自行生成 red_square/blue_square 合成图，实际训练1/2 epoch两个实验、另一个无 test 实验，执行独立 ONNX 评估、兼容/不兼容比较、租户拒绝、真实推理及停止后503。默认结果在忽略目录 `.runtime/dataset-training-evidence.json`，只作为机械链路验收；不代表真实业务数据的检测质量。实际执行状态由本次验收记录给出，脚本存在本身不等于已执行。

### 本次实际验证

- Python3.12.14：完整 ML 环境225项中224通过、1项因官方 Fashion-MNIST 缓存缺失跳过；纯标准库环境151通过、74项因可选依赖缺失跳过；compileall、差异空白检查、24个原锁定 wheel 摘要与 pip check 通过
- Chromium 真实提交8/2/2两类版本的1/2 epoch实验，以及8/2/0版本的20 epoch实验（为观察实时进度，不根据指标调参）。三者均真实优化参数、导出并重载 ONNX、独立计算评估；总体 AP50/AP50:95 都为0，无业务质量结论
- 前两者同评估口径可比；第三者明确 validation-only，蓝色类别在 val 无真值，逐类别 AP 为 null，跨 test/val 比较拒绝
- 同一登记模型的独立 CPU ONNX 重载/预测再次通过，输出[1,525,7]且全部有限；阈值后0框。这是实际推理验证，没有宣称新增推理界面或 custom 持续部署生命周期已验收
- 原始技术 ID、数据/产物/代码摘要、实际参数/损失/逐类别指标及一致性结果见 [dataset-training-verification.json](dataset-training-verification.json)。前端导航恢复的修复/浏览器复验单独记录

### 接入方展示评估状态

检测结果的 `evaluation.status`、null 与零值区分见
[交付评估语义](edge-delivery.md#检测-map-的-null--0-与评估状态)：有 GT 且无预测是已评估的 0，
无 GT 是已执行但 AP 不适用的 null；不能把 null 统一解释为未评估，也不能用小样本数量推断。
任务排队相对位置与重试优先级见 [Worker 忙态及排队](cpu-worker-deployment.md#6-忙态原因与排队位置2026-10-15-增补)。
