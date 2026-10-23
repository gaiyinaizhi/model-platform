# 数据集训练模型的独立评估

## 范围与语义

本契约只支持已经登记的 `origin=trained`、`recipe=yolox-nano-dataset/v1` CPU ONNX 模型。模型登记时的训练、val/test 评估、manifest 与技术版本保持不变。新的独立评估使用单独发布的不可变 `purpose=evaluation`、val-only 数据集，写入既有独立队列和结果表，不创建训练、不自动晋级或切换服务。

“独立”指新的任务与冻结数据版本，不是统计独立或数据泄漏认证。系统不自动审计新数据与原 train/val/test 的跨数据集重复；结果和界面明确提示此限制。用户应自行确认数据使用权与评估划分。新任务不要求仍保留原训练图像，但绑定原训练来源摘要。合成数据验收不是业务泛化质量证据。

原 `imported-coco-serving-evaluation/v1` 协议及结果字段保持不变，不因加入新分支而更改原 COCO80 指标、报告或比较语义。训练期评估、导入模型独立评估和本协议不能隐式混用。

## 请求与身份

沿用鉴权 `POST /model-evaluations`、`GET /model-evaluations`（可按模型筛选）和 `GET /model-evaluations/{id}`。提交必须带 owner-scoped `Idempotency-Key`，JSON 恰含：

```json
{
  "model_version_id": "model_...",
  "dataset_version_id": "dsv_...",
  "split": "val",
  "protocol": "dataset-trained-serving-evaluation/v1"
}
```

模型和数据归属优先检查；越权对象返回404。协议与模型来源/配方不匹配时拒绝。原键原输入重放返回同一条记录；同键不同输入冲突。失败或中断后的新评估须使用新键显式提交，不能自动重训/重试。状态仍为 queued → running → succeeded/failed。

数据要求：1–64张val图，零train/test，最多10,000个框，1–20个有序类别。连续class_id及名称必须与登记模型完全一致；不自动重排或模糊匹配。新版本不得为原训练版本。沿用原图像、框、负样本、尺寸与文件完整性预算。`source` 必须为空，因为既有公开来源声明是 COCO80 专用，不是通用来源认证协议。

## 固定服务等价协议

协议字段固定，不接受调用方调参：

- `id=dataset-trained-serving-evaluation/v1`
- `inference_protocol=yolox-nano-dataset-onnx/v1`
- `max_input_side=384`、`input_size=160`
- `score_threshold=0.01`、`score_operator=>=`
- `nms_iou_threshold=0.65`、`nms_class_agnostic=false`
- `max_detections=100`
- `resize=opencv-inter-linear-floor-max384-no-upscale/v1`
- `metric_protocol=coco-style-bbox-subset/v1`
- `matching_protocol=class-aware-greedy-iou50/v1`

首先按现有调用端协议用OpenCV INTER_LINEAR缩至最长边384，尺寸向下取整，不放大；再调用原服务模型的Pillow双线性160 letterbox、左上对齐、114填充、RGB转BGR、f32 0..255预处理。检查缩放后宽高比可表示性。使用实际登记ONNX、同一服务预测实现，按原图半开区间xyxy坐标还原框。

score恰为0.01保留；类别内NMS在IoU>0.65时抑制；每图所有类别合计最多100个检测。TP/FP/FN采用逐类、确定性贪心IoU≥0.5匹配。该AP/AR来自服务阈值过滤后的检测，不是训练期0.001阈值，也不是导入模型>0.3阈值。不同协议的数值不可直接当作质量差异。

实际计算pycocotools AP50、AP50:95、AR@100，并记录逐类别指标、precision/recall、TP/FP/FN和有界诊断。无GT的AP/AR为null，有GT但无检测为0；界面保留区别，不把null填零。

## 不可变证据

`binding.model` 固定包含 model_version_id、origin、recipe、run_id、attempt_id、evaluation_id、artifact_sha256、artifact_bytes、manifest_sha256。这里evaluation_id是原训练登记时评估ID，不是本次独立任务ID。manifest摘要基于原数据库保存的登记manifest规范化JSON，不使用查询时附加字段。

独立的 `binding.training_dataset` 保存原dataset_version_id、content_sha256、train_sha256、validation_sha256、test_sha256、evaluation_sha256、evaluation_split。test_sha256非空时原评估split为test且评估摘要等于test摘要；为空时split为val且评估摘要等于validation摘要。新 `binding.dataset` 则只描述本次val快照，不能用原训练评估摘要替代。

新类别映射只包含class_id、name、evaluator_internal_category_id（class_id+1），不伪造COCO source_category_id。类别顺序、协议、原图/标签/像素、调用端RGB、模型产物、登记manifest、提交环境/代码/依赖、实际执行环境与预测摘要均绑定到不可变结果。新结果必有 `protocol=dataset-trained-serving-evaluation/v1`；旧导入结果不新增该字段。

提交及已有复核点在只读事务中复用训练报告的跨表验证，核对模型、run、attempt、原evaluation与数据绑定，而非仅相信模型表。产物须安全路径、受限长度和SHA匹配；执行子进程还校验ONNX结构、类别和实际加载身份。结果发布不修改原模型或训练评估。

## 生命周期与明确不支持

完整复用[已验证的独立评估监督](independent-evaluation-supervision.md)：只读子进程、共享close-only租约、导入前内核父死信号、有限资源及结果通道、确认回收后终态、未知终止保持running并停止领取。无schema迁移或生命周期变更。这不是OS沙箱或任意后代治理。

本协议只开放提交、轮询、历史和详情。以下接口对新协议明确返回409：

- 报告导出：`evaluation_report_unsupported`
- 两条独立记录比较：`evaluation_comparison_unsupported`
- 错误图像预览：`evaluation_preview_unsupported`

先校验归属，比较先校验双方归属。不会把新记录交给旧COCO80报告/比较/图像校验器。界面明确不可用，原训练报告与训练比较仍是另一个入口。没有新增GPU、模型类型、自动训练、自动晋级、质量门禁、服务切换或设备认证。

## 验收

本轮的真实CPU/ONNX、独立重算、故障及全部回归结果记录在[专项验证](dataset-model-evaluation-verification.md)。它们不等同远端CI、部署或生产升级。
