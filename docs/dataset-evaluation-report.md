# 自训练模型独立评估报告

本增量只为成功的 `dataset-trained-serving-evaluation/v1` 记录增加只读报告导出与标准库离线一致性校验。接口仍是鉴权 `GET /model-evaluations/{id}/report`；原导入 COCO 的 `independent-evaluation-report/v1` 保留原结构和字节语义。比较与错误图预览仍不支持新协议，原训练期报告是另一个入口。

## 身份、字段与协议

新 schema 为 `dataset-trained-independent-evaluation-report/v1`。外层字段与原报告相同，`provenance` 增加独立的 `training_dataset`，保留原训练数据版本和 train/val/test/evaluation 摘要。`model.evaluation_id` 是原训练登记时的评估 ID，不是本次 `evaluation.id`（mev）。模型/run/attempt/原评估、产物和 manifest 摘要均保留，不重建或替换历史执行环境。

`provenance.dataset` 为本次 val-only 评估数据版本、内容摘要、图像/框数量，`source_selection_sha256` 固定 null。1–20 个类别严格有序、名称唯一，只有内部类别映射，不伪造 COCO source_category_id。完整协议固定 160 输入、score≥0.01、类别内 NMS0.65，保留 resize、最大检测数和指标/匹配协议。不能与训练时0.001或导入COCO的416/>0.3指标混用。

`exported_result` 含新协议 discriminator、汇总/逐类指标、操作点、原执行环境、scope 和固定六条警示。无GT保留null，有GT无预测保留0。只投影安全白名单，不含图像、样本路径、错误样例、原始预测、模型文件或任意警示文本；未知字段、未知警示或不一致的来源不被静默丢弃。

## 校验与只读边界

先校验所属租户，再检查状态。成功记录通过同一次 captured row 的已有详情校验，包括模型/原训练跨表血缘、当前产物、新评估快照、结果及私有预测证据。未成功状态返回409 `evaluation_not_succeeded`，受损来源返回409 `evaluation_integrity`，其他租户或不存在返回404。不会重新推理、读取原训练图片、初始化schema、生成签名密钥或写入数据库/模型/训练结果。

沿用[原报告传输约束](evaluation-report-contract.md)：精确无query GET路由、512KiB、固定文件名 `independent-evaluation-report.json`、JSON、Content-Length、no-store、nosniff、不压缩或range。相同来源重复导出相同字节。

规范化仍为 Python sort-keys/ASCII/compact JSON，不是JCS。报告、导出结果、协议、类别映射的摘要可重算。`source_result_sha256`、`source_binding_sha256`、`source_prediction_sha256` 仅为引用；因为不含原字节，离线校验不能证明这些引用对应的源证据。全部摘要均非签名，可被修改者重新计算。

`verification` 明确内部一致性、不认证来源、无签名、未重评估、不包含源结果/绑定/预测字节。`limitations` 精确为 production_allowed=false、official_coco_benchmark=false、device_certified=false、cross_dataset_overlap="not_audited"。没有导入模型的权重许可认证字段，也不声称训练数据许可已认证。独立任务及不同数据版本ID不等于已排除图像泄漏或证明泛化能力。

## 离线使用

同一现有入口严格区分两种schema；只依赖标准库的两个 portable schema 模块，不导入训练、ML、数据库或运行目录：

```sh
python -S -m scripts.validate_evaluation_report independent-evaluation-report.json
```

也可从其他目录直接执行受信源码中的 `scripts/validate_evaluation_report.py`。输入必须为有界稳定普通文件，不跟随符号链接或读取FIFO；拒绝重复键、非规范字节、非有限数、过深/过大输入、未知字段、协议混搭、血缘/类别/计数/比率/空值/环境/摘要不一致。通过只说明报告内部一致性，不代表模型质量、来源真实性或生产批准。

## 验证证据

`tests/test_dataset_evaluation_reports.py` 包括1/20类、中文、zero/no-GT、原训练val/test分支、全部嵌套对象未知字段、重算摘要后的语义篡改、只读/幂等下载、租户与状态、真实自生成常量ONNX监督评估后的API下载和标准库CLI。`tests/fixtures/dataset-evaluation-report.json` 是自生成常量ONNX的实际测量结果，仅用于契约测试，不是学习模型质量结果。其它类数/null变体是明确的合成契约数据。

本地全量、真实训练模型与浏览器验收由此次交付证据单独记录；本文件不将局部测试称作外部CI，不表示已提交、合并、部署或迁移。
