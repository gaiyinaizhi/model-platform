# 自训练模型独立评估比较

只读比较两条已成功的 `dataset-trained-serving-evaluation/v1` 独立评估。本能力与训练期实验比较、导入 COCO 独立评估比较分离；不创建评估、修改训练结果或切换服务。

## HTTP 契约

`GET /dataset-model-evaluation-comparisons?evaluation_a_id=mev_...&evaluation_b_id=mev_...`

须鉴权，两个 ID 必须合法且不同。同一模型的两次独立评估允许比较，也允许不同训练模型。严格只接受这两个查询参数，拒绝重复、未知参数及其他方法。两个同租户记录的归属在同一只读 SQLite 快照内确认，先于协议、状态及完整性检查；不存在/跨租户为404，非法/相同ID为400，协议不支持为409 `evaluation_comparison_unsupported`，非成功为409 `evaluation_not_succeeded`，完整性失败为409 `evaluation_integrity`。

响应精确字段：`schema_version`、`a_report`、`b_report`、`compatible`、`incompatibilities`、`delta`。schema 为 `dataset-trained-independent-evaluation-comparison/v1`。两个报告是原始规范 UTF-8 JSON **字符串**，不是被 JavaScript 重新序列化的对象；每个仍通过现有报告校验。单报告512KiB，外层2162688字节上限，超限拒绝而非截断。报告中的哈希仅为内部一致性证据，不是签名或来源真实性证明。

兼容时 `delta` 含 `overall` 以及按类别ID对齐的 `per_class`：

- overall：map_50、map_50_95、ar_100、precision、recall、tp、fp、fn。
- per_class每项：category_id、name、ap_50、ap_50_95、ar_100、precision、recall、tp、fp、fn。
- 所有差值方向为 **B−A**。比率差为原始小数差，页面可显示百分点，不是相对提升率。计数差为有符号整数。
- 任何一侧为null时差值仍为null，真实0不转换为缺失。没有真值的类别不会被当作0分。

## 比较条件

按固定顺序核对：dataset.version、dataset.content、dataset.scope（purpose/split/images/boxes/source_selection_sha256）、categories、category_mapping、protocol、protocol_digest、metric_protocol、metric_evaluator、metric_parameters、operating_point（仅协议和阈值）、submission.code/python/platform/dependencies、execution.code/python/platform/dependencies。

必须是同一冻结评估版本，不能只凭另一个版本内容摘要相同就放行。固定160输入、score≥0.01、class-aware NMS0.65等协议保持原样。代码摘要使用完整源码哈希，跨代码变更的结果会被保守判为不兼容；比较的是两个记录的环境，不是当前服务器环境。

任一条件不同，`compatible=false`，按上述顺序返回 `{field, reason: field + ' differs'}`，且整个delta为null，不返回局部排名。相同评估内容及类别但相互矛盾的真值数量属于完整性错误，不是假装可比较。

模型ID、模型产物/manifest、训练run/attempt、原训练评估ID、训练数据版本、独立评估ID及时间、结果/预测/报告摘要允许不同。两侧报告保留各自训练血缘；不同训练数据不妨碍同一测试条件下的测量比较，也不证明超参数差异导致因果效果。

## 范围与验收

不产生总胜者、生产许可、统计显著性判断、无泄漏认证、自动发布或切换。跨数据集重复/泄漏仍未自动审计。没有新增比较文件离线格式/CLI；原两份评估报告仍可分别下载和离线校验。错误图预览、教学视频链路不在本次范围。

专项测试 `python -m unittest discover -s tests -p test_dataset_evaluation_comparisons.py -v` 包含规范报告、两类比较隔离、全部兼容字段、不同训练血缘、0/null、1/20类别、正负差值、反向比较、权限优先级、HTTP查询与真实自生成ONNX的只读评估。自生成/合成小数据只证明执行机制，不证明模型质量；完整回归和真实训练浏览器验收以交付证据为准。
