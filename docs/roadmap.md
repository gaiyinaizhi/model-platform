# 实施路线

2026-10-12：独立 Python CPU 第一切片；已完成内容和实测状态见 [verification.md](verification.md)，具体契约见 [architecture.md](architecture.md)。

## 第一切片

本工程自主管理模型技术登记、产物、训练/评估和推理部署；Sluggard 是可选业务关联。实现小型二分类的真实训练、独立留出评估、不可变 JSON 模型、按版本独立推理进程、鉴权与租户隔离、查询/就绪/停止和重启恢复。提供独立 Worker，不在请求中训练。

验收覆盖幂等及冲突、输入边界、真实训练失败、attempt 历史、版本唯一性与血缘、真实推理、停止拒绝、产物损坏、进程中断与恢复。合成数据只用于合法无敏感示例，指标来自实际计算；不等于视觉或设备验收。

## 后续按需求推进

1. 成熟 HTTP 网关、TLS、运维认证、OS 级资源配额及更严格安全边界；当前只适合可信本地 CPU 开发。
2. 可确认实际停止的训练取消、更细重试策略、检查点恢复；已支持尚未领取的训练排队撤销，运行中取消仍未实现。
3. 推理更新、原子切流与回滚、质量门槛与发布流程、性能/内存基准。
4. visionlab 和教学平台各自接入稳定技术接口；Sluggard 只管理课程、业务审批/权限和关联，不成为本平台技术状态的主写方。
5. 更多数据来源和算法、受控导入、对象存储；不加载不可信 pickle 或 arbitrary code。
6. 选定机器人或无人机设备后实施格式转换和实机验证；设备安装、控制及运行安全归业务产品。

暂不建设管理前端、多 GPU、Kubernetes、多集群、设备部署、模型市场或完整 MLOps。

## 已增加的真实数据首例

UCI Optical Recognition of Handwritten Digits 官方完整3823训练/1797测试，64数值特征十分类；独立softmax配方，保留合成二分类回归。数据下载、许可、逐文件摘要、重复审计和可复现运行见[digits.md](digits.md)。该首例是预提取特征CPU训练/评估/持续推理，仍不代表原图视觉网络或设备能力。

## 已增加的深度学习 CPU 首例

Fashion-MNIST小CNN：官方60000训练记录中固定分层6000从零训练，独立完整10000 test最终评估；可选CPU PyTorch、ONNX导出与双后端全test一致性、不可变登记及ONNX Runtime持续推理/停止/恢复已验收。准确率80.76%；详细契约、限制及证据见[fashion.md](fashion.md)。尚无任意图像上传、GPU、设备运行包或实机验证。

## 已增加的维护与扩展基础

[静态内部配方协议](recipes.md) 和 [CPU CI](ci.md) 已加入源码。三套配方继续使用原训练/评估/推理契约，新增配方需源码审查及真实验证，不是上传式插件平台。未来 vision-lab 接入先匹配分类/检测任务和输出语义，再评审发现/业务网关接口；本次未实现业务接入。

## 检测与 mAP 增量

用户明确要求真实 YOLO 目标检测与 mAP，已按独立检测契约实施 `yolox-nano-cat/v1`，不以分类代替检测、不修改 vision-lab。官方 Apache-2.0 YOLOX-Nano 固定源码、从零CPU训练、Open Images Cat 固定128训练/32验证/32测试、官方pycocotools评估、ONNX框预测及版本服务生命周期。实际验收状态和质量限制见 [detection.md](detection.md)，数据许可/来源见 [detection-data.md](detection-data.md)。更多类别、GPU、业务集成和设备验收仍未实施。

## 自定义数据集第一步（2026-10-15）

已新增[YOLO 矩形框 ZIP 导入、校验和不可变数据集版本](dataset-imports.md)。已进一步新增[版本绑定多类别 CPU 训练、日志/评估/部署关联及兼容实验比较](dataset-training.md)。完整资产管理、设备与 GPU 支持仍未实施。现有固定 Cat 配方及其质量限制保持不变。

自定义数据集模型持续推理沿用已有生命周期，新增[服务身份与协议校验](dataset-serving.md)及独立进程回归；实际训练模型/浏览器验收以单独证据为准，不代表生产部署。

## CPU ONNX 交付第一步（2026-10-15）

新增鉴权 [ONNX 下载与交付 manifest](edge-delivery.md)，保留真实预后处理、类别、血缘和质量边界；新增 [独立 Worker 心跳/依赖就绪](cpu-worker-deployment.md)。实际验证结果见交付文档；尚未实现目标设备转换、实机验收、Webhook 或生产许可审批。

## 单一预训练资产导入切片（2026-10-18）

已实现精确hash白名单的用户上传YOLOX-Nano COCO资产、独立验证/登记及持续CPU推理；[范围和最新验证状态](pretrained-import.md)。局部实际资产/上游协议与生命周期回归已通过；本轮全量、浏览器和独立复核待汇总。不代表任意模型导入、权重许可认证、再分发、设备转换或生产发布。


## 导入模型独立评估切片（2026-10-18）

新增[独立评估契约与实现](independent-evaluation-contract.md)：保留训练型 train+val 数据集规则，另增 evaluation-purpose val-only；对已登记受限预训练模型运行固定服务协议的独立队列评估，记录不可变模型/数据/类别/代码/环境/预测摘要、AP50/AP50:95/AR@100 及可复核 TP/FP/FN。真实固定64张非 crowd COCO val2017 子集不是完整官方评测，不声明生产质量、设备认证或权重许可认证。失败终态须新幂等键明确再评估；未实现取消、自动重试或生产发布门槛。


## 独立评估只读比较（2026-10-18）

新增鉴权 `GET /model-evaluation-comparisons?evaluation_a_id=...&evaluation_b_id=...`，只比较同租户两条不同 ID 的成功独立评估，不要求模型 ID 不同。两侧完整性校验后严格比较数据版本、内容、样本顺序、类别映射、协议、提交和执行代码/Python/平台/依赖；任何差异返回明确不兼容原因和空差值。兼容时返回 AP50/AP50:95/AR@100、precision/recall、TP/FP/FN 的 B−A 差值，保留 null。

只读请求不初始化数据集 schema、不写审计/结果/模型、不运行推理；既有训练比较、导入登记血缘、serving 和部署限制不变。详见[已评审契约](evaluation-comparison-contract.md)及[验收记录](evaluation-comparison-verification.md)。现有单一权重的重复记录仅说明固定条件下的重复性，不代表多模型排名、生产质量门槛或官方 COCO benchmark。


## 训练排队撤销（2026-10-20）

新增[queued → cancelled 与显式离线 schema 迁移](queued-training-cancellation.md)，与 Worker 领取原子互斥并保留所有历史。运行中训练和独立评估取消、自动重试、检查点恢复仍未实现。
