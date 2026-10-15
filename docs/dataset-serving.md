# 数据集训练模型的本地持续推理（2026-10-15）

仅供可信本地研发/研究使用。训练完成不自动部署；部署 ready 表示指定产物的进程真实通过推理探测，不代表检测质量、生产准入、外部发布或设备安全。没有新增 Sluggard 接口、外部模型上传、热切流或自动质量审批。

## 复用已有生命周期 API

所有路由继承 Bearer tenant、对象隔离、2MiB JSON/120次每分钟/4个活动部署预算。CPU 独立进程，没有硬 CPU/RSS 隔离。

- `POST /deployments`，`{"model_version_id":"model_..."}`，推荐 `Idempotency-Key`：首次202，同key同模型200，同key不同模型409；错误租户模型404。显式按钮触发，不在训练完成时自动调用
- `GET /deployments/{id}`：desired running/stopped；observed starting/unavailable/ready/failed/stopped；只有 running+真实探测成功才 ready。检查时间与稳定 error_code 一并返回
- `POST /deployments/{id}/predict`，`{"features":[{"width":160,"height":160,"pixels":[...]}]}`：只允许一张 RGB/HWC、每边1..384、整数0..255、严格匹配像素长度。返回实际 ONNX 框预测，允许空框；禁止以演示框替代
- `POST /deployments/{id}/stop`，`{}`：先保存停止期望并拒绝新调用，再确认进程结束；后续预测503

停止后的同key重放返回原 stopped 对象，绝不会重新启动。重新启用必须明确创建新 deployment、新 key；旧 ID 与模型版本历史不变。进程崩溃撤销 ready，不在同一 API 生命周期自动重启。API 重启按持久化 desired=running 重新加载原模型及探测，不能回退或替换模型；desired=stopped 不恢复。没有独立 restart 路由。

## 版本与推理协议

数据集配方 deployment 查询和预测响应增加：recipe、artifact_sha256、run_id、attempt_id、evaluation_id、dataset_version_id、dataset_sha256、train_sha256、evaluation_sha256、evaluation_split、class_names、serving_protocol、quality。原 model_version_id、deployment_id（预测）、id（deployment查询）及状态字段保持兼容。

`serving_protocol`：preprocessing=`rgb-to-bgr-pillow-bilinear-top-left-114-f32-0-255/v1`、box_format=`original-xyxy-half-open`、input_size=160、score_threshold=0.01、evaluation_score_threshold=0.001、nms_iou_threshold=0.65、max_detections=100。服务阈值与评估阈值不同；不能拿上传图预测框数量推导 mAP。类别采用绑定数据集的有序 class_names，绝不能硬编码 Cat。

`quality` 固定 production_allowed=false、status=development-only，warnings 包括 research-only 和 deployment-readiness-is-not-model-quality；这些是研发范围声明，不是新的质量验收结果。

每次子进程加载都验证产物 SHA256、格式/契约/维数/算子白名单及类别数；再核对登记模型、嵌入的类别/数据版本/train/eval摘要、评估映射/split/role/摘要/预处理/NMS/指标兼容key。无需原始数据参与在线推理。supervisor 验证每次返回的 deployment/model/artifact/contract 及数据集身份；探测按实际类别数检查预测。失败只公开稳定错误码，不向接口泄漏文件路径、凭据或堆栈。

## 验证边界

`tests/test_dataset_serving.py` 使用明确标注的合成常量 ONNX 图运行真实 ORT、HTTP 和独立进程，覆盖类别映射、真实框处理、租户404、幂等/冲突、停止503、显式新部署、API恢复、子进程退出、损坏产物拒绝以及关联字段逐项篡改。这是机械链路回归，不是训练或模型质量证明。

真实已训练产物与浏览器完整验收结果由本轮独立验收另行记录；文档和测试存在本身不等于它已通过。固定 Cat 配方、质量限制与接口保留。

### 本轮源码回归

Python3.12.14完整ML环境228项中227通过、1项官方Fashion-MNIST缓存复核缺失跳过；纯标准库环境151通过、77可选依赖测试跳过。新增3项持续服务回归全部通过（合成常量图，真实ORT/HTTP/进程），compileall和差异空白检查通过。浏览器真实训练产物验收尚由独立流程进行，不能用上述结果替代。
