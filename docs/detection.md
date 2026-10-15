# YOLOX-Nano 真实检测 CPU 切片

状态：2026-10-12 本地 CPU 训练/评估/持续服务功能闭环已验收，质量未达到生产使用要求。

**禁止将此小样本从零训练检测器用于生产决策。** 最终 ONNX test 候选中有17个原始坐标量级超过77500，最大约3777117.25，最高score为1.0。后处理按公开契约裁剪到图像边界并进行NMS，因此在线框合法不代表原始回归稳定或置信度已校准；不能把这些候选一概称为低置信、无影响。此阶段不根据test结果调整模型或阈值挽分。

## 独立任务契约

- recipe `yolox-nano-cat/v1`，contract `rgb-cat-detection/v1`，format `yolox-nano-onnx/v1`；原三分类契约保持兼容，不修改 vision-lab
- 固定 Open Images Cat 叶类数据缓存引用，API 不接收 URL、文件路径、可执行模型或任意权重
- POST /deployments/{id}/predict 仍用 `features` 顶层键，但此配方数组仅允许一张 `{width,height,pixels}`；RGB、逐行 HWC 展平、整数0..255，边长≤384，极端比例缩放零维拒绝
- 每图输出 `{width,height,detections:[{class_id:0,score,box:[x1,y1,x2,y2]}]}`；Cat=0，box 为原图半开区间绝对像素坐标，不使用 inclusive +1；空 detections 是有效推理结果，不表示错误
- 固定160×160、Pillow bilinear缩放，左上放置/右下填114，RGB转BGR CHW float32，保留0..255值。使用整数实际resize尺寸分别计算x/y缩放与坐标还原；不同于OpenCV插值的数值误差不能混用
- confidence=objectness×最大类别概率，服务阈值0.01，评估阈值0.001，类内NMS IoU>0.65抑制，按score降序并对同分稳定排序，最多100框。低服务阈值为从零训练工程演示，不是生产建议

## 模型、训练与来源

官方 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/tree/6ddff4824372906469a7fae2dc3206c7aa4bbaee) 固定 commit 的 Apache-2.0 模型源码子集保留许可及原始sha256。Nano depth=.33、width=.25、depthwise=True，896754参数（单类）。保留原SimOTA及IoU/objectness/class loss；移除CUDA OOM fallback、可视化、loguru，CPU float32无自动混合精度。未安装官方Trainer，未下载预训练权重，不推断权重许可。

在观察val/test结果前固定：seed20261012，20 epochs，Adam lr.001，batch4，160输入，从零训练，无Mosaic/预训练/测试选模。固定128训练、32验证、32最终测试。验证集用于独立协议检查，不选阈值、epoch或模型；最终test在训练完成后评估；技术回归可以重复同一固定过程，但不得据此调整训练参数或选模型，重复运行须如实记录。低/零mAP必须原样报告。

源码摘要递归包含模型子目录；旧登记版本摘要不改写。模型输出ONNX只在受控Worker自生成；读取前核验大小与SHA256，限制metadata/operators/无外部张量/无嵌套图，不开放上传。运行时沿用独立持续服务、不可变登记、停止阻断及API重启恢复。预算都是协作式逻辑限制，不是硬CPU/RSS沙箱。

## 数据与评估

公开数据的来源、逐图片许可/归属、元数据版本、原始字节与像素hash、固定选择及跨split审计由下载器manifest记录。只能使用官方human-verified Cat正/负图，未知标签排除；正图保留全部Cat框，任何Cat group-of整图排除，occluded/truncated实例仍保留。不是任意筛选检测框后把未标注类别当负样本。

官方pycocotools COCOeval bbox：IoU .50:.05:.95、101 recall点、useCats=1、area=all、maxDets=[1,10,100]，报告mAP@.50和mAP@.50:.95。数据子集无ignore/crowd，遇到这些字段为true明确拒绝；OI group-of不等价于COCO crowd。零GT类别AP为null且不参与均值，零预测但有GT时AP=0。这是独立固定小切片上的COCO-style评估，**不是官方Open Images benchmark**，不与其官方指标直接比较。

## 验收清单

1. 纯逻辑测试：RGB形状/预算、letterbox/坐标还原、NMS、边界/NaN、COCO标准已知答案
2. 来源/许可/sha256、训练验证测试完全分离、固定split及缓存拒绝篡改
3. 真实CPU参数优化、ONNX导出、原始与ONNX原始输出/框结果/评估一致性
4. 真实HTTP训练登记、持续框预测、停止503、API重启后相同版本/产物服务恢复
5. 原三分类全部回归和真实demo，源码冻结后重跑；产物/图片/日志不提交

实际验收见下节及 [结构化证据](detection-verification.json)。

ONNX 一致性同时验证原始tensor（atol .001、rtol .0001）以及0.001评估/0.01服务两阈值的框集合；使用确定性一对一完美二分匹配，每条边必须满足相同class、score差≤1e-4、四坐标最大差≤0.1原图像素。框数量必须一致，不能重复复用候选或把漏框当一致。两后端AP仍按各自实际预测分数独立计算，差异≤1e-6；不会按匹配结果改写COCO排序。

## 实际验收（2026-10-12）

- 真实训练128图110框，validation32图26框，test32图26框；从零20epochs，896754参数，CPU训练65.09秒、导出0.62秒、两split双后端评估2.35秒（共享云CPU观察，不是基准）
- 最终test COCO-style mAP@.50 **0.5151476295（51.5148%）**；mAP@.50:.95 **0.1721211584（17.2121%）**。validation对应0.4464088835 / 0.2336161376；没有用其选择参数
- PyTorch与ONNX在32张validation和32张test上原始tensor allclose通过，0.001评估阈值与0.01服务阈值各自框数量/完美匹配通过，两后端mAP完全相同
- test保留框最大坐标差0.00864258原图像素、score差2.76894e-6。原始输出最大绝对差7.75；**门槛是 atol=.001 加 rtol=.0001，不是所有原始值绝对误差≤.001**。巨大原始候选与质量缺陷见顶部警告
- 真实API提交/幂等/训练/不可变登记，首三张validation（事先固定选样）在线返回6/10/7框；6次请求输出稳定，同一serving PID持续运行；停止后503，API重启后新PID重新加载同版本/同hash且输出完全一致，再停止
- 原三分类真实demo全部回归通过：二分类、UCI Digits accuracy0.9499165275、Fashion accuracy0.8076
- 最终本地含官方缓存177项可选环境测试全过；干净无数据CI副本精确排除一项真实官方缓存集成测试，其余176项全部通过且0skip。纯标准库177项中134通过、43项可选库测试明确跳过。compileall、pip check、git diff --check通过。全部本次API/Worker已正常退出，所有9个演示deployment均desired/observed=stopped

第一次真实训练在validation框逐序比较失败，被拒绝登记且未访问test；修复近分数排序引起的错误比较后进行中间回归，再升级为独立审查通过的一对一完美二分匹配进行最终验收。所有轮次训练配置不变，未根据test选参；中间与最终ONNX SHA256及test指标完全一致。失败attempt与历史版本保留，不覆盖记录。

最终ONNX 3663363字节，SHA256 `718c380b7b0fe9c0024d97b6e279e5a86269d378b207dbc84ef12025249b4c84`。最终执行源码摘要 `bedc28c7c168f31d3f68ce5b139e9e6822c24fe7eb7f66a99c663bf6b1f70159`。数据/权重/原始日志未提交，文档只保留指标、摘要及边界。

## 复现运行

沿用README的独立API、Worker流程；这三者使用同一MP_HOME及检测可选环境。

```sh
.venv-fashion/bin/python -m pip install -r requirements-detection.lock
export MP_HOME=.runtime/detection-local
.venv-fashion/bin/python -m model_platform init --tenant demo --tenant other
.venv-fashion/bin/python scripts/fetch_detection.py
.venv-fashion/bin/python scripts/fetch_detection.py --verify-only
```

终端A运行 `.venv-fashion/bin/python -m model_platform api --port 8080`，终端B运行 `.venv-fashion/bin/python -m model_platform worker`，终端C运行 `.venv-fashion/bin/python scripts/demo_detection.py`。所有终端须导出相同MP_HOME；本地演示不会向公网开放服务。首次准备还会流式读取大CSV元数据，不只是下载约58MB图片；来源变化、校验或许可记录不匹配应停止，不自动换图片。

无GPU/业务接入/设备包验收，无模型质量准入门槛、checkpoint恢复、自动回滚、硬资源隔离；技术闭环通过与模型质量合格严格区分。
