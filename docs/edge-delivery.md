# 既有 ONNX 模型交付（P0）

本增量提供**已经登记的 ONNX 部署产物**及版本化交付清单，不重新训练、不转换模型、不部署到设备。ONNX 图不是训练 checkpoint：没有优化器状态、训练进度或断点续训能力，也不是 PyTorch `.pt`/pickle 权重恢复接口。

适用配方：`fashion-cnn/v1`、`yolox-nano-cat/v1`、`yolox-nano-dataset/v1`。二分类/UCI 的 JSON 数值模型保持原有能力，但不支持本 ONNX 下载接口。没有外部模型上传、任意路径下载、TensorRT engine、RKNN、量化、设备安装或机器人/无人机控制。

## 鉴权、归属与两条只读路由

所有请求使用现有本地 Bearer 身份；tenant 由身份决定，不从参数指定。不得将 token 放入 URL 或 shell 命令行。非本租户和不存在的模型均返回 404；成功读取清单不使下载 URL 成为公开或免鉴权链接。

| 路由 | 成功响应 |
| --- | --- |
| `GET /models/{id}/delivery` | JSON，`schema_version: model-delivery/v1` 的交付清单 |
| `GET /models/{id}/artifact` | 登记 ONNX 的原始二进制，`application/octet-stream`；下载文件名为 `<model_version_id>.onnx` |

清单中 `artifact.download_url` 为 `/models/{id}/artifact` 相对路径。客户端只应在原受信任的本地 API origin 下解析该路径，使用同一租户鉴权下载；它不是本机文件路径、远程对象存储 URL 或带签名公开链接。API 仍固定 loopback，不能为了端侧设备访问而改成公网监听或添加公开反向代理。由操作者在合适权限下取得产物，再另行评审目标设备交付流程。

### 校验与错误

清单和下载都会先解析模型归属，检查登记身份和大小预算，再读取并验证原文件大小、SHA-256 与读取期间的稳定性。下载发送已验证的私有快照，不能将校验后被替换的原文件直接流式发出。服务器拒绝符号链接和非普通文件；它不接受任意路径参数。

- 400 / `range_unsupported`：不支持 Range 分段/续传下载，应重新请求完整产物
- 401：未鉴权或身份无效
- 404：对象不存在或不属于调用者
- 409 / `delivery_format_unsupported`：不是受支持的登记 ONNX 模型；不自动转换 JSON 或返回伪 ONNX
- 503 / `artifact_unavailable`：登记产物文件缺失
- 503 / `artifact_integrity_failed`：登记身份、文件类型、大小、摘要或读取稳定性校验失败；不会把损坏文件作为成功下载交付
- 429 / `download_capacity`：同时占用的产物快照达到 2 个；清单校验也占用该预算。通用 API 速率/连接限制继续适用

成功二进制响应含 `Content-Length`、`X-Artifact-SHA256`、摘要 ETag、附件文件名及 `Cache-Control: no-store`；不提供分段或条件缓存下载。客户端仍须检查实际收到的长度和摘要，网络中断不能算成功交付。

错误只返回受控 JSON 错误码/信息，不公开本机路径、凭据、原始异常或其他租户对象。下载没有发布、修改模型或质量审批副作用。同一 OS 用户/本机管理员仍可修改运行文件；这些检查不能代替 OS 权限隔离或不可篡改存储。

## `model-delivery/v1` 清单

客户端应核对 `schema_version`，按固定模型版本保存清单与产物，并在使用前自行复核 `artifact.bytes` 与 `artifact.sha256`。不要把文件扩展名当作格式证明；内部存储历史后缀不改变下载的 ONNX 二进制内容。

- `model_version_id`、`recipe`：对应不可变登记模型及来源配方
- `artifact`：部署产物类型标记、format、media_type、filename、相对 download_url、sha256、bytes、onnx_opset 与 `external_data: false`
- `input` / `output`：图的 tensor 名称、dtype、布局、shape 和输出语义
- `preprocessing` / `postprocessing`：图外调用协议；**下载 ONNX 不等于可以省略这些步骤**
- `classes`：有序 `{id, name}` 类别映射；自定义数据集必须使用实际映射，不得硬编码 Cat
- `runtime`：已记录的 target_profile、`CPUExecutionProvider`、已验证依赖版本；`supported_version_range: null` 表示没有承诺广泛版本兼容；`device_certified: false`，TensorRT/RKNN 明确 unsupported
- `lineage`：run、attempt、evaluation、dataset、配置摘要以及提交/执行环境；结合 `evaluation` 保留原训练和评估事实
- `quality` / `scope`：保留配方质量限制，`production_allowed` 固定 false；下载成功不会翻转生产准入，也不会重新声明业务精度或实机安全

### YOLOX 检测协议

输入 `images` 为 float32 NCHW `[1,3,160,160]`；原图 RGB uint8 HWC，每边 1..384，不做 EXIF 自动旋转。Pillow bilinear 等比例缩放，整数实际尺寸，左上放置、右/下补 114，然后 RGB→BGR、CHW，保持 float32 0..255，**不除以 255**。

输出 `detections` 为 `[1,525,5+C]`，每行 `[cx,cy,width,height,objectness,class_scores...]`；坐标是补边输入尺度上的已解码像素，grid decode 与 sigmoid 已在图内，不能再次执行。

图外后处理以最大类别分数选择 class，同分取较小 class ID；confidence=objectness×该类别分数。服务阈值 `>=0.01`，评估阈值 0.001，两者不可混用。将 cxcywh 转 xyxy，按实际 resize 的 x/y 比率分别还原原图坐标，裁剪、丢弃退化框；框是连续原图像素的半开区间，不加 inclusive `+1`。按类别 NMS，IoU>0.65 才抑制，同分稳定排序，每图全局最多100框；空框有效，非有限输出拒绝。具体以清单及 [检测协议](detection.md)、[数据集服务协议](dataset-serving.md) 为准。

### Fashion 分类协议

输入 `pixels` 为 float32 NCHW `[batch,1,28,28]`，batch 1..32；原始灰度像素转换为 float32 后除以 float32(255)，不隐式 resize。输出 `logits` 为 `[batch,10]`，不是概率；图外使用减去行最大值的 float64 稳定 softmax，再 argmax（同分取第一个类别）。完整类别顺序由 `classes` 给出。详见 [Fashion 指南](fashion.md)。

## 目标设备仍需单独验收

已有 CPU 评估与 ONNX 一致性结果仅适用于记录的源码、依赖和输入协议。P0 没有证明 ARM、Jetson、Rockchip、厂商 NPU 或特定设备可运行，也没有转换/量化后的精度、延迟、内存或功耗数据。

后续若另行实现 TensorRT/RKNN 等转换，应保留原 ONNX 与新产物的不同版本/摘要，重新验证输出一致性、相关精度、预后处理、资源预算及实机行为。不能复用原 CPU mAP 作为转换后的验收结果，更不能把加载成功、下载成功或 serving ready 当作生产决策、机器人安全或飞行安全许可。

## 本地下载与核验

先按 [CPU Worker 部署指南](cpu-worker-deployment.md) 配置本地 API、已有合法身份及运行目录。下载已存在的模型不要求重新运行训练，也不要求为下载而新建身份。输入模型 ID 应来自当前租户 `GET /models` 或成功 run 的 `model_version_id`。

现有 Python 标准库即可运行核验脚本；将下面占位模型 ID 换成自己的实际模型 ID，输出目录必须尚不存在、其父目录必须已存在：

```sh
.venv-cpu/bin/python scripts/verify_edge_delivery.py \
  --model-id model_REPLACE_WITH_OWN_MODEL_ID \
  --output-dir "$MP_HOME/delivery-first-check"
```

默认读取 `$MP_HOME/credentials.json` 的 `demo` 身份（未设置 `MP_HOME` 时为 `.runtime`），API 使用 `MP_API` 或 `http://127.0.0.1:8080`。可用 `--tenant` 指定已有租户名、`--credentials` 指定私有凭据文件路径、`--api` 指定其他本地端口；这些选项不接收 token 本身。脚本限定 HTTP `127.0.0.1` origin、禁用代理与所有跳转，并要求清单下载路径与模型 ID 精确匹配，避免把鉴权发送到其他目的地。

脚本独占创建输出目录，不覆盖旧结果。成功时保存 `manifest.json` 与 `model.onnx`；本地固定名称 `model.onnx` 与清单建议的 `<model_version_id>.onnx` 名称可以不同，身份以清单模型 ID、大小和 SHA-256 为准。标准输出为模型 ID、摘要、字节数和 verified 的 JSON，不打印凭据。

它读取清单、下载已有二进制并检查传输大小/SHA-256，**不提交训练、不创建部署、不做推理一致性验证**。真实推理与设备验收需要另外运行并记录。脚本运行证据与实际目标设备验收应分开记录，不能仅凭文档存在声称已通过。

## 本次验证记录（2026-10-15）

机器可读记录见 [edge-delivery-verification.json](edge-delivery-verification.json)。基线为 `eedf863ddb30e8fab884dc26aa45887a976ab90b`；本增量尚未推送。

- Linux x86_64 / CPython 3.12.14：`python -m unittest discover -s tests -v` 共 259 项，218 通过、41 跳过、0 失败。跳过项是未安装的可选 ML 栈及官方数据缓存，不将跳过计为通过
- 下载 7 项、心跳 9 项、下载辅助脚本 10 项，共 26 项聚焦回归通过；独立额外验证 FIFO 拒绝、读取中等长篡改、异常关闭/释放与并发容量恢复 4 项通过
- 在云电脑可见 Terminal 中，运行实际鉴权 API，下载 31,585 字节的 ONNX **手工构造未训练测试图**，SHA-256 为 `1e78c5c5ba02a503dde9a287c3070e155e6a90e22e31002e4d4796c206d86975`。ONNX checker 通过；ONNX Runtime 1.23.2 对原图与下载图执行 35 个样本（batch 1、2、32），输出完全一致、最大绝对差 0，独立 NumPy 公式亦匹配
- 上述仅证明真实二进制下载、摘要与该测试图的 CPU 执行一致性，**不是训练模型、现有 CNN 配方符合性、检测质量或设备实机验收**。该测试图通过隔离 QA fixture 登记，生产 API 没有新增模型上传入口
- 另在可见 Terminal 使用持续 Worker 和仓库 `scripts/demo.py` 完成真实标准库数值训练、登记、预测及 stop 后 503；这不替代 ONNX/YOLO 训练验证
- 本次完整检测依赖安装遇到官方锁定 Jinja2 wheel 的代理 HTTP 403；未绕过该拒绝，因此**未重新执行 YOLO 训练→下载已训练 ONNX 的完整链路**。历史已训练模型验证仍见各原验证记录，不能归为本次新运行
- 服务、临时运行目录和临时测试凭据均已清理；未改前端、推送、合并、部署到飞控或放开生产许可

可复现的下载调用见上节脚本；脚本只接受已有本地身份及受信任 loopback API。端侧消费者仍需核对协议版本、类别映射、预后处理、实际运行环境和应用目标，不能仅核对文件摘要就宣称接入完成。

## 检测 mAP 的 null / 0 与评估状态

模型详情 `GET /models/{id}` 和 ONNX 交付清单的顶层 `evaluation` 都是已登记的评估对象，
直接读取 `evaluation.status`、`evaluation.map_50`、`evaluation.map_50_95` 和 `evaluation.counts`。
本节适用于检测配方；分类配方使用其各自指标，不应强求 map 字段。

- `status="evaluated"` 且评估集中存在 GT 框：已执行评估，mAP 是数值。
  没有任何预测时 mAP 为 **0**，`counts.predictions=0`、`counts.ground_truth_boxes>0`；
  全部预测不匹配也可能为 0，所以仅凭 0 无法断定没有预测。
- `status="no_ground_truth"`：评估已执行，但整个评估集合没有 GT 框，
  mAP50 / mAP50:95 为 **null**（指标不适用），不是“没评估”，也不能填成 0。
  即使有预测，仍不能计算该 AP；参考 `counts.predictions` 解释是否存在输出。
- `evaluation.per_class[]` 的 `status="no_ground_truth"` 与 `ap_50=null` 只针对该类别。
  其他有 GT 的类别仍可评估，总体 mAP 可为数值，宏平均排除无 GT 类别。
- “未评估/评估失败”不能由 mAP=null 推断：queued/running 尚无成功登记的模型；
  失败看 `GET /runs/{id}` 的 status 与 attempts[].error_code。只有执行和登记成功才有
  可下载的模型及交付评估。本版检测 evaluation.status 只有上述两个值，
  没有另行实现 `not_evaluated` 或 `failed` 评估枚举，不应编造它们。

样本少并非返回 null 的直接规则；应核对实际评估 split、GT 数量和 status。
小样本指标即使非 null 也不能证明泛化能力或生产适用性。
