# 受限预训练 YOLOX-Nano 导入

日期：2026-10-18。本切片已实现单一审查字节的上传、独立验证、显式登记和 CPU serving。**全量回归及独立安全复核已通过；可见浏览器验收另行记录**。不支持任意 ONNX、URL 下载、训练恢复、设备转换或生产使用。

## 来源与许可边界

- 用户提供原始 `yolox_nano.onnx`，声明来自 [官方发布项](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx)
- 唯一准入 SHA-256：`c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`；长度 **3,659,407 字节**
- 官方 release asset ID `42724905` 元数据长度吻合，但 publisher digest 为 null。此关联不是发布者密码学认证；API 明示 `publisher_authenticated:false`、`provenance:user_supplied_reviewed_hash`
- 上游代码固定于 commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`，采用 [Apache-2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/LICENSE)。本仓库已有[完整许可](../model_platform/yolox_nano/LICENSE)；适配来源及修改说明见 [NOTICE](pretrained-import-NOTICE.txt)
- 仓库代码许可不自动证明该权重的独立授权。目录保留 `weight_license:unverified`、`redistribution_allowed:false`；调用方须显式 `rights_confirmed:true` 确认其使用权。平台不提供本导入权重的下载或再分发，不能把本地验证视为许可审计

`catalog_id`、`protocol`、`contract`、`format` 均为 `yolox-nano-coco-onnx/v1`；不可变目录修订为 `yolox-nano-coco-onnx/2026-10-17`，重建日期不会改变资产身份。它不加入 `/recipes` 可训练目录，不创建假的 run、attempt、数据集或评估。

## 环境与离线迁移

可选依赖见 [requirements-pretrained.lock](../requirements-pretrained.lock)：官方 PyPI、固定版本、wheel 哈希、禁止源码构建。核心为 NumPy 2.3.5、ONNX 1.20.1、ONNX Runtime 1.23.2、opencv-python-headless 4.13.0.92；本切片不需要 Torch 或 pycocotools。OpenCV 为上游 INTER_LINEAR 像素一致性所需，不能无声替换成 Pillow。锁针对本次 Linux x86_64 / Python 3.12，不宣称所有平台可安装。

从仓库根目录准备独立环境：

```sh
python3.12 -m venv .venv-pretrained
.venv-pretrained/bin/python -m pip install -r requirements-pretrained.lock
```

新运行目录沿用 README 的 `init` 流程，直接建立新 schema。已有运行目录必须先正常停止 API、Worker 和全部 serving 进程，再执行：

```sh
MP_HOME=/absolute/private/runtime .venv-pretrained/bin/python -m model_platform.migrate_imports
```

这是已有 CLI 的调用示例；路径必须替换为操作者自己的私有运行目录。迁移持有 api/worker/model-import 锁、拒绝仍存活的 serving PID（仅探测，不杀存储的 PID），独占创建 `before-imported-origin.sqlite3` 备份且不覆盖旧备份。事务重建 models，保留 rowid、ID、旧 manifest 原文、摘要和全部旧关联；检查无关表/索引/触发器、外键及版本 marker。失败回滚；已有成功 schema 再执行只验证并返回 `already current`。备份冲突应由操作者检查原因，不自动删除或覆盖。

`origin=trained` 必须有 run/attempt/evaluation 且无 import/validation；`origin=imported` 相反。复合外键和登记/读取校验共同绑定 tenant、import、validation、目录修订及摘要，避免同租户或跨租户拼接。普通 GET 不触发模型表破坏性迁移；旧 schema 创建导入返回 `409 migration_required`。

## API

所有路由使用现有 Bearer 鉴权和租户身份；不接受 body tenant、来源 URL 或本机路径。跨租户对象返回 404。

| 方法 / 路径 | 请求与响应 |
| --- | --- |
| GET /model-import-catalog | 返回 `entries`、固定目录及权利/来源限制；enabled 不代表当前可选依赖或模型质量已就绪 |
| POST /model-imports | 恰好一个 `Idempotency-Key`；JSON `{"catalog_id":"yolox-nano-coco-onnx/v1","rights_confirmed":true}`；首次201，同key/body重放200，冲突409 |
| GET /model-imports/{id} | 状态、reason、artifact_sha256、validation_id、validation 报告、model_version_id；不暴露文件路径 |
| POST /model-imports/{id}/artifact | 原始 binary，唯一 Content-Type 为 application/octet-stream，唯一 Content-Length 必须是3659407；拒绝 Transfer-Encoding、Content-Encoding、ZIP/multipart；同步验证后返回200及状态 |
| POST /model-imports/{id}/register | JSON `{}`；仅 validated 可登记，首次201、重放200且同一 model_version_id |

状态为 `created → validating → validated | rejected`。校验失败通过返回的 `status:rejected` 与 `reason` 表达，不能仅凭 HTTP 200 宣称成功。完整字节绑定后相同字节重放不重新验证；不同字节409。rejected 为终态，重新尝试须新 key/new import。上传未完成或超时不绑定字节，可重试原 import。API 恢复将遗留 validating 转为 rejected/validation_interrupted，不自动重试。

登记后沿用 `/models`、`/models/{id}`、`POST /deployments`、`/deployments/{id}/predict` 和 `/stop`。模型及预测身份带 origin/import/validation/protocol/hash/classes，`evaluation:null`、`local_evaluation_status:not_evaluated`、`production_allowed:false`、`device_certified:false`。无 recipe/run/attempt/evaluation_id。导入模型的 `/delivery` 和 `/artifact` 拒绝为 `409 delivery_format_unsupported`。

没有新增免鉴权入口或凭据。用现有客户端的 Bearer 配置调用上述流程，不把 token 粘贴到文档或命令历史；真实接口调用示例可参考 [生命周期测试](../tests/test_import_serving.py)，它仅创建可清理的临时测试身份。

## 独立推理协议

输入 `features` 为长度1数组，元素恰为 `{width,height,pixels}`：宽高整数1..384，pixels 为扁平 HWC RGB 整数0..255，长度 width×height×3，拒绝 bool/额外字段。模型张量为 `images:float32[1,3,416,416]`，输出 `output:float32[1,3549,85]`；IR6、opset11、369节点、236初始张量、904481初始元素。

1. RGB 转 BGR，OpenCV INTER_LINEAR 按 `r=min(416/h,416/w)` 缩放为 `int(w*r),int(h*r)`，贴左上、右下填114；CHW float32，值0..255，不归一化
2. 按 stride8/16/32 和 grid 解 raw xywh；不重复 sigmoid。坐标从 cxcywh 转 xyxy，除以同一理想比例 r；不采用现有训练协议的独立 realized x/y 比例
3. objectness×class score，80类 argmax；严格 score>0.3。使用上游默认 class-agnostic NMS，IoU 阈值0.45、面积及交集均用 inclusive `+1`；按 NumPy argsort 逆序，保留固定 NumPy 下的同分行为，不保证跨版本同分顺序
4. **平台额外适配**：NMS 后裁剪到请求图像边界，去除退化框，最多100框。此步单独测试，不声称未经适配的上游坐标与平台坐标无条件相同

输出为 `[{width,height,detections:[{class_id,score,box:[x1,y1,x2,y2]}]}]`。class_id 为连续0..79；COCO 原 category IDs 独立存储（cat index15/category17）。80标签及有序 ID 表的规范 JSON SHA-256 为 `5655e1a2c432e4cbf3f7417f6d0aa90d42ea03ec0880951aeba2b5fcbb3afc9d`。空框列表合法；不把空结果作为错误或质量指标。

## 安全与资源限制

- 解析前强制精确长度/hash；不匹配字节永不进入 ONNX 解析器。拒绝 custom domain/op、动态 I/O、external_data、sparse tensor、function、training graph、subgraph、metadata 及超限张量
- 目录逐级 dirfd/O_NOFOLLOW，普通文件及读前后 stat 校验，服务端 artifact ID、0600文件、原子替换/fsync；登记及重载再验身份/hash。ORT 读取已验证 bytes，CPUExecutionProvider，线程1，无扩展库或调用方 SessionOptions
- 验证独立子进程：30秒 wall、20秒 CPU、2GiB地址空间；实际静态 checker、ORT raw probe 及完整预后处理 probe 均须通过。每导入 serving 2GiB地址空间、加载30秒 alarm、每请求预测3秒 alarm；非有限或超时失败闭合，不保持 ready
- 上传15秒绝对期限；运行目录全局64个 import记录、256MiB已占用导入文件预算、单次 import操作锁；原 API16连接、120请求/tenant/min及最多4个desired-running部署预算继续适用
- 无自动删除/清理权重和记录；事务失败留下的孤儿不可部署，磁盘预算包括已存在文件。备份和恢复仍需操作者管理

以上是受限可信本机方案。地址空间上限不是 RSS 配额，rlimit/进程隔离不是 cgroup、容器、独立 OS 用户或生产多租户安全沙箱。精确hash是准入边界，checker/算子名单不能充当任意恶意图沙箱。

## 本轮新验证与复现

2026-10-18 重新恢复源代码、重新读取用户资产、重新获取固定上游源码，并重新下载12张公开照片；照片哈希与恢复的源归属清单逐张一致。没有把此前机器丢失前的结果当作本轮新证据。

- runtime 12/12：安全打开、hash前置、超时、输入、decode、NMS、adapter、真实asset加载/篡改及上游helper一致性
- 12张真实照片产生28个上游参考检测；预处理像素、decoded tensor和NMS记录均精确一致。照片先明确缩到max384再请求。额外3组合成输入及合成score用于非空/跨类/同分情况；不将其称为识别质量结果
- 新测一次验证 wall0.290秒、child0.215秒、peakRSS138792960字节；仅本机观察值，不是性能承诺
- registry16、dispatch3、真实 lifecycle3通过；完整可选依赖回归297项中296通过、1项官方数据缓存测试skip，实际导入资产及上游照片一致性已运行；先前宿主Python（含NumPy、无ONNX/torch）297项中246通过、51项可选依赖/资产测试skip；发布前新建无系统包的纯标准库venv复测297项，205通过、92项明确可选跳过。registry/dispatch包含合成或mock边界检查，不替代实际ONNX探测；独立安全复核已通过可信本机范围，可见浏览器另行记录
- 未计算本地 AP、mAP、COCO benchmark，未认证设备或生产质量

已脱敏机器证据见 [pretrained-import-verification.json](pretrained-import-verification.json)。源码、不含权重或照片；公开照片许可/署名见[既有车辆验收记录](https://github.com/gaiyinaizhi/vision-lab/blob/4bf19a68744d2fc00aac70eb6eae0ea02e0baf40/docs/verified-model-delivery.md)。

基础测试无需资产：

```sh
.venv-pretrained/bin/python -m unittest discover -s tests -p test_pretrained_yolox.py -v
.venv-pretrained/bin/python -m unittest discover -s tests -p test_model_imports.py -v
.venv-pretrained/bin/python -m unittest discover -s tests -p test_import_dispatch.py -v
```

真实资产和固定helper由操作者事先合法取得；测试不网络下载。设置 `MP_PRETRAINED_YOLOX_ASSET` 为该精确ONNX私有路径，`MP_YOLOX_REFERENCE_DIR` 为含固定commit的 demo_utils.py/data_augment.py/coco_classes.py 的目录，可选 `MP_YOLOX_PHOTO_DIR` 为已授权JPEG目录，然后运行：

```sh
.venv-pretrained/bin/python -m unittest discover -s tests -p test_pretrained_yolox.py -v
.venv-pretrained/bin/python -m unittest discover -s tests -p test_import_serving.py -v
```

缺资产/依赖/reference时相关测试明确skip；不能把skip写为实际推理通过。固定helper文件在执行前校验 SHA256，摘要见证据JSON。全量回归命令仍为 README 的 unittest discover；本轮结果见上方及证据JSON。
