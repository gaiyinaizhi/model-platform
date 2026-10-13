# 业务接入增量契约（2026-10-13）

本增量保持静态配方、SQLite 单机与原模型技术血缘不变，补充鉴权配方发现、可选部署幂等和显式历史分页。没有开放配方注册、权重上传或生产发布，没有改变模型质量与资源隔离边界。所有路由仍要求平台自己的 Bearer 身份；业务后端保管 token，浏览器不持有它。

## 配方发现

`GET /recipes` 返回 `{ "recipes": [...] }`。只有列表，无上传或注册端点；静态四配方均返回，不根据本机安装情况过滤。每项保留 `id, contract, format, runtime, classes, data_mode, optional_dependencies, budgets, capabilities`，并增加：

- `task_type`：`classification` 或 `object-detection`，不根据类别数猜测任务类型
- `availability`：`dependencies/data/worker` 都为 `unknown`。`optional_dependencies` 仅声明要求；capabilities 仅描述协议实现，不能作为安装、数据就绪、Worker 存活、服务 ready 或模型质量证明
- `training.fixed_config/fixed_dataset`：固定引用配方给出必须完全匹配的值；内联数值配方为 null，并提供 `config_schema`
- `prediction.request_schema`：POST predict 的 `{features: [...]}` JSON Schema；`predictions_schema` 描述响应中 predictions 数组，不改变原响应封装；`semantics` 声明无法仅用 JSON Schema 表达的维数、RGB 长度、框边界等约束
- `quality`：`production_allowed: false`；一般配方 `status: development-only`。YOLOX 为 `status: quality-failed`，`warnings` 包含 `high-confidence-extreme-raw-boxes` 和 `clipped-boxes-do-not-prove-quality`

元数据不含回调、文件路径、凭据、数据内容或内部游标签名材料；请求不导入可选 ML 库。发现 schema 是描述性契约，服务端原校验器仍执行完整相关性及预算检查。YOLOX 在线单张 RGB 图为 `features: [{width,height,pixels}]`，输出每图 `{width,height,detections:[{class_id:0,score,box:[x1,y1,x2,y2]}]}`，空检测有效。高置信极端原始框及禁止生产说明见 [detection.md](detection.md)。

## 部署幂等

`POST /deployments` 仍仅接受 `{model_version_id: "model_..."}`。`Idempotency-Key` 可选；若提供只能有一个头，1..128 字符，拒绝控制字符。首次创建返回 202，重放返回 200，返回结构仍为 deployment 当前状态。

- 未提供 key：完全保留旧行为，每次新建一个部署；调用方遇到不确定结果不能安全盲重试
- 同 tenant、同 key、相同 canonical 输入：永远返回原 deployment ID；空格与 JSON 序列化差异不影响输入摘要
- 同 tenant、同 key、不同输入：409 `idempotency_conflict`；其他 tenant 的 key 独立，model 对象仍要求归属本 tenant
- replay/conflict 在全局四个 desired=running 配额检查之前；满额时已成功请求可重放，不消耗新配额
- 通过 SQLite `BEGIN IMMEDIATE` 同事务保存部署、key 绑定及单条创建 audit。并发或响应丢失后的重试不会重复创建；API 重启后绑定仍在
- 未成功创建（如404、429）不消耗 key。已 stopped 或 failed 的部署重放保留原状态，不隐式重启或新建；启动另一个部署须使用新 key
- key 无自动到期或回收。独立映射表不改变旧 deployment 行、旧模型 manifest 或训练输入/摘要。现存无 key 部署不能事后补绑定

停止、预测、后台启动和 desired/observed/ready 生命周期原契约不变。幂等接收成功不等于服务 ready，更不等于质量合格。

## 历史分页

适用于 `GET /runs`、`GET /models`、`GET /deployments`。

- 无 query：保持旧响应 `{ids:[...],limit:100}`，最多最新100条，不新增 next_cursor 字段
- `?limit=N`：N 为1..100，返回 `{ids:[...],limit:N,next_cursor:"..."|null}`
- 后续 `?cursor=...`：继续原页大小；若同时提供 limit，必须与原值一致；最后一页/空集合的 next_cursor 为 null
- 仍只返回 ID，不提供批量详情、总数或状态过滤；调用方按 ID 读取原详情路由。`limit` 和 `cursor` 以外、重复 query、其他路由 query 均400

固定按 `created_at DESC,id DESC` 排序，同时间戳不会丢失/重复。首屏在单个 SQLite 读事务中捕获 rowid 高水位，之后插入的记录即使回填旧时间也不会进入该次遍历。当前 API 没有删除集合记录的入口；该快照只固定成员范围，不冻结 run/deployment 可变状态。技术记录、created_at/id 不由 API 改写。

cursor 为最多2048字符的 URL-safe token，包含版本、tenant SHA-256摘要、collection、页大小、高水位与最后位置，以独立随机256-bit内部完整性 key 做 HMAC-SHA256。token 是签名而非加密，不应放敏感数据；它不能替代 Bearer。服务端仅解析有界 JSON，拒绝篡改、跨tenant、跨collection、格式/范围不合法的 cursor（400 `invalid_cursor`）。query 总长限制2200字符。

完整性 key 在私有运行目录以0600文件存储，用完整临时文件加原子硬链接发布，支持并发首次初始化。它不是身份凭据，不从身份 token 派生。API重启继续有效；没有时间过期。删除/替换 key 会令旧cursor失效，调用方需从第一页重启；没有承诺跨数据库恢复、人工修改或删除的cursor稳定性。SQLite schema 增量只加key映射表及集合排序索引，旧技术事实不重写。

## 验证边界

新增测试在 `tests/test_integration_api.py`：真实loopback HTTP、真实SQLite、二分类训练/独立持续推理、并发/重启/停止/配额、超100条历史、稳定快照、鉴权及游标安全。签名密钥初始化及旧schema迁移有专项验证。检测发现是描述真实既有配方，不将数值测试或合成ONNX测试描述为新检测训练验收。本次没有修改检测训练参数、阈值、已登记模型或旧质量证据。
