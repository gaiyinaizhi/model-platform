# 稳定模型服务阶段 1 安全契约（独立审查后修订，阶段 1）

仅后端可信本地 CPU 研发；仅 yolox-nano-dataset/v1 已存在的同租户实际 ready deployment。不可变模型和 deployment 语义不变。不实施 UI、前端锁、生产部署/迁移、自动创建/停止部署、自动回退、质量准入或外部路由。

## 公共 API
- POST /services：唯一必需 Idempotency-Key（沿用既有 1..128 字符校验），精确 body {deployment_id}，严格 deployment_<32 lowercase hex>。首次 201，重放 200。
- GET /services：现有 ids/limit 有界列表及 limit/cursor 签名分页语义（默认100，上限100）；collection=services，tenant 隔离。
- GET /services/{id}：稳定 service_<32 lowercase hex>，返回配置 snapshot 及当前 deployment 的真实 readiness/checked_at/error_code，不相信持久化 ready。
- POST /services/{id}/switch：先验证 service 归属，再精确 body {expected_revision,target_deployment_id}，唯一必需 Idempotency-Key。revision 为真 int（非 bool）1..9007199254740991；目标同租户。首次及重放 200；期望 revision 过期或目标等于当前为409 service_conflict。不存在/跨租户为404，错误body400。
- POST /services/{id}/predict：精确 body {features}，沿用 dataset image input。返回既有真实 deployment/model/artifact/协议身份，加 service_id/service_revision。无法实际 ready 为503 not_ready；不自动选择 previous。
- health 增加 model_services_supported:true，旧字段不变。

## 不可变操作快照与实时状态分离
create/switch 返回持久化 configuration snapshot：id, revision, active_deployment_id, previous_deployment_id（首次null）, model_version_id, artifact_sha256, contract, class_names, serving_protocol, created_at, updated_at, endpoint=/services/{id}/predict。不包含ready或任何暗示当前可用的历史字段。
GET detail 返回相同 configuration 加 ready, checked_at, error_code, desired, observed（来自真实 query 的当前部署）。客户端必须 GET 刷新当前配置和实际状态。
所有 create/switch key 在同一 tenant 的 service 操作命名空间唯一，request hash 包含操作种类、service id（create为null）和完整请求参数。重复键同payload返回最初已提交snapshot，即使已再次切换、目标停止/损坏或期望revision已过时，也不重新probe/CAS/写入。不同payload409 idempotency_conflict。switch先确认service所有权，create重放也校验存储snapshot完整性。重放不得返回当前投影伪装原结果。
每次create为revision1；switch递增1且previous=原active。切换到previous就是显式回退，不需要单独自动回退逻辑。revision上限拒绝。append-only revision历史与key记录，禁止UPDATE/DELETE。service自身identity不可改，只能合法revision+1切换；不得删除。

## 准入与锁顺序
service兼容身份固定为dataset recipe+contract+完整serving_protocol+有序class_names，从实际成功Supervisor探测以及持久化模型ID/hash/recipe/contract/classes/protocol与live instance一致性检查生成；不能把Store对训练模型的现有校验称为完整校验。不同数据版本/训练参数/模型ID允许；类别顺序、协议阈值、配方或contract不同拒绝409 service_incompatible。
所有service create/switch/query/predict必须持有同一个Supervisor RLock；既有deployment stop/loop也持有该锁，因此没有stop和switch并发提交空档。统一顺序Supervisor锁 -> 短SQLite连接/事务，禁止任何持DB事务代码反向拿Supervisor锁。
重放先在短只读连接检查。新操作在无DB写事务时query/probe目标（真实推理探测），再BEGIN IMMEDIATE重新核对key、service revision、目标desired/observed和身份，在同一事务写service+revision+key+audit。期间Supervisor锁保持，网络调用绝不处于DB长事务中。外部管理员直接改DB不在支持边界；DB基本FK/CHECK/trigger仍防常见不一致。
当前Supervisor本来串行推理，阶段1保留：service predict持锁读绑定，然后在同锁内调用既有predict（RLock），核对真实结果active deployment/model/artifact/contract及协议身份后加service revision。切换必须等该调用结束；不承诺零停机或候选探测后绝不崩溃。
服务query持锁读取配置并真实query当前deployment；进程退出、stopped、损坏产物返回不可用或既有完整性错误，不悄悄读previous。配置GET可提供配置事实但不得把failed/unknown呈ready。

## 持久化与重启
新增model_services、model_service_revisions、model_service_keys及version marker model-services/v1，独立模块schema校验。现有所有表行/索引/trigger原样保留。历史快照包含完整配置身份与原操作timestamp，key关联revision且同tenant。
新空数据库直接初始化。已存在平台库无marker/部分对象/定义不符必须Store启动前fail-closed，且在cursor key或DDL写入前检查。迁移只接受完整running-training-cancel/v1前置库。schema验证包括表定义/必要索引/trigger/marker及FK/integrity；合法扩展用户索引/trigger按既有惯例保留。
显式离线python -m model_platform.migrate_services：先取得api、worker、model-import锁并证明serving已停止（仅存储PID liveness probe，绝不发送杀进程信号）；独占安全备份before-model-services.sqlite3，O_EXCL/NOFOLLOW held FD+SQLite backup；已存在备份不得覆盖。持BEGIN IMMEDIATE完成新增schema与marker，核对现有行/定义完全不变并integrity/FK；失败回滚DDL/marker，保留备份。已完整迁移为no-op，不产生重复备份；旧/部分状态不修复。
恢复API后service绑定与revision不变；依赖现有Supervisor按desired恢复相应deployment，重新实际probe前不得ready。停止被绑定deployment允许但该服务不可用；没有自动解绑、重启、回退或停止旧版。保留旧版继续计入既有4部署预算。
降级仅停全部进程后恢复迁移前备份，会丢失备份后全部写入；操作者必须禁止旧二进制打开新库（当前旧二进制允许额外schema，技术上不能强制阻止误用）；不提供无损降级承诺。生产迁移绝不执行。

## 验收
严格API、租户隔离、兼容性、并发CAS单胜者、旧key在多次切换后重放仍原snapshot且不重复审计、切换/stop/predict锁竞态、丢响应、提交前后崩溃、恢复ready前拒绝、损坏产物/候选崩溃不隐式回退、容量不变；migration所有前置/备份/partial/rollback/history场景。
真实临时自产dataset ONNX产物的A-B-A，固定endpoint实际identity随revision变化；重启后不变；停止当前503；原deployment API仍可用。CPU训练/ORT证据和全量stdlib+可选CPU回归独立报告；不把合成数据称质量证据。清理仅本次持有Popen身份的进程。不执行push/PR/merge。

## 独立审查补充
产物摘要与loaded identity在serving子进程加载时验证，后续probe验证内存实例及返回身份；加载后磁盘产物损坏不一定立即被发现，重新加载必须拒绝。阶段1不增加每请求重新哈希文件。service准入/查询/预测须比较持久配置及模型身份与live instance、当前代码生成的完整protocol，而非仅相信Store返回值。没有live instance时即使存储observed=ready也必须不可用。事务内核对须使用同一DB连接，不调用会再开连接的helper。重放仅验证保存snapshot与历史结构的一致性，不要求原部署此刻ready或产物存在。
