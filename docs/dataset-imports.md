# 自定义 YOLO 数据集导入与不可变版本

2026-10-15 增量：仅导入、校验、登记和查询小型 YOLO 矩形框数据集。先取得合法使用权，选择负样本策略，上传 ZIP 查看逐文件报告，再明确发布不可变技术版本。此步骤不训练自定义类别，不改变已有 `yolox-nano-cat/v1` 配方、固定 Cat 数据源或其禁止生产状态。

## 运行与依赖

基础 API、二分类/UCI 仍为标准库；仅 ZIP 图像校验需要可选 Pillow 和 PyYAML。使用 Pillow 解码真实图像，避免自行实现图像解码；使用受约束的 PyYAML SafeLoader 解析数据，不执行构造器、下载或任意 YAML 指令。它们来自官方 PyPI，固定 Linux x86_64 / CPython 3.12 wheel 哈希见 `requirements-datasets.lock`；完整检测依赖锁也引用此文件。未安装时上传明确返回 `503 dataset_dependencies_missing`，不伪造成功；元数据、列表、旧训练 API 不要求这些库。

```sh
python3 -m venv .venv-datasets
.venv-datasets/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements-datasets.lock
.venv-datasets/bin/python -m pip check
.venv-datasets/bin/python -m model_platform init --tenant demo --tenant other
.venv-datasets/bin/python -m model_platform api --port 8080
```

使用私有、空白的 MP_HOME 或默认忽略的 `.runtime/`。`init` 不覆盖已有身份。API 仅监听 loopback，不需要 Worker 完成数据集校验；已有训练/服务仍使用原独立 Worker 和 supervisor。上述最小依赖环境不能运行 Cat 训练/ONNX 服务，不能当成完整检测环境。

## ZIP 结构与规则

```text
data.yaml
images/train/a.png
images/val/b.jpg
images/test/c.png        # 可选整个 test split
labels/train/a.txt
labels/val/b.txt
labels/test/c.txt
```

允许整个结构放入一个外层目录，允许 split 内子目录，但不接受多个数据集、图片清单文件、自定义 split 路径、嵌套 ZIP 或任意其他文件。图片必须为单帧 JPEG/PNG，扩展名与真实格式一致；相同 stem 的 jpg/png 不能同时存在。

```yaml
path: .                 # 可省略；只能是 . 或空字符串
train: images/train
val: images/val
test: images/test        # 若没有 test 数据，省略本行
nc: 2                   # 可省略；存在时必须等于 names 数量
names: [cat, dog]        # 也支持连续的整数键 {0: cat, 1: dog}
```

- train 和 val 必须声明且各有至少一张图片，test 可选但声明后也必须非空
- names 是 1..100 个互异非空字符串，每项最多 80 字符；未知 YAML 字段、重复键、标签、锚点、别名、过深嵌套、外部路径/URL/download 指令均拒绝
- 每个图片必须有同相对 stem 的 UTF-8 标签文件；孤立标签报错
- 每个非空标注行为 `class_id x_center y_center width height`，只支持矩形框，不支持多边形/分割/关键点
- class_id 必须为 names 范围内的整数；坐标有限、归一化，宽高正数，框整体在图内（边界浮点容差 1e-9）
- `negative_policy=reject` 拒绝空标签；`allow_empty` 允许空/纯空白标签明确表示没有目标。缺失标签始终报错，绝不默认为负样本
- 图片原字节 SHA-256 和解码后 RGB 像素摘要均用于跨 train/val/test 泄漏检测，换 ZIP 压缩或修改图片元数据不能隐藏相同像素。相同 split 内重复保留；不是近似图像查重或感知相似度检测
- 解码尺寸、RGB 摘要和框坐标以图片原始像素矩阵为准，不应用 EXIF 旋转，也不重映射框。浏览器可能按 EXIF 方向显示不同外观；本步骤不声称已规范化到浏览器方向，后续训练接入需显式处理此契约

## 明确预算

- ZIP 请求 1 byte..32 MiB；必须有单一十进制 Content-Length 和 application/zip，无 Transfer-Encoding/Content-Encoding
- ZIP 展开总量最多 128 MiB、2048 entries；单文件 8 MiB；压缩比最多 200:1，实际读取也检查边界与 CRC
- 只支持 ZIP STORED/DEFLATED；拒绝加密、符号链接/设备/FIFO、重复/大小写冲突路径、文件与父目录冲突、绝对/Windows/反斜线/控制字符/`.`/`..` 路径；完整路径最多 240 UTF-8 bytes
- 最多 1000 图片，每图 20M 像素，整包累计 200M 像素；逐图解码，不保留全量解码像素
- data.yaml 最多 64 KiB/10000 tokens/10 层；每个标签 1 MiB/10000 行；整包最多 100000 框
- 报告最多 200 个错误，超出标记 errors_truncated；错误包含相对路径、代码、可选行号，不返回主机目录或凭据
- 每 runtime 仅一个验证/发布操作；冲突返回 429 dataset_busy；仍保留原 API 16 连接、120 请求/tenant/分钟预算
- 持久化数据目录最多 512 MiB，包括 staging、version、临时/孤儿文件；发布复制需额外空间。每 runtime 最多 256 个 import 记录、100 个 version。没有自动清理、删除或生命周期 API；达到上限后由可信本地操作者审查/备份后另建私有 runtime，不能直接修改已登记版本

这些是逻辑预算，不是 OS 级 CPU/RSS/cgroup 或生产多租户沙箱。

## 鉴权 API

所有路由使用既有 Bearer token，tenant 从 token 决定，不能在请求中指定。其他 tenant 读/写对象返回 404。IDs 分别为 `dsi_`/`dsv_` 加 32 位小写十六进制；created_at 为 Unix 秒浮点数。

1. `POST /dataset-imports`，Content-Type application/json，必须单个 Idempotency-Key（1..128字符）

```json
{"name":"Factory parts","rights_confirmed":true,"negative_policy":"reject"}
```

name 为 1..100 字符且无首尾空白。首次 201；同 tenant/key/规范化元数据返回同 import，200；同 key 不同元数据 409 idempotency_conflict。不同 tenant 的相同 key 独立。导入创建不上传本机路径。

2. `POST /dataset-imports/{id}/archive`，原始 ZIP body（不使用 multipart/base64）

同步校验后 200 返回 import，包括 status=valid 或 invalid、逐文件 report。格式/语义无效 ZIP 是可显示的 invalid 报告；鉴权、传输或存储失败为 `{error:{code,message}}`。同原始 ZIP SHA-256 重试返回原报告；任何不同字节均 409 dataset_upload_conflict，需创建新 import。响应丢失时 GET 对账，同包重试安全；报告不可覆盖。进行中 GET 可仍显示 awaiting_upload，下一次写操作可能返回 dataset_busy，应稍后重试。

3. `GET /dataset-imports/{id}`

```json
{
  "id":"dsi_0123456789abcdef0123456789abcdef",
  "name":"Factory parts","status":"valid","negative_policy":"reject",
  "created_at":1792020000.0,"version_id":null,
  "report":{
    "valid":true,"errors":[],"warnings":[],"errors_truncated":false,
    "counts":{"images":2,"labels":2,"boxes":2,"negative_images":0},
    "classes":[{"id":0,"name":"part","boxes":2,"images":2}],
    "splits":{"train":{"images":1,"boxes":1,"negative_images":0},"val":{"images":1,"boxes":1,"negative_images":0},"test":{"images":0,"boxes":0,"negative_images":0}},
    "content_sha256":"64 lowercase hex characters"
  }
}
```

status 为 awaiting_upload/valid/invalid/published；上传前 report=null；无效报告 content_sha256=null。counts/classes/splits 是本次扫描结果，无效包中可能为部分统计，应与错误一起显示。error 为 `{code,path,message,line?}`，line 从 1 起。warnings 当前保留为空数组。发布后 version_id 指向登记版本。

4. `POST /dataset-imports/{id}/publish`，JSON `{}`

仅 valid 可发布，否则 409 dataset_not_valid。重新核验 staging 的完整文件集合与哈希，复制到独立版本目录，再次核验，原子 rename 后事务登记。首次 201；重复发布 200。同 tenant、name、content_sha256 重复导入复用同版本（200），不同 tenant 永不复用技术 ID。修改名称或规范化内容产生新版本；同 ZIP 的打包时间/压缩参数/单层外目录不参与内容身份，但 data.yaml/标签原字节属于内容，所以 YAML 格式修改也会形成新摘要。

5. `GET /dataset-versions` -> `{versions:[VersionSummary],limit:100}`，按时间倒序；每个 tenant 仅见自己的记录。全局版本上限 100，所以无隐式未列出页；此路由不支持 query 参数
6. `GET /dataset-versions/{id}` -> 完整 Version，包括 manifest；读取时核验实际文件，损坏/缺失返回 409 dataset_integrity。列表只读元数据，不代表刚完成逐字节完整性校验

VersionSummary 字段：id、name、created_at、content_sha256、source_import_id、negative_policy、counts、classes、splits。完整 Version 额外返回：

```json
{
  "manifest": {
    "format":"yolo-detection/v1","negative_policy":"reject",
    "counts":{},"classes":[],"splits":{},
    "files":[{"path":"images/train/a.png","bytes":123,"sha256":"64 hex"}],
    "samples":[{"image":"images/train/a.png","label":"labels/train/a.txt","split":"train","width":640,"height":480,"boxes":1,"image_sha256":"64 hex","label_sha256":"64 hex"}]
  }
}
```

counts/classes/splits 与摘要字段相同。content_sha256 是 manifest 规范化 JSON 的 SHA-256；manifest 包含每个图片、标签和 data.yaml 原字节摘要。响应只有元数据/相对路径，无原始图片下载、权重或绝对路径。最大 manifest 可超过旧 BFF 的 1 MiB 限制，业务网关需为这些受限数据集路由设置独立响应预算，不能放宽旧训练/推理预算。

## 错误与恢复

- 400 invalid_input/invalid_archive_body：协议错误或上传未完成；尚未保存报告时仍可使用同 import 重传
- 401 unauthorized、404 not_found：沿用既有鉴权边界
- 409 idempotency_conflict/dataset_upload_conflict/dataset_not_valid/dataset_integrity：不可重解释为成功
- 413 body_limit：ZIP 长度不在预算内，读取前拒绝
- 429 dataset_busy/dataset_capacity/rate_limit：稍后重试或由本地操作者处理容量
- 503 dataset_dependencies_missing：安装可选数据集依赖后重试
- 无效数据集返回 200 + status=invalid，错误代码如 unsafe_path、duplicate_path、invalid_zip、data_yaml、invalid_image、invalid_label、invalid_box、class_id、missing_label、negative_not_allowed、split_overlap；必须显示报告而不是演示数据

SQLite 触发器阻止修改/删除已发布版本和覆盖已保存校验报告。磁盘快照是复制文件，不是链接，原上传 ZIP、staging 的旧打开句柄或后续数据修改不能改变登记字节。读取/发布核验文件哈希。可信本地管理员仍可直接损坏磁盘或数据库；这是逻辑不可变，不是 WORM/对管理员安全的存储。

进程在 rename 与 DB commit 间终止可能留下未登记孤儿目录，不会变成版本、不会被训练或部署；孤儿计入容量且不自动删除。复制/rename/DB 错误在正常异常路径清理未登记目标并保留可重试 staging。DB 提交后清理异常可能导致响应错误，但 GET/重复 publish 可取得已登记版本。没有断电耐久性、自动备份/恢复或生产权限验收。

## 验证范围

```sh
.venv-datasets/bin/python -m compileall -q model_platform scripts tests
.venv-datasets/bin/python -m unittest discover -s tests -p test_datasets.py -v
.venv-datasets/bin/python -m unittest discover -s tests -v
```

测试使用自生成无敏感 PNG/ZIP、真实 loopback HTTP、SQLite 和临时文件，覆盖合法样本/分类统计、错误路径/图像/标注/YAML/ZIP、CRC/DEFLATE 异常、跨 split 原字节/像素泄漏、负样本、预算、权限/重放、复制隔离、DB 不可变、并发发布、存储/事务失败及旧 Cat 配方目录保持不变。不代表真实业务数据质量、实际自定义训练或设备验收。最终实测汇总见 dataset-verification.json；未安装 ML 依赖的可选测试会跳过，不能称为全 ML 回归通过。
