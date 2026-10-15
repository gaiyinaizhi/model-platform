# Open Images Cat：固定工程检测小切片

该数据切片用于本地检验 YOLOX 检测训练、真实 bbox AP、ONNX 和推理服务契约。它不是 Open Images 官方挑战结果，也不是业务质量或商业使用许可保证。完整训练/推理契约见 [detection.md](detection.md)。

## 已固定的数据

- Open Images 官方 `train` / `validation` / `test` 分别取 128 / 32 / 32 张图；没有把同一集合复制成测试集
- train：96 张 Cat 正样本、32 张经过人工确认的 Cat 负样本，共 110 个 Cat 框
- validation、test：各 24 正、8 负、26 个 Cat 框
- 叶类 `Cat`，官方 MID `/m/01yrx`；模型 `class_id=0`，COCO `category_id=1`
- 每图保存原始 image-level 人工验证记录、所有 Cat bbox 原记录、图片尺寸、官方原来源页面、作者、标题和逐图许可字段
- JPG 原字节 SHA-256、完整 RGB 像素 SHA-256、来源 URL 审计没有跨集合重合；64-bit dHash 跨集合汉明距离 ≤4 的候选为 0。启发式近重复筛查不保证场景、摄影者或主体完全独立

最终规范化 manifest SHA-256：

`06a9b65a1e1c9826b2e4c4ae0284d28b6c2ab89b2dfc63d83ebca796dc2e3387`

manifest 使用 UTF-8、`ensure_ascii=False, sort_keys=True, indent=2`，末尾一个换行。像素摘要精确定义为 SHA-256(`f"{width},{height}:".encode()` + `image.convert("RGB").tobytes()`)；不是只哈希缩略图或 JPG。路径相对数据集根目录，不含本机路径和生成时间。

## 为何可以使用 COCO-style bbox AP

[Open Images 官方说明](https://storage.googleapis.com/openimages/web/factsfigures.html) 区分已确认存在、已确认不存在以及未知的图级类别。对于正标签，官方对该类别实例提供穷尽框，group-of 情形有例外。Cat 在[官方层级](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json)中无子类别，因此该切片无需父子类扩展。

选择时：

1. 只接受 Cat 的 `verification` / `crowdsource-verification` 且 `Confidence=1` 或 `0`；未知类别状态的图整张排除，不当作负例
2. 正样本必须有 Cat 框；负样本必须没有 Cat 框
3. 任意 Cat 框 `IsGroupOf!=0` 或 `IsDepiction!=0`，整张图排除。不能简单删框后把对应目标变成假负例，也没有把 Open Images group-of 错当作 COCO crowd
4. 保留所选图中所有 Cat 框，包括 occluded、truncated、inside 标志；不按大小静默移除目标
5. 仅取元数据明确 `Rotation=0` 的图，避免原始来源旋转与标注坐标不一致
6. 对每个官方 split，按 image-level CSV 顺序先收集 `3*N` 正、`N` 负候选，再读取这些图全部 Cat 框及完整图片元数据；通过以上过滤后按 ImageID 排序，分别取固定正负配额。该规则是确定性工程选样，不代表自然类别分布

于是对于这个唯一目标类别，每个评估图的存在/不存在状态均明确，且不留未解决的 group-of：可转换成普通 COCO bbox GT，用 pycocotools 计算 AP50 和 AP50:95。名称必须为“固定完整 Cat 工程小切片的 COCO-style AP”；[Open Images 官方挑战评估](https://storage.googleapis.com/openimages/web/evaluation.html)采用不同的缺失标签、层级和 group-of 规则，不能把两者指标混称。小测试集的结果方差大，低分或零分也应如实记录。

## 来源与许可边界

- [官方下载页与格式说明](https://storage.googleapis.com/openimages/web/download_v7.html)
- 图像下载端点来自[官方 downloader.py](https://raw.githubusercontent.com/openimages/dataset/master/downloader.py)声明的 `open-images-dataset` bucket，键为 `<official_split>/<ImageID>.jpg`；实现只使用该官方 S3 HTTPS 主机
- [官方 class mapping](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv)
- 精确 bbox、图级标签和逐图 metadata HTTPS URL，以及 GCS generation、ETag、响应长度，都在脚本 `MANIFEST_BASE.sources/source_headers` 中固定，并写入运行时 manifest

Google LLC 将标注以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布。所选图片逐图官方 metadata 均记载 [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)，作者和原始落地页随 manifest 保存。但 [Open Images 许可说明](https://storage.googleapis.com/openimages/web/factsfigures.html#licenses)明确不保证图片许可状态。

本次只核对官方逐图许可元数据，没有逐一实时核实原 Flickr 页权利，没有声称取得额外商业授权。图片和原始标注仅保存在被忽略的本地运行目录，不进仓库、不上传、不公开发布。若之后需要再分发图片、展示至公共产品或商业使用，应另行核实逐图权利及适用署名要求。

## 运行与完整性

从仓库根目录运行；使用已安装检测可选依赖的环境，脚本自身不安装包：

```sh
python scripts/fetch_detection.py
python scripts/fetch_detection.py --verify-only
```

默认目录为 `$MP_HOME/datasets/openimages-cat-reviewed-v1`；未设置 MP_HOME 时为 `.runtime/datasets/openimages-cat-reviewed-v1`。首次下载约 192 张图（实际 JPG 约 55 MB，精确大小由下载结果决定）。原 metadata 是大 CSV：图片元数据会完整流式扫描，其余按候选范围读取，网络元数据流量明显大于图片。它们不会整份写进仓库或运行目录。

指定已有受控缓存：

```sh
python scripts/fetch_detection.py --output .runtime/detection-data --verify-only
python scripts/fetch_detection.py --output .runtime/detection-data --rebuild
```

- `--verify-only` 不联网，检查固定 manifest SHA、每张 JPG、完整 RGB 像素、尺寸、类别/框、正负配额、来源及近重复筛查
- `--rebuild` 重新读取固定 GCS generation 的官方元数据，验证 ETag/长度并重新生成选择；原 JPG 可重用，但最终 manifest 必须恰好匹配固定 SHA。不因下载失败替换图片或悄悄换数据
- 已有缓存篡改会报错，不自动“修好”后继续训练；上层 API 不能通过请求控制外部来源 URL
- 所有下载仅允许官方 HTTPS 主机；重定向仍检查 allowlist，禁止凭据 URL、HTTP 和非标准端口
- 每张下载最多 10 MiB；安全验证 JPEG 后解码，最多 16 Mi 像素；CSV 每行最多 2 MiB，每源实际读取最多 800 MiB，单源读取预算 180 秒、网络操作 timeout 60 秒；最多 3 个元数据任务和 8 个图片下载
- 限时/字节预算是客户端读取预算，不是操作系统级沙箱或整体进程资源强隔离；可信本地管理员仍能修改程序和存储

摘要须准确区分：已下载 JPG、完整解码像素、class mapping、hierarchy 和最终 manifest 都有本地实际 SHA-256。大 CSV 的身份锁是官方 GCS generation 与服务端 ETag/响应头；图级标签和 bbox 只读取所需前缀，没有声称计算过其完整 CSV 的本地 SHA-256。最终 manifest SHA-256 锁住实际选用记录及来源属性。重新生成不匹配时必须停止并调查，不能更新固定摘要来迁就变化。

## 数据准备验证

2026-10-12：实际下载并安全解码 192 张 JPG；全部逐图摘要、正负样本、所有框边界、三 split 配额和近重复审计通过。准备工具的实际故障测试覆盖：禁止主机/HTTP/含凭据 URL/非标准端口、目录穿越、非法图像、超像素图像、manifest 篡改、JPG 篡改、图片符号链接逃逸、超大缓存；全缓存校验在禁止网络函数被调用的条件下通过。重新读取固定官方元数据的再现验证状态以本次实施验收报告为准。数据准备通过不等于模型训练或生产验收通过。
