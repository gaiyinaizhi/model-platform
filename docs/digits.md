# 真实 UCI 手写数字 CPU 案例

本案例使用真实公开手写数字数据，完整走通独立 Worker 训练、重新加载产物后的留出评估、技术登记、持续推理子进程、重复 HTTP 调用和停止后拒绝推理。它是标准库 CPU 十分类示例，不代表通用视觉、机器人、无人机、端侧或生产环境验收。

## 来源、授权与固定数据

- 数据：UCI **Optical Recognition of Handwritten Digits**，E. Alpaydin 与 C. Kaynak，1998
- [官方数据页](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)、[DOI / 引用](https://doi.org/10.24432/C50P49)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 2026-10-12 已核查官方页面的 CC BY 4.0 标识；归属声明随运行数据写入 `source/ATTRIBUTION.txt`
- 下载只使用[官方 ZIP](https://archive.ics.uci.edu/static/public/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip)，不使用镜像、第三方重打包或 sklearn 的 1,797 行子集
- 完整保留 `optdigits.tra` 的 3,823 行与 `optdigits.tes` 的 1,797 行及其原始顺序；按照作者提供的分组，训练来自 30 位书写者，测试来自另外 13 位
- 每行 64 个整数特征组成 8×8 图像，取值 0..16；标签为 0..9。特征是原始 32×32 位图经过官方分块计数后的结果，不是原始图片文件

引用：Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49.

首次实际下载的固定 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| 官方 ZIP | `0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0` |
| optdigits.tra | `e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd` |
| optdigits.tes | `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8` |
| optdigits.names | `3e82f7202d72a2b7dbdbc324c8c90fe8853164f5d6ab978a071357ad3de89f02` |

脚本验证 ZIP 和固定文件摘要、行数、维度、整数范围、标签全集，校验失败直接停止。缓存同样重新校验，不悄悄接受新版文件。仅读取固定 ZIP 成员，不执行其中任何内容，也不按 ZIP 提供的路径解压。运行目录保存原始 ZIP、原始 CSV、原始 `optdigits.names`、归属说明、请求快照和元数据；这些文件不进入 Git。

元数据报告各类别数量、集合内重复特征/完整行，以及跨集合相同特征/完整行。首次实测所有重复与重叠计数均为 0。脚本不根据审计结果删除、移动、重抽样或重新分割任何记录。API 中 source/checksum/split 是调用方的血缘声明，并非 API 独立认证了远程来源；本脚本在提交前完成官方文件的固定摘要验证。

## 配方和评估协议

配方 `softmax-digits/v1`、模型契约 `digits-64x10/v1`：

- 纯 Python 标准库 softmax 多分类线性模型，64 输入、10 输出，零初始化
- 仅把像素除以 16；不从测试集学习预处理参数
- 固定 `epochs=40`、`learning_rate=0.03`、`seed=1729`，在首次测试集评估之前确定；学习率按 `0.03 / (1 + epoch_index * 0.05)` 衰减
- 固定种子的训练行洗牌与逐样本梯度更新；不搜索超参数，不根据本次测试结果调整配方
- 全部 3,823 行用于训练；全部 1,797 行只用于模型登记前的独立阶段评估
- 评估阶段校验并重新加载实际 JSON 模型，计算 accuracy、macro F1、log loss、10×10 confusion matrix 和样本数；这是同一平台内的留出评估，不是第三方审计
- 推理输出 0..9 的十个概率及最大概率类别；在线最多 128 行。输入必须仍是 64 个 0..16 整数，客户端不要预先除以 16

原始文件摘要和 API 规范化 JSON 摘要分别追踪，二者不是同一个摘要。模型清单包含配置、代码摘要、Python/平台、数据血缘、模型摘要和实际训练时间。

## 从仓库根目录复现

解释器沿用 `.python-version` 的 Python 3.12.14，零第三方依赖。首次下载需要访问 UCI；有完整有效缓存后可离线复用。使用隔离运行目录，三个终端须使用同一个 `MP_HOME`。

准备（已初始化时不要重复 init）：

```sh
export MP_HOME="$PWD/.runtime/digits-case"
.venv/bin/python -m model_platform init --tenant demo
.venv/bin/python scripts/fetch_digits.py
```

终端 A：

```sh
export MP_HOME="$PWD/.runtime/digits-case"
.venv/bin/python -m model_platform api --port 8080
```

终端 B：

```sh
export MP_HOME="$PWD/.runtime/digits-case"
.venv/bin/python -m model_platform worker
```

终端 C：

```sh
export MP_HOME="$PWD/.runtime/digits-case"
export MP_API="http://127.0.0.1:8080"
.venv/bin/python scripts/demo_digits.py
```

端口被占用时，API 可用 `--port 0` 选择空闲端口，随后将它打印的地址设置为终端 C 的 `MP_API`。演示仅允许 HTTP loopback 目标；凭据只从当前 `MP_HOME/credentials.json` 读取，不要求在命令行或聊天中粘贴令牌。

演示自动校验/获取缓存、提交完整数据、校验幂等提交、至多等待训练 180 秒、检查技术登记、等待实际 ready，并对测试集中每个标签第一次出现的样本连续调用 10 次（每批 10 行）。样本选择不依赖预测结果；完整准确率来自全部 1,797 行。演示校验十类概率、版本摘要、重复预测一致性，最后在 `finally` 内停用部署并确认之后的推理返回 HTTP 503。API / Worker 本身由各自终端 Ctrl+C 停止。

JSON 输出和 `MP_HOME/digits/<run_id>-report.json` 记录计算指标、样本预测、数据审计、运行环境、模型清单中的训练时长、提交到登记的观察时长，以及 10 次真实 HTTP 请求的原始/min/median/mean/max 时延。HTTP 时延包含鉴权、ready 探测、JSON 编解码与本机传输开销，不能描述为纯模型推理速度或性能基准。共享机器负载会影响结果；没有硬 CPU/RSS 隔离或 GPU。

`MP_HOME/digits/request.json` 含完整真实数据，不应上传或提交；报告、原始数据、模型权重、数据库、日志和凭据均留在 Git 忽略的 `.runtime/`。自定义运行目录应置于此目录或仓库之外；其他路径不会自动受 Git 忽略保护。

## 验证记录

2026-10-12：已实际从 UCI 下载官方文件，固定摘要、全部行数和审计通过；完整紧凑 JSON 请求为 896,482 字节，并通过 API 的输入校验。并完成以下两次真实端到端执行。


| 实测项 | 首次执行 | 重启后的相同配置执行 |
| --- | ---: | ---: |
| 训练 / 测试行数 | 3,823 / 1,797 | 3,823 / 1,797 |
| Accuracy | 0.9499165275459098 | 相同 |
| Macro F1 | 0.9501111995181309 | 相同 |
| Log loss | 0.15636690333762782 | 相同 |
| 训练时间（秒） | 9.909409214000334 | 10.919850311000118 |
| 10 行 HTTP 批请求 median（毫秒） | 5.885310500161722 | 6.380765999892901 |
| 重复 HTTP 推理 | 10 次，结果一致 | 10 次，结果一致 |
| 停止后 HTTP 状态 | 503 | 503 |

上述时间均为这台共享云端机器的一次观察，不是硬件性能承诺或严格基准。环境：Linux x86_64、Python 3.12.14、标准库、9 个可见逻辑 CPU、未使用 GPU。脚本在单训练 Worker 中执行，没有声称并行使用所有可见 CPU。

两次训练创建不同 run/model ID；逐项比对 accuracy、macro F1、log loss、整个混淆矩阵、样本数与示例预测完全一致，JSON 模型文件 SHA-256 也完全相同：

`7f33c2f690a82691ee4645ab6a5725587ed2b00d730f6371c099c1a6a8a7d1e8`

首次 run 为 `run_df26f029b04f474f88a67a06fc6117e6`；重复 run 为 `run_64f2b06fcb074afaa7a1c20b8fc602c3`。两次执行之间修复了二分类输入错误处理，未改变数字训练算法或超参数；重复执行的源码摘要为 `cf496a225f8d5ed7ecc198b58325f97ac005871227b7bca42da3c2ddaf2e91c6`。这些结果证明本环境下的实际重复执行一致，不承诺所有 Python/CPU 平台都按位相同。

原始报告位于忽略目录 `.runtime/digits-case/demo-report.json`、`.runtime/digits-case/demo-repeat-report.json`，以及各 run 的 `digits/<run_id>-report.json`。测试集每类首个样本的展示中有 2 个被误分类，照实保留，不替换成更好看的示例。完整测试集正确 1,707 / 1,797，不把小批展示准确率当作完整评估。

另外实际检查了：有效缓存不访问网络；完整集合行数；输出文件权限 0600；损坏摘要被拒绝；错误维度、越界像素、小数像素被拒绝；非官方域名或降级到 HTTP 的重定向被拒绝；脚本编译通过；运行数据与报告命中 Git 忽略规则。演示自己的 API、Worker 与 serving 子进程均已停止，没有留下持续运行服务。

### 完整官方测试混淆矩阵

行是真实标签、列是预测标签，顺序均为0..9；总计1797行，主对角线1707。

| 真值 \ 预测 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 176 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 1 | 0 | 169 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 7 |
| 2 | 0 | 3 | 172 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| 3 | 1 | 0 | 3 | 167 | 0 | 3 | 0 | 2 | 3 | 4 |
| 4 | 0 | 1 | 0 | 0 | 176 | 0 | 0 | 1 | 3 | 0 |
| 5 | 0 | 0 | 1 | 0 | 0 | 177 | 1 | 0 | 0 | 3 |
| 6 | 0 | 1 | 0 | 0 | 2 | 0 | 177 | 0 | 1 | 0 |
| 7 | 0 | 0 | 0 | 0 | 2 | 5 | 0 | 163 | 1 | 8 |
| 8 | 0 | 9 | 0 | 0 | 0 | 3 | 0 | 0 | 158 | 4 |
| 9 | 0 | 1 | 0 | 0 | 4 | 2 | 0 | 0 | 1 | 172 |
