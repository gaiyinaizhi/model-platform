# CPU 持续集成

工作流：[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)。每次 `push`、`pull_request` 都运行，也可通过 `workflow_dispatch` 手动重跑同一组检查。两项 job 相互独立，均使用 GitHub 托管 `ubuntu-24.04` x86_64、[`.python-version`](../.python-version) 中的 CPython **3.12.14**。

## 两层检查

- **Standard library / Python 3.12.14**：创建不继承系统包的 `.venv`；确认没有 torch、NumPy、ONNX、ONNX Runtime；执行全部源码的 `compileall` 和 `unittest discover -s tests -v`。不安装第三方依赖。所有基础回归都会发现并执行，只有明确依赖可选库或官方数据缓存的测试按测试自身条件跳过。超时 10 分钟。
- **Optional ML / CPU ONNX**：独立 `.venv-fashion` 从已有 [`requirements-fashion.lock`](../requirements-fashion.lock) 安装完整 CPU 栈，强制 wheel 和 SHA-256 校验，运行 `pip check`，实际导入各库并核实 CPU-only PyTorch 和 ONNX Runtime CPU provider。随后执行 `test_fashion.py` 中的全部轻量测试；空测试集、任何 skip、失败或错误均让 job 失败，防止依赖缺失被误报通过。超时 20 分钟。

“可选 ML”指模型能力与基础运行环境独立，并非允许 CI 失败：该 job 每次触发均运行，没有 `continue-on-error`。依赖下载失败会准确标为该 job 失败，不影响标准库 job 独立执行。

轻量 ONNX 回归使用明确标注的合成图，覆盖真实 CPU 解析/预测、预处理、非法图/外部权重拒绝、HTTP 推理进程、版本绑定、停止、损坏/崩溃及重启恢复；也覆盖训练辅助函数，但不训练 CNN。它们不能证明真实数据上的识别质量。

## 网络、数据与权限边界

CI 的 checkout、Python 安装和可选依赖安装需要网络；**测试本身不下载数据**。默认不运行 `fetch_digits.py`、`fetch_fashion.py` 或正式数据训练 demo，也不上传 `.runtime`、数据、权重、凭据或日志作为 artifact。官方完整数据训练继续按 [digits.md](digits.md)、[fashion.md](fashion.md) 手动验收，避免每个 PR 依赖外部数据源并反复执行训练。本次不增加独立的正式数据 smoke 工作流。

仅声明 `contents: read`；checkout 不持久化凭据，不引用 secrets，不使用 `pull_request_target`，也无写回、发布或部署步骤。PR 运行不得改为带 secrets 的高权限事件。并发组按工作流、事件及 PR/分支区分，新提交取消同组旧运行。job 超时只限制 CI 占用时间，不是应用的硬 CPU/RSS 隔离。

仅使用以下官方 Actions，固定完整提交 SHA（已从官方 release 的 commit 链接核对）：

- [`actions/checkout` v6.0.2](https://github.com/actions/checkout/releases/tag/v6.0.2)：[`de0fac2e4500dabe0009e67214ff5f5447ce83dd`](https://github.com/actions/checkout/commit/de0fac2e4500dabe0009e67214ff5f5447ce83dd)
- [`actions/setup-python` v6.2.0](https://github.com/actions/setup-python/releases/tag/v6.2.0)：[`a309ff8b426b58ec0e2a45f0f869d46889d02405`](https://github.com/actions/setup-python/commit/a309ff8b426b58ec0e2a45f0f869d46889d02405)

升级时从官方仓库重新核对 release → commit，更新 SHA 与版本注释，再执行本地检查；不要只替换成可变 major tag。这里固定解释器、Action 代码和可选 Python wheels；托管 runner 镜像仍会更新，不宣称完整操作系统可复现。

已核对 [`actions/python-versions` 官方清单](https://github.com/actions/python-versions/blob/main/versions-manifest.json) 中存在 Python 3.12.14 的 Linux 24.04 x64 安装包；安装仍取决于执行时 GitHub 下载服务可用。

## 本地复现

从仓库根目录、Linux x86_64 / Python 3.12.14 执行。若已有同名 venv，确认其未继承系统包且符合对应依赖要求；不要让已安装的可选依赖掩盖基础环境回归。

```sh
python -m venv .venv
.venv/bin/python -m compileall -q model_platform scripts tests
.venv/bin/python -m unittest discover -s tests -v

python -m venv .venv-fashion
.venv-fashion/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements-fashion.lock
.venv-fashion/bin/python -m pip check
.venv-fashion/bin/python -m unittest discover -s tests -p test_fashion.py -v
```

最后一条用于方便本地查看结果；**必须确认没有 skip**。CI 另外实际导入所有可选库、核实 CPU 后端，并通过工作流中的 Python runner 强制无 skip。全部 CI shell/Python 步骤以工作流文件为准。

## 验证状态

2026-10-12，Linux x86_64 / CPython 3.12.14：将当时的源码/测试快照复制到隔离临时目录，不复制官方数据缓存，从工作流提取并实际执行全部 `run` 步骤，两个 venv 均为新建。

- 标准库环境版本/无可选库检查及 `compileall` 通过；发现 75 项测试，64 项通过、11 项预期跳过（10 项 ONNX + 1 项官方缓存），无失败
- 现有 lock 的全部 21 个直接/传递依赖新下载并完成 hash-locked 安装；`pip check` 通过；NumPy、ONNX、ONNX Script、ONNX Runtime、PyTorch 实际导入与 CPU 检查通过
- 轻量 Fashion/ONNX 17/17 通过、零 skip；没有下载官方数据或训练 CNN
- 将可选 job 的预检查、测试步骤改用无依赖 venv 做负向实验，两者分别因缺少 NumPy、10 项 ONNX skip 返回退出码 1，证实不会出现依赖缺失的虚假绿灯
- YAML 解析、全部 `run` 脚本的 `bash -n`、触发条件/权限/完整 SHA/超时/并发/独立 job 结构检查通过

以上计数对应 CI 配置验证时的源码快照；后续新增测试的完整回归结果见 [verification.md](verification.md)。未运行 GitHub 托管 Actions，也未推送、修改远端权限或分支保护。官方 Action 本身与 GitHub 调度无法由本地 shell 验证，远端 CI 结果必须以授权提交后对应 Actions run 为准。
