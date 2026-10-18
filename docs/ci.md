# CPU 持续集成

工作流：[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)。每次 `push`、`pull_request` 都运行，并保留已有的 `workflow_dispatch` 手动入口；本轮仅使用自动 push/PR 触发，不手动调用工作流，也不增加独立真实数据 smoke 工作流。两项 job 相互独立，均使用 GitHub 托管 `ubuntu-24.04` x86_64、[`.python-version`](../.python-version) 中的 CPython **3.12.14**。

## 两层检查

- **Standard library / Python 3.12.14**：创建不继承系统包的 `.venv`；确认没有 torch、NumPy、ONNX、ONNX Runtime 或 OpenCV；执行全部源码的 `compileall` 和 `unittest discover -s tests -v`。不安装第三方依赖。基础回归全部发现并执行，依赖可选库、官方数据缓存或私有资产的测试按自身条件跳过。超时 10 分钟。
- **Optional ML / CPU ONNX**：独立 `.venv-fashion` 同时安装 [`requirements-detection.lock`](../requirements-detection.lock) 和 [`requirements-pretrained.lock`](../requirements-pretrained.lock)，前者包含 Fashion、数据集及检测依赖，后者补齐固定 OpenCV 预处理依赖。安装强制 wheel 和 SHA-256 校验，运行 `pip check`，实际导入 NumPy、ONNX、ONNX Runtime、ONNX Script、CPU PyTorch、Pillow、pycocotools 和 cv2，并核实 CPU-only PyTorch 与 ONNX Runtime CPU provider。随后发现整个 `tests/`，仅排除下述六项精确命名的集成测试，执行所有其余测试；空测试集、任何 skip、失败或错误均让 job 失败。超时 20 分钟。

“可选 ML”指模型能力与基础运行环境独立，并非允许 CI 失败：该 job 每次触发均运行，没有 `continue-on-error`。依赖下载失败会准确标为该 job 失败，不影响标准库 job 独立执行。

## 精确集成排除与实际覆盖

可选 job 仅排除以下 ID，并在运行前要求**每个 ID 恰好发现一次**。测试被改名、移除或重复发现时，选择步骤会失败；不会按模块、类名、skip 原因或可选依赖缺失批量放过测试。

1. `test_fashion_data.OptionalOfficialCacheTests.test_full_source_is_reverified_offline`：需要完整官方 Fashion 缓存
2. `test_pretrained_yolox.ActualAssetTests.test_validation_load_predict_tamper`：需要授权私有预训练资产
3. `test_pretrained_yolox.UpstreamParityTests.test_exact_upstream_helpers`：需要同一私有资产及固定摘要的上游参考源码
4. `test_import_serving.ImportedServingTests.test_lifecycle_restart_stop_replay`：需要同一私有资产
5. `test_import_serving.ImportedServingTests.test_ownership_no_download_or_fake_evaluation`：需要同一私有资产
6. `test_import_serving.ImportedServingTests.test_corruption_never_ready`：需要同一私有资产

这些测试保留在完整本地验收中；CI 的排除不代表它们已通过。真实预训练资产、上游协议一致性及导入后持续服务的独立验收见 [pretrained-import.md](pretrained-import.md)。

常规回归仍覆盖：标准库训练/存储/API/生命周期；使用合成 ONNX 图的真实 CPU 解析、预测、非法图/外部权重拒绝、HTTP 推理进程、版本绑定、停止、损坏/崩溃及重启恢复；无真实训练的 YOLOX 小图导出/加载、合成几何和标准 AP 已知答案；数据集解析、隔离、缓存拒绝和安全边界。预训练增量包含无需权重的输入/标签/hash/超时边界、OpenCV 非方形预处理、解码、NMS、裁剪和数量上限，以及使用明确标注 mock 验证器/不透明 fixture 的导入登记和迁移回归。以上不能证明真实模型识别质量，mock 验证也不等同于实际资产验证。

## 网络、数据与权限边界

CI 的 checkout、Python 安装和可选依赖安装需要网络；**测试本身不下载数据、上游参考源码或预训练权重**。不设置 `MP_PRETRAINED_YOLOX_ASSET`、`MP_YOLOX_REFERENCE_DIR` 或 `MP_YOLOX_PHOTO_DIR`，不依赖开发机缓存。默认不运行下载脚本或正式数据训练 demo，也不上传 `.runtime`、数据、权重、凭据或日志作为 artifact。测试可以生成并清理临时合成图和数值模型。官方完整数据训练继续按 [digits.md](digits.md)、[fashion.md](fashion.md) 和 [detection.md](detection.md) 单独验收，避免每个 PR 依赖外部数据源或重复训练真实模型。

仅声明 `contents: read`；checkout 不持久化凭据，不引用 secrets，不使用 `pull_request_target`，也无写回、发布或部署步骤。PR 运行不得改为带 secrets 的高权限事件。并发组按工作流、事件及 PR/分支区分，新提交取消同组旧运行。job 超时只限制 CI 占用时间，不是应用的硬 CPU/RSS 隔离。

仅使用以下官方 Actions，固定完整提交 SHA（原 CI 建立时已从官方 release 的 commit 链接核对）：

- [`actions/checkout` v6.0.2](https://github.com/actions/checkout/releases/tag/v6.0.2)：[`de0fac2e4500dabe0009e67214ff5f5447ce83dd`](https://github.com/actions/checkout/commit/de0fac2e4500dabe0009e67214ff5f5447ce83dd)
- [`actions/setup-python` v6.2.0](https://github.com/actions/setup-python/releases/tag/v6.2.0)：[`a309ff8b426b58ec0e2a45f0f869d46889d02405`](https://github.com/actions/setup-python/commit/a309ff8b426b58ec0e2a45f0f869d46889d02405)

升级时从官方仓库重新核对 release → commit，更新 SHA 与版本注释，再执行本地检查；不要只替换成可变 major tag。这里固定解释器、Action 代码和可选 Python wheels；托管 runner 镜像仍会更新，不宣称完整操作系统可复现。

原 CI 建立时已核对 [`actions/python-versions` 官方清单](https://github.com/actions/python-versions/blob/main/versions-manifest.json) 中存在 Python 3.12.14 的 Linux 24.04 x64 安装包；安装仍取决于执行时 GitHub 下载服务可用。

## 本地复现

从仓库根目录、Linux x86_64 / Python 3.12.14 执行。若已有同名 venv，确认其未继承系统包且符合对应依赖要求；不要让已安装的可选依赖掩盖基础环境回归。

```sh
python -m venv .venv
.venv/bin/python -m compileall -q model_platform scripts tests
.venv/bin/python -m unittest discover -s tests -v

python -m venv .venv-fashion
.venv-fashion/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements-detection.lock -r requirements-pretrained.lock
.venv-fashion/bin/python -m pip check
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv-fashion/bin/python - <<'PY'
from pathlib import Path
import subprocess
import yaml
workflow = yaml.safe_load(Path('.github/workflows/ci.yml').read_text())
wanted = {
    'Verify optional dependencies and CPU runtime',
    'Run offline optional regressions with exact integration exclusions',
}
steps = [step for step in workflow['jobs']['optional-ml']['steps'] if step['name'] in wanted]
assert len(steps) == len(wanted), 'Expected both workflow verification steps'
for step in steps:
    subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', step['run']], check=True)
PY
```

PyYAML 已由检测栈包含的数据集 lock 安装。以上从工作流读取并执行相同的依赖预检查与严格测试选择，不另行维护一份排除名单；其余 CI shell/Python 步骤以工作流文件为准。完整本地实际资产验收应按专项文档显式提供授权资产和参考源码，不把它们复制到 CI。

## 验证记录

历史记录：2026-10-12 的初始 CI 快照实跑两套新建 venv，基础测试 75 项（64 通过、11 项可选跳过），Fashion/ONNX 17/17 通过、零 skip；完整 hash-locked 安装和缺依赖负向实验通过。后续检测增量在不含数据/模型/venv 的干净源码副本中实跑相同选择：176 项通过、零 skip；当时的标准库全套为 177 项（134 通过、43 项可选跳过）。这些计数对应历史源码，不是当前覆盖总数。

2026-10-18 预训练导入 CI 增量：从当前源码复制干净临时副本，不复制 `.runtime`、模型、图片或 venv，清除私有资产/上游参考/照片的环境变量；使用已有、不继承系统包的同级 Python 3.12.14 venv，提取并执行工作流中**完全相同**的可选依赖预检查与严格测试选择。

- 发现 **297** 项；六个排除 ID 各发现一次；其余 **291/291 通过、零 skip、零失败/错误，124.994 秒**
- `pip check`、全部 25 个直接/传递依赖版本与 lock 的一致性、重复依赖的版本/hash 一致性通过；实际导入及 CPU 检查通过，cv2 为 4.13.0（发行包 `opencv-python-headless==4.13.0.92`）
- 缺失/重复每个排除 ID 的 12 项负向选择实验，以及 skip、失败和空测试集的负向实验均按预期拒绝；正常选择对照通过
- YAML 解析、全部 `run` 的 `bash -n`、保留的触发条件和只读权限检查、本文相对链接检查通过；运行前后均无官方数据缓存目录

本次复用已有可选环境，没有重新下载 wheel 或重建 hash-locked 安装，也没有重跑完整标准库 job；没有下载数据/权重、运行实际资产集成或手动触发远端工作流。该记录是本地验证，GitHub 托管 Actions 的调度、安装和远端结果必须以授权提交后的对应自动 Actions run 为准；本地通过不等同于远端已通过。
