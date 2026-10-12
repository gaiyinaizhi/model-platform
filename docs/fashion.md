# Fashion-MNIST 小型 CNN / ONNX CPU 切片

## 本阶段契约

配方 `fashion-cnn/v1` 独立于原标准库配方。`POST /runs` 仅接受固定公开数据引用：

```json
{"recipe":"fashion-cnn/v1","dataset":{"source":"Fashion-MNIST","version":"official-v1","rights_confirmed":true,"reference":"fashion-mnist-official/v1"},"config":{"epochs":5,"learning_rate":0.001,"seed":20261012,"train_count":6000,"batch_size":128}}
```

无任意文件路径、URL、图像训练内联或外部模型导入。安装/下载由可信本地操作者完成；Worker 离线重新验证官方缓存，不能只信下载脚本。固定种子从官方60000训练样本中每类抽600，共6000训练；官方10000 test仅最终评估，不参与选择超参数或早停。不能把此示例称为60000全量训练。公开数据引用不赋予读取任意租户私有文件的权限。

模型独立实现两层小CNN（1→8→16通道，3×3卷积、ReLU、2×2池化、784→10线性层），Adam从零训练。输入API为 `features`，每次1..32张，各784个行优先0..255整数；内部固定转float32 `[N,1,28,28]` 并除255，无test拟合统计。输出十项概率及0..9类别。类别顺序为T-shirt/top、Trouser、Pullover、Dress、Coat、Sandal、Shirt、Sneaker、Bag、Ankle boot。

登记 `fashion-onnx/v1` / `fashion-28x28x10/v1`，存单文件ONNX而不是pickle。仅从不可变登记的artifact_id读取文件，先核对SHA256再解析，拒绝外部张量和非白名单算子；无任意自定义算子注册。内部现有artifact_id文件名沿用`.json`后缀，但format明确表示内容为ONNX二进制，不能按扩展名推断格式。CPU ONNX Runtime是该配方唯一服务引擎。独立训练/服务进程、ready实际推理、停止和API重启恢复复用现有生命周期。

逻辑预算：训练600秒、导出/评估各120秒（阶段间/批间检查，不是OS硬抢占）、2MiB单产物、CPU两线程、单Worker、最多4部署。ONNX/PyTorch属本地可信原生运行库；没有cgroup或硬内存/CPU沙箱，不对公网部署。

## 实施与验证状态

2026-10-12真实闭环已通过，详细可审计指标见[fashion-verification.json](fashion-verification.json)。原二分类和UCI标准库路径无需安装深度学习依赖。

## 数据权利与归属

Fashion-MNIST由Han Xiao、Kashif Rasul、Roland Vollgraf（Zalando Research）提供，MIT许可，Copyright © 2017 Zalando SE。官方仓库：https://github.com/zalandoresearch/fashion-mnist 。固定commit为b2617bb6d3ffa2e429640350f613e3291e10b141；下载使用该官方仓库HTTPS raw URL，拒绝跳转。完整许可保存于[licenses/fashion-mnist-MIT.txt](licenses/fashion-mnist-MIT.txt)。论文：Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms，https://arxiv.org/abs/1708.07747 。

四个官方gzip共30,878,645字节，固定解压上限合计54,950,048字节；逐文件MD5（官方README）、SHA256、压缩/原始字节数、IDX头/类别分布同时验证。每次Worker重新验证gzip，不能依靠可修改的metadata.json。固定目录分量与文件使用no-follow文件描述符；无原始文件持久化解压、任意归档路径或任意下载URL。单文件连接/读取30秒、总体下载180秒上限；缓存缺失/损坏不静默换源或覆盖。

## 可选环境与命令

本次环境为Linux x86_64 / CPython3.12.14。仅此配方安装独立venv，不改变标准库核心。CPU专用PyTorch官方wheel，不安装CUDA、torchvision或torchaudio；ONNX、ONNX Script、ONNX Runtime及NumPy来自PyPI。安装后执行pip check；lock针对本次平台而不是多平台兼容声明。

```sh
python3 -m venv .venv-fashion
.venv-fashion/bin/python -m pip install -r requirements-fashion.lock
.venv-fashion/bin/python -m pip check
export MP_HOME="$PWD/.runtime/fashion-case"
.venv-fashion/bin/python -m model_platform init --tenant demo
.venv-fashion/bin/python scripts/fetch_fashion.py
```

终端A启动`.venv-fashion/bin/python -m model_platform api --port 8080`，终端B启动`.venv-fashion/bin/python -m model_platform worker`；两者使用同一MP_HOME。终端C运行`.venv-fashion/bin/python scripts/demo_fashion.py`。API始终loopback；demo读取本地demo凭据而不打印。代码摘要绑定提交时源码，运行期间不要修改model_platform源码。

缺少可选库时Fashion attempt失败为`optional_dependencies_missing`，旧配方仍工作；缓存缺失/损坏为`dataset_missing`/`dataset_corrupt`，不会自动联网重下、绕过校验或注册失败产物。

## 2026-10-12 实际验收

- 固定seed20261012、6000训练（600/类）、5epochs、Adam0.001；官方完整10000 test一次最终评估，无test调参。准确率 **80.76%**，macro-F1 **0.8048124060**，log loss **0.5301679883**。这是小训练子集基线，不是高精度或生产质量承诺。
- ONNX Runtime与PyTorch在全部10000 test上类别一致率100%，最大绝对logit误差1.335144e-5（预先固定阈值1e-4）。训练样本动态batch1/2/17/128均一致；合成偏差注入测试证实超阈值和argmax变化都会拒绝。
- 9098可训练参数，单文件ONNX **48,485 bytes**；SHA256 `c3d72d128db7f713c0370c43ca37843ff36c7fd3bc8d3da22538b4c613687c2d`。
- 训练循环2.342秒、导出/重新加载1.163秒、全test双后端评估0.581秒；从HTTP提交到观察到登记8.026秒。10次HTTP推理（10张/次）稳定，延迟中位数9.203ms。这是共享CPU单次观察，含鉴权、ready探测、序列化，不是性能基准。
- 持续服务真实运行、停止后503；使用同一真实训练模型的隔离runtime副本，API kill/restart后重新加载、预测完全一致、旧孤儿进程退出，损坏产物被拒绝且不ready，随后停止。正式原产物保持完好。
- 同一环境回归官方UCI完整3823/1797：accuracy94.99165275%、macro-F1 0.9501111995、log loss0.1563669033；旧二分类端到端与停止503通过。
- CPython3.12.14、Linux6.18.44 x86_64/glibc2.41、9逻辑CPU可见，训练使用2线程，GPU未用。实际版本torch2.10.0+cpu（cuda=None）、numpy2.3.5、onnx1.20.1、onnxscript0.5.6、onnxruntime1.23.2。21直接/传递依赖完整hash锁，官方wheel252,708,049 bytes（约241MiB），venv约1.1GiB；全部21wheel下载校验通过，pip check通过。
- 无pickle/checkpoint加载；无远端部署。验收自启API、Worker与服务进程均清理。

最终自动化：Fashion环境75/75通过；原零依赖venv64通过、11项可选跳过；系统解释器65通过、10项跳过（系统有NumPy但无完整深度学习栈）。编译、文档相对链接和pip check均通过；数据安全测试不需要网络。具体命令见[verification.md](verification.md)。
