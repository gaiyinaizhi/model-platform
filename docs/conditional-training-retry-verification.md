# 有条件训练重试后端验收

2026-10-20，本地源码基线为已合并的 `d1497aaeda3cf79e0644d7dac78209dfd9e71718`。本记录是本地验证，不表示新变更已推送、远端 CI 已运行、已部署或已修改实际运行目录。

## 实际结果

Linux / Python 3.12.14，两份不含原 `.runtime`、数据、权重或凭据的隔离源码副本：

- 纯标准库环境：`python -m unittest discover -s tests -v`，445 项，316 通过、129 项按既有可选依赖/实际资产条件跳过，0 失败/错误，50.830 秒。
- 新建可选 CPU 环境：按未改动的 `requirements-detection.lock` 和 `requirements-pretrained.lock` 执行 `pip install --require-hashes --only-binary=:all:`。25 个锁定直接/传递包安装成功；`pip check`、实际导入和 CPU provider 检查通过。包含 torch 2.10.0+cpu、NumPy 2.3.5、ONNX 1.20.1、ONNX Runtime 1.23.2、ONNX Script 0.5.6、OpenCV 4.13.0 和 Pillow 12.3.0。
- 完全使用原 `python -m scripts.run_optional_regressions`：发现 445 项，六个原有精确集成排除 ID 各发现一次，其余 439/439 通过，0 跳过/失败/错误，153.933 秒。带 main guard 的 spawn-safe runner 和全部排除规则逐字节未变。
- 新增 `test_conditional_training_retry.py` 20 项全部通过；另新增真实发布的合成数据集版本绑定重试测试，在可选全套内通过，也独立运行通过。
- `python -m compileall -q model_platform scripts tests`、变更文件空白/结尾换行及相关文档相对链接检查通过。
- 独立安全复核无阻塞发现，并独立重跑新 20 项全部通过。

首次使用原图片预览环境运行严格可选套件时，因缺少训练开发依赖产生 10 项 skip，runner 按既有规则返回失败。随后新建完整锁定环境，才取得上述 439/439 结果。没有删除测试、增加排除、允许 skip、修改锁文件或放宽生产 schema/安全检查。

## 覆盖

新增验证使用真实 SQLite 事务、loopback HTTP/Auth 和多进程独立连接/barrier：

- 同一 run 和不可变输入、环境、请求摘要、数据集绑定、历史日志与 attempt 不变；排队只追加一次 retry audit，领取才增加下一序号 attempt。
- 四个并发相同失败代际请求最多一个受理；丢弃已经受理的 HTTP 响应，再领取并快速失败后，重放旧 ID 返回 409，不能产生第三次执行。新失败必须显式采用其新 ID；三次上限不变。
- foreign/missing、其他 run 的 attempt、无 attempt、断序/非整数/非失败历史、带模型的失败记录、非法请求体/ID、共享训练/独立评估满容量均无状态写入。
- 源码摘要变化只拒绝新的有条件路径，旧 `{}` 兼容；恢复正确源码可重试过去的 `execution_code_changed`，Worker 已加载源码防护仍生效。
- retry 与 claim/cancel/recovery 竞争、取消后重放、原创建顺序、独立评估不变；202 详情可以真实推进为 running 或任一终态，不伪造 queued。
- 数据库定义和迁移 marker 不变；全套同时执行现有 schema 兼容、平台生命周期、取消/迁移、模型导入、数据集、报告/比较/预览、配方和 Worker 就绪回归。

保障是同一已确认失败最多受理一次排队，不是网络层只出现一个 POST。底层浏览器/传输重传同一前置条件仍须被安全处理。

## 边界

新增 HTTP 测试只用固定内存测试身份；既有完整套件按原流程在隔离临时目录生成短期测试身份并清理，没有接触真实用户凭据、实际账号或保留运行目录。测试创建的临时 API、Worker、serving 和合成数据由原清理流程回收。

六项真实资产/官方缓存集成排除见[原 CI 契约](ci.md)，未把排除计作通过。本轮没有下载数据/预训练权重、重训真实视觉模型、修改实际运行目录、迁移用户数据库、部署或声称模型质量改善。合成记录/图片与数值 CPU 回归不代表真实 Cat 或自定义业务数据训练验收。

后端运行源码摘要已变化，新有条件路径会准确拒绝旧摘要 run；不重写任何旧记录。上述结果记录于本轮发布前；获得发布授权后，配套前端 source lock 必须更新为实际发布的后端 commit 及完整源码摘要，远端 CI 以对应 PR/commit 的 Actions 记录为准。前端/BFF/浏览器结果由其交付记录说明，不以本后端结果代替。
