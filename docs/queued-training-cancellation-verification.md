# 排队撤销后端验收

2026-10-20，本地源码基线为 model-platform 706b582dc55ea9b2c3d92505e0babb53b8311f57。本记录不是推送、生产升级或实际模型质量证明。

## 已执行

- Python 3.12.14，新增取消与迁移模块21项测试全部通过，另在既有独立评估队列模块增加1项全局领取顺序回归。
- 受影响后端组合回归共142项，142通过，0失败，0跳过。范围包括新取消/迁移、模型导入与原 imported-origin 迁移、报告/比较/图片预览、独立评估、自定义数据集绑定、配方目录及不创建身份的 Worker 就绪测试。
- `python -m compileall -q model_platform scripts tests` 通过。

取消验证使用真实 SQLite 事务与 loopback HTTP/Auth，测试身份仅为内存中固定 fixture。竞争测试使用多个独立进程、连接和 barrier，包括 claim/cancel、重复 cancel、retry/cancel；断言不存在取消成功同时产生运行 attempt 的混合状态。成功返回发生在事务提交之后。

迁移使用隔离旧 schema fixture，包含非连续 rowid、历史模型/评估/attempt/日志、数据绑定、自定义索引/触发器，验证全部原行和 schema 保留、旧分页 cursor 的成员边界不变、备份0600且不可覆盖/跟随符号链接、故障回滚、重复迁移校验、marker/触发器/未知列拒绝。崩溃遗留的训练和独立评估 running 记录迁移时不变，随后通过正常恢复入口变为 interrupted/failed。只对测试 fixture 做离线备份恢复副本核验。

## 范围限制

组合 runner 固定内存 cursor 材料，并通过审计钩子拒绝访问 credentials.json、cursor-integrity.key 和临时 cursor 密钥文件。会初始化持久身份的历史 subprocess/HTTP 全套测试及两项真实 Worker 子进程测试未在本轮运行；其中 succeeded 取消响应预期已从501更新为409，新 loopback/API 测试独立覆盖相同新状态语义。未声称全仓库测试通过。

本轮没有新建凭据、下载权重、重新训练视觉模型、操作真实保留的运行目录、向远端推送或执行生产部署。前端/BFF、浏览器和最终打包验收由各自记录单独说明。
