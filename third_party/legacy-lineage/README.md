# 历史训练源码与资源归档

[返回首页](../../README.md) · [随机初始化历史训练](../../docs/随机初始化历史训练.md) · [训练沿革](../../docs/历史训练沿革.md)

本目录保存通用预训练、decision-v1 迁移和闪刀专家训练的历史源码及卡组资源。训练器按版本保存，配套文件用 SHA256 固定身份。感谢 [ygo-agent](https://github.com/sbl1996/ygo-agent) 的作者和贡献者提供模型、强化学习与环境实现。

## 目录与用途

| 路径 | 内容 |
| --- | --- |
| `code/versions/` | 五个历史 PPO 训练器快照 |
| `code/network/` | decision-v1 迁移前的 JAX 网络与结构化编码器 |
| `code/decision-v1/` | 259M 所属 corrected 工程的四个模型文件 |
| `code/library/` | 历史 ygoai 依赖、卡号映射、checkpoint 与 cluster stats 工具 |
| `native/ygoenv/` | 历史 Python/C++ 原生环境源码与构建文件 |
| `native/versions/` | v11 与 tribute-fix 的阶段源码 |
| `assets/deck/largepool-v3/` | 训练、验证、测试、官方验证卡组及原始划分清单 |
| `records/` | 续训清单、corrected 协议、启动参数与 259M 元数据的公共副本 |
| `docs/hash-manifest.json` | 本目录文件大小、SHA256、来源及公开时的调整 |

## 训练器版本

| 文件 | SHA256 前 8 位 | 来源与作用 |
| --- | --- | --- |
| `cleanba.cef71dbd.py` | `cef71dbd` | 早期通用训练器；其身份与 2026-08-26 续训 manifest 一致 |
| `cleanba.aa1fe2ef.py` | `aa1fe2ef` | decision-v1 迁移前备份，加入 cluster stats |
| `cleanba.1026a147.py` | `1026a147` | decision-v1 训练、辅助损失及 checkpoint 元数据 |
| `cleanba.expert0919.py` | `71b590fe` | 2026-09-19 专家工程保存的训练器副本 |
| `cleanba.corrected0920.py` | `71b590fe` | 2026-09-20 corrected 工程副本，与前一文件逐字节一致 |

`code/network/` 的两个文件直接对应架构迁移前备份。`code/decision-v1/` 的四个文件直接对应 corrected 工程，并与本次核查时 `src/ygoai/rl/jax/` 的对应文件一致。共享依赖和阶段网络各自保留，组装历史工程时按训练阶段选择版本。

## 校验与资源获取

```bash
python scripts/verify_historical_archive.py
python scripts/download_historical_resources.py --install
python scripts/verify_historical_archive.py --native
```

训练集有 11,459 副卡组，验证集 1,433 副，测试集 1,432 副，官方验证集 15 副。卡组文本及划分清单直接随 Git 发布。两个早期 Linux x86_64 / CPython 3.11 原生二进制通过[历史资源 Release](https://github.com/zwdmw/GGBOY-ygoagent/releases/tag/training-lineage-20261009)提供，固定下载身份见 [historical-resources.json](../../configs/historical-resources.json)。冻结语义表、数据库、Lua 脚本和专家资源由现有 v0.1.0 资源包提供。

历史启动参数与源码身份已经归档；历史多阶段 GPU 复跑的验收进度记录在[随机初始化历史训练](../../docs/随机初始化历史训练.md)。当前整合入口的安装、短训练和评估记录见[全新 GPU 复现报告](../../docs/全新GPU复现报告.md)。

## 来源记录

2026-10-09 核查了原始训练树及其历史导出。早期训练器与续训清单、迁移前网络与备份、corrected 模型与所属工程逐文件交叉核对。本地历史导出中 14,499 个文件通过原始 SHA256SUMS 校验；新增采集的 53 个证据文件也全部通过下载后校验。

原始日志及服务器来源信息保存于独立审计目录。公共记录使用实验相对路径。`split_summary.json` 及少量阶段元数据中的机器目录字段、路径键已改为相对名称，清单同时保留原文件哈希与公开副本哈希。训练代码、YDK 和其余数据文件按归档字节保存；根目录 `.gitattributes` 固定这些文件的字节，方便在不同系统上校验。

## 许可

历史模型、训练器与原生环境源码沿用已保存的 ygo-agent MIT 与 Apache-2.0 声明，见 [third_party/ygo-agent](../ygo-agent) 和 [NOTICE](../../NOTICE.md)。卡组数据与原生链接依赖的来源记录见[第三方资源说明](../../docs/第三方资源说明.md)及 [SOURCES.json](../SOURCES.json)。
