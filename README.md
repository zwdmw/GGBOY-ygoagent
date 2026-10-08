# GGBOY-ygoagent

[English](README.en.md) · [快速开始](docs/快速开始.md) · [常见问题](docs/常见问题.md) · [参与贡献](CONTRIBUTING.md) · [下载 v0.1.0](https://github.com/zwdmw/GGBOY-ygoagent/releases/tag/v0.1.0)

闪刀姬专家模型的训练、验证、推理、HTTP 服务和 YGOPro TCP 对战项目。统一命令为 `ygo-sky`，支持接入 233 服，也可指定兼容服务器和端口。模型、语义表、卡片数据库和引擎通过 SHA256 绑定，源码与大资源分开发布。

`463M` 指基线累计训练步数 **463001600**，不是参数量。资源包另含 `464001024` 步的镜像专项模型，两者有独立身份和评估配置。

感谢 [ygo-agent](https://github.com/sbl1996/ygo-agent) 的作者和贡献者开放模型、强化学习训练及环境实现。本项目的 `ygoai`、训练器和修改版 `ygoenv` 基于该项目整理与扩展，上游版权与 MIT / Apache-2.0 声明保留在 [third_party/ygo-agent](third_party/ygo-agent)。也感谢 YGOPro core、卡片脚本及相关依赖的维护者。

## 许可范围

| 内容 | 许可或当前状态 | 说明 |
| --- | --- | --- |
| 本项目新增的原创整合代码与文档 | MIT | [LICENSE](LICENSE)、[范围说明](NOTICE.md) |
| ygo-agent / 修改版 ygoenv | 保留上游 MIT 与 Apache-2.0 声明 | [上游声明](third_party/ygo-agent/YGO-AGENT-LICENSE.txt) |
| YGOPro core | MIT；静态链接依赖许可待补齐 | [来源清单](third_party/SOURCES.json) |
| Lua 卡片脚本 | 随资源保留 GPLv2；精确来源与修改记录待核实 | [资源说明](docs/第三方资源说明.md) |
| 网络组件、权重、卡片数据、语义表等派生资源 | 仍有未明确的来源或再分发条件 | [待确认事项](docs/第三方资源说明.md) |

根目录 MIT 许可仅覆盖上述原创部分；各组成部分按各自许可使用。公开下载资源不代表所有内容已获得统一 MIT 授权。

## 选择你的使用路线

| 想做什么 | 从哪里开始 | 需要什么 |
| --- | --- | --- |
| 运行模型或提供 HTTP 推理 | [路线一：运行模型](docs/快速开始.md#路线一运行模型) | CPU 即可，无需原生引擎 |
| 与朋友在 233 服对战，或接入其他服务器 | [路线二：接入 233](docs/快速开始.md#路线二接入-233) | 模型资源、兼容卡组、同房间的对手 |
| 继续训练或进行原生评估 | [路线三：继续训练](docs/快速开始.md#路线三继续训练) | 训练依赖、原生引擎；GPU 路径已做短训练验收 |
| 修文档、改代码、报告问题 | [贡献指南](CONTRIBUTING.md) | 轻量检查无需权重、GPU 或对战服务器 |

## CPU 首次运行

已验收环境为 **Linux x86_64 / CPython 3.11**。在终端依次执行：

```bash
git clone https://github.com/zwdmw/GGBOY-ygoagent.git
cd GGBOY-ygoagent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export YGO_SKY_HOME="$PWD"

# 下载约 280 MiB，自动校验 SHA256；重复执行可复用有效缓存或已安装资源
python scripts/download_resources.py --install
ygo-sky verify
ygo-sky infer --device cpu --observation resources/example-observation.npz --legal-count 2
```

首次推理包含 JAX 编译，可能需要几十秒到一分钟，耗时随机器变化。一次已验收的示例输出如下：

```json
{
  "action": 1,
  "legal_count": 2,
  "inference_ms": 7763.585820999651,
  "model_sha256": "20ddfd6cc92a6d960e617bce2a94ec8ccb50acd785445879b57e1bbe342f1558",
  "global_step": 463001600
}
```

`action` 是合法动作索引；示例观测用于接口验收。输出中应有模型 hash、累计步数和合法动作数，耗时无需与示例一致。下载失败或校验不通过时，参见 [常见问题](docs/常见问题.md)。

## 下载与版本

[v0.1.0 Release](https://github.com/zwdmw/GGBOY-ygoagent/releases/tag/v0.1.0) 提供以下附件：

| 附件 | 大小 | 用途 |
| --- | --- | --- |
| `sky-striker-resources.tar.gz` | 293,216,446 字节，约 279.6 MiB | 三个主模型、历史 checkpoint、卡片与语义资源、卡组、参考原生引擎 |
| `sky-striker-source.tar.gz` | 约 1.3 MiB | v0.1.0 的固定源码快照 |
| `sky_striker_expert-0.1.0-py3-none-any.whl` | 约 243 KiB | v0.1.0 Python 包；配置与资源仍需源码目录 |
| `manifest.json`、`SHA256SUMS` | 小型文本 | 版本、文件大小与 SHA256 |
| `acceptance.json`、`README.md` | 小型文本 | 该版本的验收结果与使用说明 |

主分支包含后续文档与工具改进；[CHANGELOG](CHANGELOG.md) 记录尚未发布的变更。下载工具使用 [固定下载清单](configs/resources.json)，不会自动切换到未知的新权重。目前资源以整包发布，按用途拆包的工作列在 [路线图](docs/路线图.md)。

## 环境与功能状态

| 路径 | 当前验收情况 |
| --- | --- |
| Linux x86_64 / Python 3.11 CPU 安装与推理 | 干净环境通过 |
| GPU 推理与 32 步短训练 | 兼容的现有 JAX CUDA 环境通过；全新 CUDA 安装未验收 |
| HTTP 直接运行 | 真实模型请求通过 |
| 233：`s1.ygo233.com:233`，协议 `0x1362` | 兼容卡组完成一场对局、387 次模型决策 |
| Windows、macOS、WSL2、Python 3.12 完整运行 | 未验收；参考原生二进制限定 Linux x86_64 / Python 3.11 |
| Docker / systemd | 提供模板，尚未验收 |
| 实验原生 TCP 桥 | 未通过输入等价回归；当前 233 客户端使用独立网络编码路径 |

历史验证/测试划分未完整归档，目标卡组存在于训练池中。小样本验收说明链路能运行，不能据此宣称胜率优势、泛化能力或训练收敛。

## 项目结构与文档

```text
src/ygo_sky/     CLI、模型加载、训练入口、评估、HTTP、TCP 客户端
src/ygoai/       上游模型与 PPO 基础库
configs/        模型契约、资源下载清单、训练配方、评估与 233 配置
native/         训练引擎、实验 TCP 桥源码及构建工具
scripts/        资源准备、轻量检查、数据审计、引擎回归与接入验收
examples/       HTTP 调用示例
deploy/         Docker / systemd 模板
docs/           使用、训练、资源、FAQ、路线图和验收说明
third_party/    上游许可与来源记录
resources/      从 Release 单独安装的模型与运行资源
.github/        CI、Issue 与 Pull Request 模板
```

- [快速开始](docs/快速开始.md) / [常见问题](docs/常见问题.md)
- [安装与部署](docs/安装与部署.md) / [对战与推理](docs/对战与推理.md)
- [训练与验证](docs/训练与验证.md) / [模型与资源契约](docs/模型与资源契约.md)
- [验收报告](docs/验收报告.md) / [第三方资源说明](docs/第三方资源说明.md)
- [贡献指南](CONTRIBUTING.md) / [路线图](docs/路线图.md) / [整理记录](docs/整理计划.md)

欢迎中文或英文 Issue 和 PR。修复文档、补充可复现的安装记录、核实资源来源，都能帮助其他人更顺利地使用项目。
