# Sky Striker 463M Expert

闪刀姬专家策略的训练、验证、推理、HTTP 服务与 YGOPro TCP 对战项目。
统一入口为 `ygo-sky`，模型、卡片数据库、语义表、卡组和原生引擎通过 SHA256 绑定。
源码与大资源独立打包，原实验快照和内部来源记录留在整理工作区。

`463M` 表示基线累计训练步数 `463001600`。另附 `464001024` 步的镜像专项模型，两个模型有独立身份和评估配置。

## 致谢

感谢 [ygo-agent](https://github.com/sbl1996/ygo-agent) 项目及其作者和贡献者开放游戏王智能体的模型、强化学习训练和环境实现。本项目的 `ygoai`、训练器和修改版 `ygoenv` 基于该项目整理与扩展；原有版权与 MIT / Apache-2.0 声明保留在 [third_party/ygo-agent](third_party/ygo-agent)。感谢 YGOPro core、卡片脚本及相关开源依赖的作者和维护者。

## 下载

公开源码仓库：[zwdmw/GGBOY-ygoagent](https://github.com/zwdmw/GGBOY-ygoagent)。
模型、历史对手池、卡片资源和参考引擎从 [v0.1.0 Release](https://github.com/zwdmw/GGBOY-ygoagent/releases/tag/v0.1.0) 下载。Release 同时提供源码包、wheel、`manifest.json`、`SHA256SUMS` 和该版本的独立安装验收结果。

## 快速开始

已验收环境为 Linux x86_64 / CPython 3.11。CPU 干净环境安装和推理通过；GPU 训练与推理在兼容的现有 JAX CUDA 环境通过。全新环境的 CUDA 依赖安装尚未完成验收。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
export YGO_SKY_HOME="$PWD"

mkdir -p release
curl -fL https://github.com/zwdmw/GGBOY-ygoagent/releases/download/v0.1.0/sky-striker-resources.tar.gz -o release/sky-striker-resources.tar.gz
curl -fL https://github.com/zwdmw/GGBOY-ygoagent/releases/download/v0.1.0/manifest.json -o release/manifest.json
ygo-sky install-resources release/sky-striker-resources.tar.gz --sha256 "$(python -c 'import json; print(json.load(open("release/manifest.json"))["resources_sha256"])')"
ygo-sky verify
ygo-sky infer --device cpu --observation resources/example-observation.npz --legal-count 2
```

GPU 环境安装选项为 `python -m pip install -e '.[cuda]'`。训练与原生评估还需要安装引擎：

```bash
ygo-sky install-native
python -m pip install --no-deps -e native/ygoenv
ygo-sky train --config configs/train/smoke.json --output runs/smoke
ygo-sky evaluate --config configs/eval/native-mirror.json --pairs 1 --output runs/eval
```

新增 checkpoint 使用 `ygo-sky register-model --checkpoint PATH --name my-model` 注册。
实际训练配置、资源 hash、引擎身份和进程退出状态保存在输出目录。

## 服务与 233 接入

```bash
ygo-sky serve --device cpu --host 127.0.0.1 --port 8765
```

服务提供 `/health`、`/infer` 和 `/reset`；每场对局使用独立 `session`。
非本机地址监听必须配置 `YGO_API_TOKEN`。请求示例见 `examples/http_infer.py`。

```bash
export YGO_ROOM='your-private-room'
ygo-sky duel --config configs/duel/233.json --device gpu --output runs/233

# 任意兼容服务器和端口
ygo-sky duel --host HOST --port PORT --version 0x1362 \
  --deck resources/decks/sky-striker-233.ydk --device gpu --output runs/custom
```

233 默认地址为 `s1.ygo233.com:233`。默认使用经过服务器收牌检查的兼容卡组 `sky-striker-233.ydk`。
冻结训练卡组另存为 `sky-striker.ydk`；兼容变体仅用于接入验证，未评估竞技强度。
状态、决策、脱敏网络消息及服务器返回的原始录像写入对战输出目录。

## 项目结构

```text
src/ygo_sky/     CLI、模型加载、训练入口、评估、HTTP、TCP 客户端
src/ygoai/      原模型和 PPO 所需基础库
configs/        模型契约、训练配方、评估条件、233 接入配置
native/         训练引擎和隔离 TCP 桥源码、固定依赖与构建工具
scripts/        语义表生成、数据审计、引擎回归和接入验收
examples/       HTTP 调用示例
deploy/         Docker / systemd 部署模板
docs/           安装、训练、对战、资源契约和验收说明
third_party/    已收集的上游许可和来源说明
resources/      独立资源包：权重、卡片脚本、数据库、卡组、参考引擎
```

整理工作区额外保留 `snapshots/`、`evidence/`、`inventory/` 和 `release/`。
发行源码包省去原始快照、内部目录、虚拟环境、构建缓存及运行日志。

## 文档与复现范围

- [安装与部署](docs/安装与部署.md)
- [训练与验证](docs/训练与验证.md)
- [对战与推理](docs/对战与推理.md)
- [模型与资源契约](docs/模型与资源契约.md)
- [验收报告](docs/验收报告.md)
- [第三方资源与许可](docs/第三方资源说明.md)
- [整理计划与完成情况](docs/整理计划.md)

已归档的训练从 259M 初始化权重和 463M 基线继续，恢复权重时不恢复 optimizer。
历史独立验证集/测试集划分未完整归档，固定目标卡组存在于训练池中；报告中的小样本对局用于功能验收，不能据此声称泛化能力或胜率优势。

本项目新增的原创整合代码与文档采用 [MIT License](LICENSE)。上游代码继续适用各自许可，详见 [NOTICE.md](NOTICE.md)。网络组件来源许可、卡片数据、模型权重及派生资源的再分发条件仍有未明确项，公开提供下载不表示这些内容获得了统一 MIT 授权，详见第三方资源说明。
