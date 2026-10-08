# 全新 GPU 环境复现报告

[从零复现训练](从零复现训练.md) · [安装与部署](安装与部署.md) · [复现数据](全新GPU复现数据.json) · [项目验收](验收报告.md)

2026-10-08，在新建项目目录和独立虚拟环境中，先安装依赖，再实际执行五个公开配方的短程训练、模型注册、GPU 推理和 64 局成对评估。源码从公开 GitHub 仓库克隆；虚拟环境由 Miniconda 的 CPython 3.11 创建，未继承其他环境的包。

## 环境与安装

| 项目 | 实际配置 |
| --- | --- |
| 系统 | Ubuntu 22.04 / Linux x86_64 / glibc 2.35 |
| Python | CPython 3.11.11 |
| GPU | NVIDIA GeForce RTX 4080 SUPER，驱动报告 32760 MiB |
| 驱动 | 595.71.05 |
| JAX / jaxlib / CUDA plugin | 0.4.28 |
| 环境隔离 | `.venv`，`include-system-site-packages = false` |
| 安装命令 | `python -m pip install -e '.[cuda]'` |
| 依赖检查 | `No broken requirements found.` |
| GPU 预检 | `JAX_PLATFORMS=cuda` 下显示 `cuda:0` |

完整依赖版本见 [GPU 依赖快照](GPU依赖快照.txt)。需要与本次实验一致的依赖版本时，可在安装命令中加上 `-c docs/GPU依赖快照.txt`。

服务器默认的 HTTP 软件源下载缓慢；实测后改用 HTTPS 源 `https://repo.huaweicloud.com/repository/pypi/simple` 完成安装。可通过 `--index-url` 为自己的网络选择软件源。

资源使用公开 v0.1.0 的本地归档，走指南中提供的 `--archive ... --install` 流程。归档为 293216446 字节，SHA256 为 `a718650043db90e3a51e5f8cd94d81351daacdfec263466aa670de1a70611ffd`；大小和 hash 均校验通过。三个主模型的资源验证均通过。

`ygo-sky install-native` 安装的参考引擎 SHA256 为 `65628475e6373f506fb59abc02627ab88ab8e5ee85ba959cc3503279eba4aee5`，`ygoenv` 导入成功。

## 文档原样短训练

```bash
ygo-sky train --config configs/train/smoke.json \
  --platform gpu --output runs/smoke
```

| 检查 | 结果 |
| --- | --- |
| 配置 | 仓库中的 `smoke.json`，保持原样 |
| 起点 | `base-463m`，累计 463001600 环境步 |
| 追加采样 | 32 环境步，每批 16 步，共两批 |
| 最终 checkpoint 累计步数 | 463001632 |
| `exit.json` | `exit_code = 0` |
| 参数检查 | 293 个张量形状一致，286 个发生变化，全部数值有限 |
| 输出 | 两份 checkpoint 和同名 sidecar、TensorBoard、cluster stats、`run.json` |

最终权重 SHA256：`a505d27a33cd5dd19758ba34ef4d1df1c44e43ebafd33b05ef6cb25df1a784df`。训练器的 actor 预采样日志会记录额外采样；以上步数以保存 checkpoint 的 sidecar 为准。

接着执行：

```bash
ygo-sky register-model \
  --checkpoint runs/smoke/checkpoints/1791470918_463001632_steps.flax_model \
  --name repro-smoke
ygo-sky verify --model repro-smoke
ygo-sky infer --model repro-smoke --device gpu \
  --observation resources/example-observation.npz --legal-count 2
```

注册、校验和 GPU 推理通过。推理返回动作索引 `1`、合法数 `2`、累计步数 `463001632`，模型 hash 与新权重一致。首次推理耗时约 42.5 秒，包含加载与编译预热；稳定运行时的延迟需要在预热后另行测量。

## 配方与对战验证

五份训练配方均已通过实际 `tyro` 参数解析。两份对手池按照文档命令准备，PPO 池含 2 个成员，镜像池含 7 个成员；成员 hash 和准确累计步数均匹配 manifest。正式训练命令均通过 `--dry-run` 展开。

实际短程训练结果如下。每次运行均完成两批采样，检查点累计步数与起点加本次环境步数一致；五份检查点均通过 hash、张量形状和数值有限检查。

| 配方 | 起点累计环境步 | 本次追加环境步 | 最终累计环境步 | 发生变化的张量 | 退出码 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `smoke` | 463001600 | 32 | 463001632 | 286 / 293 | 0 |
| `corrected-selfplay` | 259000320 | 64 | 259000384 | 286 / 293 | 0 |
| `mixed-50-30-20` | 259000320 | 64 | 259000384 | 286 / 293 | 0 |
| `ppo-continue` | 463001600 | 256 | 463001856 | 286 / 293 | 0 |
| `mirror-specialist` | 463001600 | 256 | 463001856 | 286 / 293 | 0 |

PPO 日志记录了冻结对手分支的 1/8 与 3/8 环境分配。镜像专项保持 8 个 actor，其中 2 个使用固定锚点、6 个使用动态对手池，日志记录了这两类分支。完整短跑配置、检查点 SHA256、池成员和分支日志摘录见[复现数据](全新GPU复现数据.json)。

新模型对 `base-463m` 按指南执行 32 对换座位评估：

```bash
ygo-sky evaluate --config configs/eval/native-mirror.json \
  --model repro-smoke --opponent base-463m \
  --pairs 32 --platform gpu --output runs/eval-smoke
```

64 局全部完成，32 对均完整；新模型 35 胜、29 负、0 平，未完成局为 0。评估逐步检查模型输出，所有步骤通过数值有限检查。双方换座位时初始观测 hash 相同。胜率为 54.69%，Wilson 95% 区间为 42.57%–66.27%。该结果用于训练后模型的链路验收；模型强度的比较需要独立评估。胜负区间、对局身份和逐局结果保存在[复现数据](全新GPU复现数据.json)中。

其他配方使用两批采样做短跑，保留模型起点、随机种子、训练器、卡组混合比例和全部对手池成员。生成本次短跑配置的命令如下；此前先按[训练指南](从零复现训练.md#4-重现-463m-之后的公开实验路线)准备两份对手池。

```bash
python - <<'PY'
import json
from pathlib import Path

folder = Path("runs/recipe-check-configs")
folder.mkdir(parents=True)
for name in ("corrected-selfplay", "mixed-50-30-20", "ppo-continue", "mirror-specialist"):
    config = json.loads(Path(f"configs/train/{name}.json").read_text())
    args = config["args"]
    args.update(num_steps=8,
                local_num_envs=8 if name == "ppo-continue" else 2,
                local_env_threads=4 if name == "ppo-continue" else 2,
                num_minibatches=2 if name == "ppo-continue" else 1,
                log_frequency=1)
    batch = args["num_steps"] * args["local_num_envs"] * args["num_actor_threads"] * len(args["actor_device_ids"])
    args["total_timesteps"] = 2 * batch
    (folder / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
PY

ygo-sky train --config runs/recipe-check-configs/corrected-selfplay.json \
  --platform gpu --output runs/check-corrected-selfplay
ygo-sky train --config runs/recipe-check-configs/mixed-50-30-20.json \
  --platform gpu --output runs/check-mixed-50-30-20
ygo-sky train --config runs/recipe-check-configs/ppo-continue.json \
  --pool runs/pool-ppo --platform gpu --output runs/check-ppo-continue
ygo-sky train --config runs/recipe-check-configs/mirror-specialist.json \
  --pool runs/pool-mirror --platform gpu --output runs/check-mirror-specialist
```

各配方会进行独立的首次 JAX 编译，本次多线程配方的预热耗时数分钟。等待模型保存后检查 `exit.json` 和 sidecar，避免将编译时间误当成训练停止。

## 本次修复

起始源码提交为 `d5d9d5daa943f6f6739e4cdd04eb15002bc2cb4b`。代码修复已发布于 [23f2f09](https://github.com/zwdmw/GGBOY-ygoagent/commit/23f2f09f80c100ea32a819e610221277e1bf90fa)，并在实验目录应用相同修改。实验中发现并同步修复：

- GPU 指南原来安装 `.[train]`，实际 GPU 命令需要 `.[cuda]`；现在统一依赖安装方式，并增加 JAX GPU 和原生引擎预检。
- 原生包将 `setuptools`、`wheel` 列为运行依赖，导致 `--no-deps` 安装后的 `pip check` 报缺少 `wheel`。它们由安装时的构建环境提供，已从运行依赖中移除。
- `ppo-continue.json` 带有 `cleanba_pool.Args` 未定义的镜像锚点和策略参数，实际解析以退出码 2 失败。已移除这些字段，保留 PPO 历史对手池训练器支持的配置。新增回归检查覆盖所有公开配方的参数。
- 将短训练中的“32 learner step”改为“32 环境步、两批采样”，与实际步数统计一致。

## 结果解释

本次验证覆盖从新环境安装依赖，到实际 PPO 更新、保存、注册、推理及对战评估的执行链。正式长训练继续使用公开配方中的总步数和并行规模；本次短跑用于验证各训练入口和保存行为。

历史模型的数值重现还涉及初始化权重、完整随机状态与 optimizer 状态。本项目提供固定公开权重的续训路线，训练时重新初始化 optimizer；本报告按实际执行的链路与步数记录结果。
