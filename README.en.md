# GGBOY-ygoagent

[中文](README.md) · [Quick start (Chinese)](docs/快速开始.md) · [Training from scratch (Chinese)](docs/从零复现训练.md) · [Contributing](CONTRIBUTING.md) · [Release v0.1.0](https://github.com/zwdmw/GGBOY-ygoagent/releases/tag/v0.1.0)

A Sky Striker expert policy project with training, evaluation, inference, an HTTP service, and a YGOPro TCP client. The CLI is `ygo-sky`. **463M means 463,001,600 accumulated training steps**, not the parameter count.

Thanks to [ygo-agent](https://github.com/sbl1996/ygo-agent), its authors and contributors for the policy, reinforcement learning and environment implementations on which this project builds. Their notices are retained in [third_party/ygo-agent](third_party/ygo-agent).

## License scope

The root [MIT license](LICENSE) covers original integration code and documentation. Upstream code retains its own terms, including the ygo-agent MIT and Apache-2.0 notices. Card scripts carry GPLv2 notices. Licensing/provenance for the network components, weights and some derived resources still has unresolved items. See [NOTICE](NOTICE.md), the [license table](README.md#许可范围) and [resource notes](docs/第三方资源说明.md) before redistribution.

## First CPU inference

The verified clean installation target is **Linux x86_64 / CPython 3.11**. CPU inference does not need the native training engine.

```bash
git clone https://github.com/zwdmw/GGBOY-ygoagent.git
cd GGBOY-ygoagent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export YGO_SKY_HOME="$PWD"
python scripts/download_resources.py --install
ygo-sky verify
ygo-sky infer --device cpu --observation resources/example-observation.npz --legal-count 2
```

The download is about 280 MiB and has a pinned size/SHA256. Valid caches and existing installations are reused; invalid existing files are preserved. For offline installation, use `python scripts/download_resources.py --archive /path/to/sky-striker-resources.tar.gz --install`. Use `--verify-only` to check installed baseline resources without network access.

The first call compiles the JAX graph and can take tens of seconds or about a minute. Output contains a legal action index, legal action count, inference time, model SHA256 and `global_step: 463001600`. The bundled observation is an interface fixture.

## HTTP and duels

```bash
ygo-sky serve --device cpu --host 127.0.0.1 --port 8765
```

The service provides `/health`, `/infer` and `/reset`. Use a separate session per duel; set `YGO_API_TOKEN` when listening outside localhost. See [the HTTP example](examples/http_infer.py) and [deployment notes](docs/安装与部署.md).

To join 233, have an opponent enter the same room with a compatible client:

```bash
export YGO_ROOM='your-private-room'
ygo-sky duel --config configs/duel/233.json --device cpu --output runs/233-first
```

Defaults: `s1.ygo233.com:233`, protocol `0x1362`, and `sky-striker-233.ydk` (40 main / 14 extra). Use `--host`, `--port` and `--version` for another compatible server. The completed server acceptance duel used GPU; CPU server duels have not been verified and initial compilation may affect timeouts. The compatible deck was checked for connectivity, not competitive strength.

## Training and verification status

Training needs `.[train]`, `ygo-sky install-native` and `python -m pip install --no-deps -e native/ygoenv`. The reference binary requires Linux x86_64 / CPython 3.11. For GPU dependencies use `.[cuda]`; see [training steps](docs/快速开始.md#路线三继续训练).

CPU inference and direct HTTP serving passed acceptance. GPU inference and a 32-step training run passed in an existing compatible JAX CUDA environment. A clean CUDA install, Docker, systemd, and full Windows/macOS/WSL2 runtime support remain unverified. The experimental native TCP bridge has not passed input parity checks. Small acceptance samples do not establish win rate or generalization.

## Contribute

Chinese and English Issues and PRs are welcome. [CONTRIBUTING](CONTRIBUTING.md) includes an English checklist. Lightweight checks need neither model downloads nor a GPU:

```bash
python -m pip install PyYAML
python scripts/check_project.py
```

See the [FAQ](docs/常见问题.md), [roadmap](docs/路线图.md) and [changelog](CHANGELOG.md). Most detailed documentation is currently in Chinese; translations are welcome.
