# 参与贡献 / Contributing

欢迎中文或英文 Issue 和 Pull Request。文档纠错、安装记录、FAQ、来源与许可核查，都适合作为首次贡献。你不需要下载权重或拥有 GPU 才能参与。

## 报告问题

先查看 [快速开始](docs/快速开始.md) 和 [FAQ](docs/常见问题.md)，再搜索已有 Issue。仍未解决时使用 [问题模板](https://github.com/zwdmw/GGBOY-ygoagent/issues/new/choose)，提供：

- 版本或 commit、系统与架构、Python 版本；GPU 问题加驱动与 JAX / jaxlib 版本。
- 可复制的执行命令、预期结果、实际结果和最短复现步骤。
- 相关错误末尾；资源问题提供文件大小 / SHA256，对战问题提供脱敏后的状态摘要。

请移除 SSH 信息、密码、令牌、房间口令、私人聊天及不相关的个人路径。无需上传大权重或完整私人日志。若你想反馈使用体验，也可以使用功能 / 文档建议模板。

## 本地修改与轻量检查

在 GitHub fork 仓库后 clone 你的 fork，为本次修改新建分支。以下命令适用于 Linux。Windows 文档维护可用 `py -3.12 -m venv .venv` 创建环境，再用 `.venv/Scripts/Activate.ps1` 激活；激活后仍使用 `python` 执行安装与检查命令。

```bash
git clone https://github.com/YOUR_USERNAME/GGBOY-ygoagent.git
cd GGBOY-ygoagent
git switch -c docs/improve-quickstart
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install PyYAML
python scripts/check_project.py
git diff --check
```

替换 `YOUR_USERNAME` 和分支名。轻量检查包含项目文档的本地链接与锚点、JSON / YAML 格式、Python 语法，以及无需资源的契约、网络和下载测试。检查过程不下载权重、不训练、不连接对战服务器。

如改动影响打包，另执行：

```bash
python -m pip install 'setuptools>=75' wheel
python -m pip wheel --no-deps . --wheel-dir release/ci-wheel
python scripts/check_project.py --wheel-dir release/ci-wheel
```

CI 在 Linux 的 Python 3.11 / 3.12 上执行这些轻量检查。这个矩阵只说明源码检查通过，完整模型运行仍以 [验收状态](README.md#环境与功能状态) 为准。

## 涉及模型、训练或协议的修改

按 [快速开始](docs/快速开始.md) 安装资源和相应依赖，并执行与改动相关的真实检查。

```bash
# 需要完整资源；部分测试检查训练命令，但不启动长训练
PYTHONPATH=src:scripts python -m unittest discover -s tests -v
```

训练或原生评估的改动需额外按 [训练与验证](docs/训练与验证.md) 验证。协议改动可先使用 fixtures，再在你有使用权限的兼容服务器上测试，避免把真实服务器访问作为默认 CI。

PR 中写明模型 hash、配置、种子、环境和结果；无法执行的路径如实标明。不要改写固定模型配置的 hash 来绕过资源校验。注册新模型见 [资源契约](docs/模型与资源契约.md)；已有发布版本的资源身份应可追溯。

## 提交 Pull Request

- 一次 PR 聚焦一个问题，说明触发条件、改动后的行为和验证结果。
- 用户可见的变更加入 [CHANGELOG](CHANGELOG.md) 的 Unreleased，并同步相关文档。
- 为影响文件读写、协议或资源契约的变化添加有意义的小型测试；纯文档修改通常无需新单元测试。
- 保留上游版权和许可；新增第三方内容注明来源、版本与适用许可。原创贡献按本项目相应部分的许可提交。
- 使用 PR 模板列出执行过的检查和未验证路径。提交后留意 CI 结果并响应 review。

项目结构见 [首页](README.md#项目结构与文档)，下一步方向见 [路线图](docs/路线图.md)。欢迎不同经验水平的贡献者；请围绕具体问题讨论，尊重彼此的时间。

## English checklist

Chinese and English Issues and PRs are welcome. Documentation fixes, installation reports, translations and provenance research are useful first contributions.

1. Fork and clone the repository; create a branch for one focused change.
2. Install Python 3.11 or 3.12 and PyYAML, then run `python scripts/check_project.py`. Model weights and a GPU are unnecessary for these checks.
3. For packaging changes, run `python -m pip wheel --no-deps . --wheel-dir release/ci-wheel` and `python scripts/check_project.py --wheel-dir release/ci-wheel` with setuptools >=75 and wheel installed.
4. For model/training/protocol changes, perform relevant resource-backed validation and report environment, hashes, configuration and results. Clearly identify untested paths.
5. Update affected documentation and the Unreleased changelog; retain third-party notices and cite the source/license of new dependencies or assets.
6. Open a PR using the template. Include the problem, resulting behavior and actual checks. Remove credentials and private data from logs.

Use the [Issue forms](https://github.com/zwdmw/GGBOY-ygoagent/issues/new/choose) for reproducible bugs or usability suggestions. See [NOTICE](NOTICE.md) for license scope; some resource and network licensing remains unresolved.
