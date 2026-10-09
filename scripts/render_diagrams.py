"""Render the architecture and training diagrams with the Python standard library.

Run: python scripts/render_diagrams.py
The SVGs are committed so GitHub can show them without a build step.
"""
from __future__ import annotations

import ast
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images"


class SVG:
    def __init__(self, height, title, description, language):
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc" xml:lang="{language}">
<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker>
  <marker id="teal-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#0f766e"/></marker>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', 'Noto Sans SC', sans-serif; fill: #172b43; }}
    .title {{ font-size: 32px; font-weight: 700; letter-spacing: -.5px; }}
    .subtitle {{ font-size: 18px; fill: #54657a; }}
    .heading {{ font-size: 22px; font-weight: 650; }}
    .body {{ font-size: 18px; }}
    .small {{ font-size: 16px; fill: #54657a; }}
    .label {{ font-size: 15px; font-weight: 700; fill: #54657a; letter-spacing: 1px; }}
    .mono {{ font-family: 'Cascadia Code', Consolas, monospace; font-size: 16px; fill: #34516a; }}
    .step {{ font-size: 18px; font-weight: 700; fill: #0f766e; }}
  </style>
</defs>
<rect width="1200" height="{height}" rx="24" fill="#f6f8fc"/>
<rect x="36" y="35" width="5" height="32" rx="2" fill="#0f766e"/>
''']
        self.text(56, 61, title, "title")
        self.text(56, 96, description, "subtitle")

    def rect(self, x, y, width, height, fill="#fff", stroke="#dbe3ed", dashed=False, radius=16):
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')

    def text(self, x, y, value, kind="body", color=None):
        color_style = f' style="fill:{color}"' if color else ""
        self.parts.append(f'<text x="{x}" y="{y}" class="{kind}"{color_style}>{escape(str(value))}</text>')

    def lines(self, x, y, values, kind="body", spacing=29):
        for index, value in enumerate(values):
            self.text(x, y + index * spacing, value, kind)

    def line(self, points, teal=False, dashed=False):
        route = "M " + " L ".join(f"{x} {y}" for x, y in points)
        color = "#0f766e" if teal else "#64748b"
        marker = "teal-arrow" if teal else "arrow"
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        # The light halo keeps crossings distinct and labels unobstructed.
        self.parts.append(f'<path d="{route}" fill="none" stroke="#f6f8fc" stroke-width="7" stroke-linejoin="round"/>')
        self.parts.append(f'<path d="{route}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" marker-end="url(#{marker})"{dash}/>')

    def box(self, x, y, width, height, title, body, tint="white", dashed=False, body_kind="body"):
        fills = {"white": "#ffffff", "teal": "#e9f7f4", "blue": "#edf3fe", "violet": "#f1edfc", "amber": "#fff7e9"}
        strokes = {"white": "#dbe3ed", "teal": "#a3d8ce", "blue": "#b9cbee", "violet": "#cbbdeb", "amber": "#e9cc93"}
        self.parts.append("<g>")
        self.rect(x, y, width, height, fills[tint], strokes[tint], dashed)
        self.text(x + 20, y + 35, title, "heading")
        self.lines(x + 20, y + 69, body, body_kind)
        self.parts.append("</g>")

    def save(self, name):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        target = OUTPUT / name
        target.write_text("\n".join(self.parts) + "\n</svg>\n", encoding="utf-8", newline="\n")
        print(target.relative_to(ROOT).as_posix())


def metadata():
    report = json.loads((ROOT / "docs/全新GPU复现数据.json").read_text(encoding="utf-8"))
    args = report["evaluation"]["configuration"]["expert_checkpoint_metadata"]["model_args"]
    archived = json.loads((ROOT / "third_party/legacy-lineage/records/corrected/20260920_259000320_steps.flax_model.json").read_text(encoding="utf-8"))
    if args != archived["model_args"]:
        raise ValueError("Archived and accepted model structures differ; review diagrams")
    tree = ast.parse((ROOT / "src/ygo_sky/observation.py").read_text(encoding="utf-8"))
    shapes = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SHAPES" for t in node.targets))
    steps = {name: json.loads((ROOT / f"configs/model/{name}.json").read_text(encoding="utf-8"))["global_step"] for name in ("initial-259m", "base-463m", "specialist-464m")}
    return args, shapes, steps


def architecture(lang, args, shapes):
    zh = lang == "zh"
    t = lambda chinese, english: chinese if zh else english
    c, heads, layers = args["num_channels"], args["structured_heads"], args["decision_layers"]
    s = SVG(935, t("模型结构 · decision-v1", "Model architecture · decision-v1"), t("结构化局面 → 对局记忆 → 合法动作评分", "Structured observations → recurrent memory → legal action scores"), lang)
    for x, label in [(36, t("01  观测与语义", "01  INPUTS")), (326, t("02  结构化编码", "02  ENCODING")), (676, t("03  记忆与决策", "03  MEMORY & HEADS")), (972, t("04  输出", "04  OUTPUTS"))]:
        s.text(x, 136, label, "label")
    s.box(36, 156, 240, 185, t("卡片与冻结语义", "Cards + semantics"), [f'cards_: {shapes["cards_"][0]} × {shapes["cards_"][1]}', t("位置 / 控制者 / 动态状态", "Zones / controller / state"), t("CDB 属性 + Lua 效果特征", "CDB + Lua effect features"), t("语义表固定，编码器可训练", "Fixed tables; learned encoder")], "blue", body_kind="small")
    s.box(36, 371, 240, 119, t("全局与选择提示", "Global + selection"), [t("LP / 回合 / 阶段 / 张数", "LP / turn / phase / counts"), "global_: 23 · selection_: 12"], body_kind="small")
    s.box(36, 520, 240, 146, t("公开事件", "Public events"), [t("32 条事件及关联卡片", "32 events + card references"), "public_events_: 32 × 16", t("事件顺序编码 + 注意力", "Position encoding + attention")], body_kind="small")
    s.box(36, 686, 240, 139, t("候选动作与角色", "Candidates + roles"), ["action_ir_: 128 × 24", t("来源 / 目标 / 分组 / 选择", "Source / target / groups"), t("合法候选与填充掩码", "Legal slots + padding mask")], body_kind="small")
    s.rect(326, 156, 300, 656, "#fff", "#a3d8ce")
    s.text(346, 193, "StructuredEncoder", "heading")
    s.text(346, 224, t(f"共享特征宽度 {c}", f"Shared feature width: {c}"), "small")
    s.rect(346, 251, 260, 150, "#e9f7f4", "#e9f7f4", radius=12)
    s.text(362, 284, t("局面注意力", "Scene attention"), "heading")
    s.lines(362, 316, [t("卡片 + 全局 + 公开事件", "Cards + global + events"), t(f'{args["structured_scene_layers"]} 层 Transformer · {heads} 头', f'{args["structured_scene_layers"]} Transformer layers · {heads} heads'), t("构建当前局面特征", "Contextual scene features")], "small", 28)
    s.rect(346, 425, 260, 164, "#edf3fe", "#edf3fe", radius=12)
    s.text(362, 459, t("候选动作编码", "Candidate encoding"), "heading")
    s.lines(362, 492, [t("绑定动作来源与目标角色", "Bind source / target roles"), t(f'{args["structured_action_cross_layers"]} 层动作 → 局面交叉注意力', f'{args["structured_action_cross_layers"]} action → scene cross layer'), t(f'{args["structured_action_set_layers"]} 层候选集合注意力', f'{args["structured_action_set_layers"]} candidate self-attention layer'), t("比较同一提示的候选动作", "Compare options in a prompt")], "small", 26)
    s.text(346, 641, t("三路编码输出", "Three encoder outputs"), "heading")
    s.lines(346, 676, [t("状态向量 → LSTM", "State vector → LSTM"), t("动作特征 → FiLMActor", "Action features → FiLMActor"), t("局面与角色 → DecisionActor", "Scene + roles → DecisionActor"), t("状态汇聚融合局面与动作", "State pooling joins scene + actions")], "small", 30)
    s.box(676, 156, 238, 145, "LSTM", [t(f'{args["rnn_channels"]} 维循环状态', f'{args["rnn_channels"]}-dim recurrent state'), t("逐次决策更新对局记忆", "Memory across decisions"), t("每局维护独立状态", "Separate state for each duel")], "teal", body_kind="small")
    s.box(676, 370, 238, 145, "FiLMActor", [t("LSTM 状态调制动作特征", "Memory conditions actions"), t("生成基础动作分数", "Base action logits"), t(f"每个候选动作 {c} 维", f"{c} features per candidate")], "blue", body_kind="small")
    s.box(676, 566, 238, 172, "DecisionActor", [t(f"{layers} 层决策残差分支", f"{layers} residual decision layers"), t("记忆 + 局面 + 动作角色", "Memory + scene + roles"), t("再次比较候选并补充分数", "Compare; add logit delta"), t("附带公开变化预测头", "Public-outcome auxiliary")], "violet", body_kind="small")
    s.box(972, 423, 192, 139, t("合成动作分数", "Policy scores"), [t("基础分数 + 决策残差", "Base logits + delta"), t("屏蔽填充候选", "Mask padded slots"), t("得到合法动作 logits", "Legal action logits")], "teal", body_kind="small")
    s.box(972, 610, 192, 108, t("选择动作", "Choose action"), [t("训练：按概率采样", "Training: sample"), t("推理：argmax", "Inference: argmax")], body_kind="small")
    s.box(676, 777, 238, 98, t("价值头 · 训练", "Critic · training"), [t(f'{args["critic_depth"]} × {args["critic_width"]} MLP → V(s)', f'{args["critic_depth"]} × {args["critic_width"]} MLP → V(s)')], "amber", dashed=True, body_kind="small")
    s.box(972, 777, 192, 98, t("辅助头 · 训练", "Auxiliary head"), [t("下一决策点的公开变化", "Next public-state delta")], "amber", dashed=True, body_kind="small")
    # Input routes, with a distinct branch for the candidates.
    for y in (245, 430, 585, 749):
        s.line([(276, y), (326, y)])
    # Recurrent features are shared by policy and value heads.
    s.line([(914, 200), (940, 200), (940, 145), (795, 145), (795, 156)], teal=True)
    s.line([(676, 268), (650, 268), (650, 397), (676, 397)])
    s.line([(650, 397), (650, 592), (676, 592)])
    s.line([(650, 592), (650, 826), (676, 826)], dashed=True)
    s.line([(626, 222), (676, 222)], teal=True)
    s.line([(626, 445), (676, 445)])
    s.line([(626, 660), (676, 660)])
    s.line([(914, 437), (972, 437)])
    s.line([(914, 643), (946, 643), (946, 544), (972, 544)], teal=True)
    s.line([(1068, 562), (1068, 610)], teal=True)
    s.line([(850, 738), (850, 755), (1068, 755), (1068, 777)], dashed=True)
    s.text(36, 879, t("实线：策略计算 · 虚线：训练监督", "Solid: policy computation · Dashed: training supervision"), "small")
    s.text(36, 910, t("实现与参数依据：StructuredEncoder / RNNAgent / DecisionActor / 已验收 checkpoint 元数据", "Source: StructuredEncoder / RNNAgent / DecisionActor / accepted checkpoint metadata"), "small")
    s.save(f"model-architecture.{lang}.svg")


def training(lang):
    zh = lang == "zh"
    t = lambda chinese, english: chinese if zh else english
    s = SVG(875, t("训练全流程 · 从准备到对战", "Training workflow · resources to duels"), t("固定实验输入 → 并行采样 ↔ PPO 更新 → 保存 → 评估 → 推理与接入", "Pin inputs → parallel rollouts ↔ PPO updates → save → evaluate → inference & duels"), lang)
    s.box(36, 140, 1128, 114, t("01  准备本次实验", "01  Prepare the experiment"), [t("配置与 seed  ·  公开起点权重  ·  语义表与卡号映射  ·  引擎版本  ·  卡组与冻结对手池", "Config + seed  ·  published checkpoint  ·  semantics + code list  ·  engine  ·  decks + frozen opponents"), t("逐项绑定资源身份；保存展开参数、SHA256 与训练记录", "Record expanded arguments, resource SHA256 identities and run metadata")], "blue", body_kind="small")
    s.box(36, 334, 294, 239, t("02  并行对战环境", "02  Duel environments"), ["YGOPro core + ygoenv", t("按配方选择卡组与对手", "Recipe selects decks / opponents"), t("当前策略自博弈", "Self-play with current policy"), t("或与冻结历史模型对战", "Or face frozen historical policies"), t("返回观测、奖励与终局", "Return observations / rewards"), t("累计计数按环境步记录", "Count environment steps")], "teal", body_kind="small")
    s.box(399, 334, 330, 239, t("03  Actor 采样线程", "03  Actor rollout threads"), [t("当前权重 + 双方独立 LSTM", "Current weights + per-seat LSTM"), t("对合法动作按策略概率采样", "Sample from the legal policy"), t("收集连续 num_steps 步轨迹", "Collect num_steps-long rollouts"), t("观测 / 动作 / log-prob / 价值", "Observations / actions / log-probs"), t("奖励 / done / 公开变化标签", "Values / rewards / done / deltas"), t("胜负奖励 + 配方 LP shaping", "Win/loss reward + LP shaping")], body_kind="small")
    s.box(798, 334, 366, 239, t("04  Learner 更新", "04  Learner updates"), [t("GAE 计算优势与价值目标", "GAE advantages + value targets"), t("PPO 策略损失 + 价值损失", "PPO policy loss + value loss"), t("熵项 + 公开变化辅助损失", "Entropy + public-outcome auxiliary"), t("按序列与 minibatch 更新 Adam", "Sequence minibatches + Adam"), t("将新权重送回 Actor", "Send refreshed weights to actors"), t("重启时按配方初始化优化器", "Recipes reinitialize Adam on restart")], "violet", body_kind="small")
    s.line([(180, 254), (180, 334)])
    s.line([(1050, 254), (1050, 334)])
    s.line([(982, 334), (982, 294), (564, 294), (564, 334)], teal=True)
    s.text(599, 283, t("新策略权重", "Updated policy weights"), "small")
    s.line([(330, 426), (399, 426)])
    s.text(342, 413, t("观测", "Obs"), "small")
    s.line([(399, 518), (330, 518)], teal=True)
    s.text(342, 505, t("动作", "Act"), "small")
    s.line([(729, 454), (798, 454)])
    s.text(741, 441, t("轨迹", "Batch"), "small")
    s.line([(982, 573), (982, 610), (208, 610), (208, 648)])
    s.box(36, 648, 344, 147, t("05  保存模型与记录", "05  Save artifacts"), ["checkpoint + JSON sidecar", t("精确步数 / 参数 / 资源身份", "Exact steps / args / identities"), "TensorBoard / run.json"], body_kind="small")
    s.box(428, 648, 344, 147, t("06  固定协议评估", "06  Paired evaluation"), [t("固定 seed，同牌序交换座位", "Fixed seeds; swap player seats"), t("双方使用独立 LSTM 状态", "Independent recurrent states"), t("登记胜负 / 超时 / checkpoint hash", "Log results / timeouts / model hash")], "blue", body_kind="small")
    s.box(820, 648, 344, 147, t("07  注册与对战接入", "07  Register & deploy"), ["register-model → verify → infer", t("HTTP 推理 / YGOPro TCP 对战", "HTTP inference / YGOPro TCP"), t("已验收示例：233 服 GPU 对战", "Accepted example: GPU duel on 233")], "teal", body_kind="small")
    s.line([(380, 720), (428, 720)])
    s.line([(772, 720), (820, 720)], teal=True)
    s.text(36, 835, t("历史阶段如何衔接见训练沿革图；新实验保留各自的权重、配置与评估结果。", "See the lineage diagram for historical stages. Each experiment keeps its own weights, config and results."), "small")
    s.save(f"training-workflow.{lang}.svg")


def lineage(lang, steps):
    zh = lang == "zh"
    t = lambda chinese, english: chinese if zh else english
    s = SVG(748, t("训练沿革 · 通用策略到闪刀专家", "Training lineage · general policy to Sky Striker"), t("两条训练计数：通用阶段积累对战经验，专家阶段继承权重并重新计步", "General training develops the policy; the expert branch inherits weights and starts a new counter"), lang)
    s.rect(36, 140, 1128, 222, "#edf3fe", "#b9cbee")
    s.text(60, 175, t("通用训练", "GENERAL TRAINING"), "label")
    xs = (60, 338, 616, 894)
    upper = [
        (t("随机初始化", "Random init"), [t("环境步 0", "Environment step 0"), "seed 0 · checkpoint=None"]),
        (t("通用自博弈", "General self-play"), [t("通用计数逐步推进", "General counter advances"), "→ 732,954,624"]),
        (t("decision-v1 迁移", "decision-v1 update"), ["732,954,624", t("加入决策残差与辅助头", "Add decision + auxiliary heads")]),
        (t("专家分叉起点", "Branch checkpoint"), ["861,929,472", t("保存通用模型权重", "General policy checkpoint")]),
    ]
    for x, (title, body) in zip(xs, upper):
        s.box(x, 203, 246, 125, title, body, body_kind="small")
    for x in xs[:-1]:
        s.line([(x + 246, 268), (x + 278, 268)])
    s.rect(36, 425, 1128, 231, "#e9f7f4", "#a3d8ce")
    s.text(60, 460, t("闪刀姬专家训练", "SKY STRIKER EXPERT TRAINING"), "label")
    lower = [
        (t("继承通用权重", "Inherit weights"), [t("专家计数从 0 开始", "Expert counter starts at 0"), t("闪刀姬专项自博弈", "Sky Striker self-play")]),
        (t("259M 续训起点", "259M starting policy"), [f'{steps["initial-259m"]:,}', "initial-259m"]),
        (t("463M 基线", "463M baseline"), [f'{steps["base-463m"]:,}', "base-463m"]),
        (t("镜像专项模型", "Mirror specialist"), [f'{steps["specialist-464m"]:,}', "specialist-464m"]),
    ]
    for x, (title, body) in zip(xs, lower):
        s.box(x, 492, 246, 125, title, body, "teal" if x >= 616 else "white", body_kind="small")
    for x in xs[:-1]:
        s.line([(x + 246, 555), (x + 278, 555)], teal=True)
    s.line([(1017, 328), (1017, 393), (20, 393), (20, 555), (60, 555)], teal=True)
    s.text(338, 386, t("继承权重 · 专家训练重新计步", "Inherit weights · reset the expert-stage counter"), "step")
    extra = steps["specialist-464m"] - steps["base-463m"]
    s.text(36, 693, t(f"镜像专项在 463M 基线上追加 {extra:,} 环境步；每个节点的完整权重身份见资源契约与沿革记录。", f"The mirror specialist adds {extra:,} environment steps to the baseline. Checkpoint identities are recorded separately."), "small")
    s.text(36, 723, t("历史关系已核对记录与 SHA256；当前入口的短训练验收见 GPU 复现报告。", "Historical links are checked against records and SHA256. Current short-run acceptance is in the GPU report."), "small")
    s.save(f"training-lineage.{lang}.svg")


def main():
    args, shapes, steps = metadata()
    for lang in ("zh", "en"):
        architecture(lang, args, shapes)
        training(lang)
        lineage(lang, steps)


if __name__ == "__main__":
    main()
