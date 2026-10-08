"""Policies for the network client.

A policy is anything with ``choose(state) -> int``.  The index it returns is an
index into ``state.actions``, which is built by :mod:`.actions` and ordered
exactly like ``ygoenv``'s ``legal_actions_`` -- the same convention the offline
re-simulator recovers from recorded responses.

``RandomPolicy`` and ``FirstPolicy`` exist to validate the channel end to end
without a model in the loop; ``FirstPolicy`` is the same "always take option 0"
rule as ygoenv's ``GreedyAI``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

__all__ = [
    "DecisionState",
    "Policy",
    "RandomPolicy",
    "FirstPolicy",
    "ScriptedPolicy",
    "make_policy",
    "POLICIES",
]


@dataclass
class DecisionState:
    """Everything the client can tell a policy about one decision."""

    msg: int
    player: int  # who the engine is asking (always us)
    our_player: int  # our duel-side index
    actions: list
    board: object = None
    turn: int = 0
    phase: int = 0
    lp: tuple = (0, 0)
    extra: dict = field(default_factory=dict)
    #: 直接给一份**未掩码**的 ``StateSnapshot``。给了它，打分路径就不再经
    #: ``ShadowBoard`` 反推盘面——离线粒子（`mirrorforce/search`）手上是
    #: ``DuelDriver``，``state.capture`` 出来的就是这个类型，与语料侧同型。
    #:
    #: **掩码规则不受影响**：两条路都只经 ``worldmodel.state.mask_for`` 一处。
    #: ``ShadowBoard`` 从来不是过滤器，它是线上路径的状态容器；报文的分发过滤
    #: 在 ygopro 服务端（``single_duel.cpp``），不在这条路径上。
    snapshot: object = None
    #: 披露账本（``board`` 为空时由这里给）
    disclosure: object = None
    #: 回合方（``board`` 为空时由这里给）
    turn_player: int = 0

    @property
    def n(self) -> int:
        return len(self.actions)


class Policy:
    name = "policy"

    def choose(self, state: DecisionState) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def on_duel_end(self, result) -> None:
        pass


class RandomPolicy(Policy):
    """Uniform over the legal actions."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, state: DecisionState) -> int:
        return self.rng.randrange(state.n)


class FirstPolicy(Policy):
    """Always the first option -- ygoenv's ``GreedyAI`` rule."""

    name = "first"

    def choose(self, state: DecisionState) -> int:
        return 0


class ScriptedPolicy(Policy):
    """Replays a fixed list of indices; used by the tests."""

    name = "scripted"

    def __init__(self, picks: list[int], fallback: int = 0):
        self.picks = list(picks)
        self.fallback = fallback
        self.pos = 0

    def choose(self, state: DecisionState) -> int:
        if self.pos < len(self.picks):
            idx = self.picks[self.pos]
            self.pos += 1
            return min(idx, state.n - 1)
        return min(self.fallback, state.n - 1)


POLICIES = {
    "random": RandomPolicy,
    "first": FirstPolicy,
}


def make_policy(name: str, seed: int = 0) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; have {sorted(POLICIES)}")
    cls = POLICIES[name]
    try:
        return cls(seed=seed)
    except TypeError:
        return cls()
