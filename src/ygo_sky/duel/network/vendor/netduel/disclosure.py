"""公开身份账本：谁看过哪一张，双方都知道。

「披露即输入」是本项目的一条硬规则：一张卡被发动、被展示、被确认之后，它的
身份对双方就是公开的，此后**不得再被重新匿名化**。这个模块是这条规则在整个
仓库里的唯一实现——语料生成侧（``worldmodel/engine.py``）与在线 bridge 侧
（``netduel/board.py``）都用它，因为它喂的是同一个可见性谓词
（``worldmodel/state.identity_visible``），两边各写一份就会悄悄漂移。

放在 ``netduel`` 而不是 ``worldmodel``：``worldmodel.state`` 已经依赖
``netduel.board``，反向导入会成环。这个模块只依赖 ``netduel.constants``。
"""

from __future__ import annotations

import struct
from collections import Counter

from . import constants as C

__all__ = ["DisclosureLedger", "resolve_disclosure"]


class DisclosureLedger:
    """公开身份的记账。**语料侧与在线 bridge 共用这一份实现。**

    「披露即输入」要求：一张卡被发动、被展示、被确认之后，它的身份对双方就是
    公开的，此后不得再被重新匿名化。这个账本只吃公开报文，标准客户端与 MD
    adapter 都能跑同一套，所以由它派生的输入列不违反部署观测契约。

    按 ``(控者, 区, 卡号) -> 张数`` 记，不按实例坐标记。原因在
    ``field::remove_card``（``ygopro-core/field.cpp:204``）：手牌/卡组/墓地/
    除外/额外卡组都是 vector，删掉一张就 ``reset_sequence`` 把后面的区序整体
    前移，而**这些位移不发** ``MSG_MOVE``。按坐标记的账本会在对手打出任意一张
    手牌之后悄悄对不上号，把公开过的卡又变回未知。同卡号的实例本来就不可区分，
    多重集张数不受区序位移与洗牌影响。

    以前两侧各写了一份坐标集合（``worldmodel/engine.py`` 与
    ``netduel/board.py``），两份都有这个缺陷，且没有任何机制保证它们同步。
    """

    __slots__ = ("counts",)

    #: 洗过之后没人知道这一区里哪一张在哪儿。丢弃是保守方向：只会少公开。
    FORGET_ON_SHUFFLE = (C.LOCATION_DECK, C.LOCATION_EXTRA)

    def __init__(self) -> None:
        self.counts: Counter = Counter()

    def clear(self) -> None:
        self.counts.clear()

    def disclose(self, controller: int, location: int, code: int) -> None:
        if code:
            self.counts[(int(controller), int(location), int(code))] += 1

    def forget_zone(self, controller: int, location: int) -> None:
        controller, location = int(controller), int(location)
        for key in [k for k in self.counts
                    if k[0] == controller and k[1] == location]:
            del self.counts[key]

    def observe_move(self, previous: int, current: int, code: int) -> None:
        """一张已公开的卡换区时把这份记账挪过去。

        源区没有这个卡号的公开记录就什么都不做——那说明动的是身份未公开的卡。
        ``MSG_MOVE`` 在两个隐藏区之间移动时卡号会被服务器抹掉，抹掉的那一条
        这里就不认账，宁可少记一次公开。
        """
        code = int(code) & 0x7FFFFFFF
        if not code or not previous:
            return
        from_key = (int(previous) & 0xFF, (int(previous) >> 8) & 0xFF, code)
        if self.counts.get(from_key, 0) <= 0:
            return
        self.counts[from_key] -= 1
        if self.counts[from_key] <= 0:
            del self.counts[from_key]
        if current:
            self.counts[
                (int(current) & 0xFF, (int(current) >> 8) & 0xFF, code)
            ] += 1

    # -- 报文解析。两侧喂进来的都是原始载荷，解析只有这一份。 ---------------

    def observe_chaining(self, body: bytes) -> None:
        """``MSG_CHAINING``：发动即公开，位置无关（手牌发动的手坑同样公开）。"""
        if len(body) < 16:
            return
        code, at = struct.unpack_from("<II", body, 0)
        self.disclose(at & 0xFF, (at >> 8) & 0xFF, int(code))

    def observe_confirm(self, body: bytes, msg: int) -> None:
        """``MSG_CONFIRM_CARDS`` / ``_DECKTOP`` / ``_EXTRATOP``：展示即公开。

        每张卡 7 字节：``code(4) controler(1) location(1) sequence(1)``。
        ``MSG_DECK_TOP`` 故意不接：它只给 controler 与带高位标志的 code，
        凑不出区，而它指的是卡组里的卡。
        """
        head = 3 if msg == C.MSG_CONFIRM_CARDS else 2
        if len(body) <= head:
            return
        count, offset = body[head - 1], head
        for _ in range(count):
            if offset + 7 > len(body):
                break
            (code,) = struct.unpack_from("<I", body, offset)
            # 高位是"表侧"标志，不是卡号的一部分
            self.disclose(body[offset + 4], body[offset + 5], code & 0x7FFFFFFF)
            offset += 7

    def observe_move_message(self, body: bytes) -> None:
        if len(body) < 16:
            return
        code, previous, current, _reason = struct.unpack_from("<IIII", body, 0)
        self.observe_move(previous, current, code)

    def observe_shuffle(self, msg: int, body: bytes) -> None:
        """洗牌与墓地卡组互换会打散一区的对应关系。

        ``MSG_SHUFFLE_HAND`` **不在此列**：它只改手牌顺序，而本账本按多重集
        记，顺序本来就不参与。丢弃手牌记账等于把已经公开的信息又藏起来。
        """
        if not body:
            return
        player = int(body[0])
        if msg == C.MSG_SHUFFLE_DECK:
            self.forget_zone(player, C.LOCATION_DECK)
        elif msg == C.MSG_SHUFFLE_EXTRA:
            self.forget_zone(player, C.LOCATION_EXTRA)
        elif msg == C.MSG_SWAP_GRAVE_DECK:
            self.forget_zone(player, C.LOCATION_DECK)
            self.forget_zone(player, C.LOCATION_GRAVE)

    #: 只对卡组主人可见的确认。``MSG_CONFIRM_CARDS`` 在
    #: ``libduel.cpp:1063`` 之后写 ``playerid``，服务器按这个 id 决定发给谁：
    #: 玩家检索自己卡组时，另一方只看见"他检索了"，看不见检索到什么。
    OWNER_ONLY_LOCATIONS = (C.LOCATION_DECK,)

    def resolve(self, snapshot, viewer: int | None = None) -> frozenset[tuple]:
        return resolve_disclosure(snapshot, self.counts, viewer)


def resolve_disclosure(snapshot, disclosed, viewer: int | None = None) -> frozenset[tuple]:
    """把「(控者, 区, 卡号) -> 已公开张数」解析成具体的实例坐标集合。

    ``DuelDriver.disclosed`` 只按多重集记账，因为手牌/卡组/墓地是 vector，
    ``field::remove_card`` 会静默重排区序（见 ``engine.py`` 的说明）。落到
    哪几个区序在信息上是任意的——同卡号的实例本来就不可区分——所以这里按
    **区序升序**做规范化分配，而不是去读引擎的真实身份对应关系。多记的张数
    按该区实际张数截断：重复公开同一张不会把没公开过的那张也带出来。

    ``snapshot`` 必须是**未掩码**的裸快照，掩码后的手牌已经没有卡号可分组。

    ``viewer`` 给了就按 audience 过滤。卡组内的确认只发给卡组主人
    （``parse_public_events`` 给它的 audience 是 ``0b10``），所以另一方的视角
    里那几张仍然是未知的。语料生成侧读的是全知报文流，不过滤就会把对手私下
    检索到的卡号写进我方视角——那是泄漏，不是"披露即输入"。
    """
    if not disclosed:
        return frozenset()
    groups: dict[tuple[int, int, int], list[int]] = {}
    for card in snapshot.cards:
        if not card.code:
            continue
        groups.setdefault(
            (card.controller, card.location, card.code), []
        ).append(card.sequence)
    out = set()
    for (controller, location, code), count in disclosed.items():
        if (viewer is not None and controller != viewer
                and location in DisclosureLedger.OWNER_ONLY_LOCATIONS):
            continue
        sequences = sorted(groups.get((controller, location, code), ()))
        for sequence in sequences[: max(int(count), 0)]:
            out.add((controller, location, sequence, code))
    return frozenset(out)
