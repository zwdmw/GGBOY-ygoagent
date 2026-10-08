"""Shadow board: the duel state rebuilt from the message stream alone.

A network client has no duel engine.  Everything it knows comes from
``STOC_GAME_MSG``, which is what makes this the same problem the product's
shadow duel has to solve.

The load-bearing message is ``MSG_UPDATE_DATA``: after a board change the host
re-queries a zone and forwards the result -- but only for **three** zones.
``RefreshMzone``, ``RefreshSzone`` and ``RefreshHand`` run from the main
message block; ``RefreshGrave`` is reachable only from ``MSG_SWAP_GRAVE_DECK``,
``RefreshExtra`` only from ``MSG_SHUFFLE_EXTRA`` (``single_duel.cpp:882`` and
``:871``), and the banished pile has no refresh at all.  So the graveyard, the
banished pile and the extra deck are not cross-checked by a refresh: the
message stream is their only source, and a card leaving one of them renumbers
every card behind it whether we follow the move or not.

The payload of a refresh is

    uint8  MSG_UPDATE_DATA
    uint8  player
    uint8  location
    then, per card slot, one self-delimiting segment:
        int32  segment length        (4 == empty slot, nothing follows)
        uint32 present_flags         (QUERY_* bits actually included)
        ...    the fields, in the order card::get_infos writes them

Two things make this workable and are worth stating plainly:

* the segments are length-prefixed, so the *number of cards in the zone* falls
  out of parsing -- unlike the reference GUI client, which walks its own list
  and therefore needs separate move tracking just to know the zone size;
* the host queries with ``use_cache=1``, so a segment carries only the fields
  that changed since the core last queried that card.  The core clears a card's
  cache whenever its ``info_location`` changes (``card.cpp``, QUERY_POSITION
  branch), so a card that moved always arrives complete -- merging by slot with
  an ``info_location`` check is enough, and that is what :meth:`_merge` does.

Not reconstructible from the stream, and tracked separately here:

* the *contents* of either deck.  Ours is known from the list we submitted
  minus everything that has been seen elsewhere (the "remaining multiset,
  unknown order" state the project already settled on for observations); the
  opponent's is never known, only its size.
* zones the host does not refresh (``LOCATION_DECK``), whose size is tracked
  from ``MSG_START`` plus the moves that cross the deck boundary.
* the **graveyard, the banished pile and the extra deck**, maintained from
  ``MSG_MOVE`` with the reference client's own semantics (``client_field.cpp``
  ``AddCard`` / ``RemoveCard`` / ``ResetSequence``): list zones erase and
  renumber, field zones are fixed slots, and a card returning face-down to the
  extra deck goes in front of the face-up pendulum block at its end.  The
  opponent's extra deck is a *count* rather than a zone, because
  ``RefreshExtra`` is sent to its owner alone.
* the **hand after a draw**.  ``MSG_DRAW`` is not followed by a hand refresh,
  so a hand rebuilt from refreshes alone is a card short until some unrelated
  message happens to trigger one.
* the **position of a face-down card**.  The host blanks the whole query
  segment of a face-down card -- position included -- so the face-down
  *flavour* survives only in ``MSG_MOVE`` and ``MSG_POS_CHANGE``.
* **xyz materials, counters, equip links and persistent card targets.**  The
  host's refresh flags are
  ``0x881fff`` for the monster zone and ``0x681fff`` for the rest
  (``single_duel.h:37-42``); neither contains ``QUERY_OVERLAY_CARD``
  (0x10000) or ``QUERY_COUNTERS`` (0x20000), so a query segment never carries
  either.  They come from the message stream instead -- ``MSG_MOVE`` in and
  out of ``LOCATION_OVERLAY`` for materials, ``MSG_ADD_COUNTER`` /
  ``MSG_REMOVE_COUNTER`` for counters, ``MSG_EQUIP`` for equip links, and
  ``MSG_CARD_TARGET`` / ``MSG_CANCEL_TARGET`` for effect-target links --
  exactly as the reference GUI client does it (``duelclient.cpp:2750-2845``
  and ``:3436-3480``).  The standard protocol has no active ``MSG_UNEQUIP``;
  movement retirement is exact, while a relation-only unequip cannot be
  reconstructed until the server exports it.

  Materials are load-bearing for observations, not decoration:
  ``get_cards_in_location`` emits one row per material *before* the row of the
  monster carrying them (``ygopro.h``, the ``n_xyz`` loop), so losing them
  shifts every later row and silently corrupts the card index every action
  refers to.

  Both are kept in board-level dicts keyed by ``(controller, location,
  sequence)`` rather than on :class:`ShadowCard`.  A refresh rebuilds the card
  objects of a zone whenever ``info_location`` changes, so anything held on
  the object itself would be dropped by a move that ``MSG_UPDATE_DATA``
  reports; a slot key survives that, and is reconciled against the refreshed
  zone afterwards (:meth:`_reconcile_slots`), which makes the state
  self-healing rather than merely hopeful.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field

from . import constants as C
from .disclosure import DisclosureLedger

__all__ = ["ShadowCard", "ShadowBoard", "parse_query_segments"]

# QUERY_* bits in the order card::get_infos writes them, with the number of
# uint32 words each contributes.  Variable-length ones are handled explicitly.
_SCALAR_FIELDS = (
    (C.QUERY_CODE, "code"),
    (C.QUERY_POSITION, "info_location"),
    (C.QUERY_ALIAS, "alias"),
    (C.QUERY_TYPE, "type"),
    (C.QUERY_LEVEL, "level"),
    (C.QUERY_RANK, "rank"),
    (C.QUERY_ATTRIBUTE, "attribute"),
    (C.QUERY_RACE, "race"),
    (C.QUERY_ATTACK, "attack"),
    (C.QUERY_DEFENSE, "defense"),
    (C.QUERY_BASE_ATTACK, "base_attack"),
    (C.QUERY_BASE_DEFENSE, "base_defense"),
    (C.QUERY_REASON, "reason"),
    (C.QUERY_REASON_CARD, "reason_card"),
    (C.QUERY_EQUIP_CARD, "equip_card"),
)

_SIGNED = {"attack", "defense", "base_attack", "base_defense"}


def parse_query_segments(data: bytes) -> list[dict | None]:
    """Split one query buffer into per-slot field dicts (``None`` = empty)."""
    out: list[dict | None] = []
    pos = 0
    n = len(data)
    while pos + 4 <= n:
        (seg_len,) = struct.unpack_from("<i", data, pos)
        if seg_len < 4 or pos + seg_len > n:
            break
        if seg_len <= C.LEN_EMPTY:
            out.append(None)
            pos += seg_len
            continue
        (flags,) = struct.unpack_from("<I", data, pos + 4)
        if seg_len == C.LEN_HEADER:
            # header only: every requested field was cache-suppressed
            out.append({"_flags": 0, "_nochange": True})
            pos += seg_len
            continue
        if flags == 0:
            # the host memsets the whole segment body for a card we are not
            # allowed to see (single_duel.cpp RefreshMzone / RefreshHand), so
            # this is "an unknown card occupies this slot", not "no change".
            out.append({"_flags": 0, "_hidden": True})
            pos += seg_len
            continue
        p = pos + 8
        fields: dict = {"_flags": flags}
        for bit, name in _SCALAR_FIELDS:
            if not (flags & bit):
                continue
            fmt = "<i" if name in _SIGNED else "<I"
            (value,) = struct.unpack_from(fmt, data, p)
            fields[name] = value
            p += 4
        if flags & C.QUERY_TARGET_CARD:
            (count,) = struct.unpack_from("<i", data, p)
            p += 4
            fields["targets"] = list(struct.unpack_from(f"<{count}I", data, p))
            p += 4 * count
        if flags & C.QUERY_OVERLAY_CARD:
            (count,) = struct.unpack_from("<i", data, p)
            p += 4
            fields["overlay"] = list(struct.unpack_from(f"<{count}I", data, p))
            p += 4 * count
        if flags & C.QUERY_COUNTERS:
            (count,) = struct.unpack_from("<i", data, p)
            p += 4
            raw = struct.unpack_from(f"<{count}I", data, p)
            fields["counters"] = {v & 0xFFFF: v >> 16 for v in raw}
            p += 4 * count
        if flags & C.QUERY_OWNER:
            (fields["owner"],) = struct.unpack_from("<i", data, p)
            p += 4
        if flags & C.QUERY_STATUS:
            (fields["status"],) = struct.unpack_from("<I", data, p)
            p += 4
        if flags & C.QUERY_LSCALE:
            (fields["lscale"],) = struct.unpack_from("<I", data, p)
            p += 4
        if flags & C.QUERY_RSCALE:
            (fields["rscale"],) = struct.unpack_from("<I", data, p)
            p += 4
        if flags & C.QUERY_LINK:
            fields["link"], fields["link_marker"] = struct.unpack_from("<II", data, p)
            p += 8
        out.append(fields)
        pos += seg_len
    return out


@dataclass
class ShadowCard:
    """What the stream lets us know about one card."""

    code: int = 0
    alias: int = 0
    type: int = 0
    level: int = 0
    rank: int = 0
    attribute: int = 0
    race: int = 0
    attack: int = 0
    defense: int = 0
    # The host sends QUERY_BASE_ATTACK / QUERY_BASE_DEFENSE and
    # ``parse_query_segments`` already decodes them; they are the engine's
    # runtime base stats, which are not the printed ones -- the cdb stores -2
    # for a "?" ATK and a Link monster's markers in its def column, and a card
    # off the field reports zero.  The corpus captures the engine's values, so
    # a bridge that substitutes the database disagrees with training on every
    # CARD token of those cards.
    base_attack: int = 0
    base_defense: int = 0
    lscale: int = 0
    rscale: int = 0
    link: int = 0
    link_marker: int = 0
    status: int = 0
    controller: int = 0
    location: int = 0
    sequence: int = 0
    position: int = 0
    overlay: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    equip_card: int = 0
    targets: list = field(default_factory=list)
    # QUERY_OWNER is absent from the stock host refresh masks.  We still keep
    # it here because a client can infer ownership from the card's first public
    # zone and then preserve it across control changes.
    owner: int = -1
    info_location: int = 0
    hidden: bool = False  # the host blanked the whole segment for us
    #: True once a query segment has filled this card's stats in.  A card the
    #: stream only ever *named* -- one just drawn, or one followed through
    #: MSG_MOVE into a zone the host does not refresh -- has zeroes where the
    #: engine would report the card's printed values, so the encoder has to
    #: know to read the database instead of trusting the zeroes.
    queried: bool = False

    @property
    def known(self) -> bool:
        """False for an opponent card whose identity the host withheld."""
        return not self.hidden and self.code != 0

    @property
    def face_down(self) -> bool:
        return bool(self.position & C.POS_FACEDOWN)

    def apply(self, fields: dict) -> None:
        self.queried = True
        if fields.get("code") == 0 and "code" in fields:
            # QUERY_CODE with code 0 is the other "you may not see this" shape
            # (MSG_UPDATE_CARD's censored 16-byte segment)
            self.hidden = True
        for name in (
            "code",
            "alias",
            "type",
            "level",
            "rank",
            "attribute",
            "race",
            "attack",
            "defense",
            "base_attack",
            "base_defense",
            "lscale",
            "rscale",
            "link",
            "link_marker",
            "status",
            "equip_card",
            "owner",
        ):
            if name in fields:
                setattr(self, name, fields[name])
        if "overlay" in fields:
            self.overlay = fields["overlay"]
        if "counters" in fields:
            self.counters = fields["counters"]
        if "targets" in fields:
            self.targets = fields["targets"]
        if "info_location" in fields:
            loc = fields["info_location"]
            self.info_location = loc
            self.controller = loc & 0xFF
            self.location = (loc >> 8) & 0xFF
            self.sequence = (loc >> 16) & 0xFF
            # EFFECT_REVEAL_ONFIELD can OR POS_REVEAL into the position byte
            self.position = ((loc >> 24) & 0xFF) & 0x0F
            if self.owner not in (0, 1):
                self.owner = self.controller


#: fixed-slot zones: a move writes the slot, it does not shift the list
_SLOT_ZONES = frozenset({C.LOCATION_MZONE, C.LOCATION_SZONE})

#: list zones: a move appends or erases and everything after it renumbers
_LIST_ZONES = frozenset(
    {C.LOCATION_HAND, C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA}
)

#: the three messages that publish a card's identity by showing it
_CONFIRM_MSGS = (C.MSG_CONFIRM_CARDS, C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_EXTRATOP)

_REFRESHED = (
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_HAND,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
)


class ShadowBoard:
    """Duel state maintained from ``STOC_GAME_MSG`` only."""

    def __init__(self) -> None:
        self.our_player = 0
        self.turn_player = 0
        self.phase = 0
        self.zones: dict[tuple[int, int], list] = {
            (p, loc): [] for p in (0, 1) for loc in _REFRESHED
        }
        self.deck_count = [0, 0]
        self.extra_count = [0, 0]
        # our own deck: the submitted list minus everything seen elsewhere
        self.our_deck_start: Counter = Counter()
        self.seen_out_of_deck: Counter = Counter()
        self.updates = 0
        self.unknown_messages: Counter = Counter()
        # xyz materials, counters and face-down positions, keyed by
        # (controller, location, sequence); see the module docstring for why
        # they do not live on ShadowCard
        self.materials: dict[tuple[int, int, int], list] = {}
        self.counters: dict[tuple[int, int, int], dict] = {}
        self.positions: dict[tuple[int, int, int], int] = {}
        # Public relation state reconstructed from MSG_EQUIP and
        # MSG_CARD_TARGET/CANCEL_TARGET.  The stock host refresh masks omit
        # QUERY_EQUIP_CARD/TARGET_CARD, so the wire stream is the only online
        # source.  Forward edges are sufficient; core's reverse sets are
        # exact inverses.
        self.equip_targets: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        self.card_targets: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
        #: Cards whose identity the duel has made public, by instance
        #: 公开身份账本。**与语料侧 (``worldmodel/engine.py``) 共用同一个
        #: ``DisclosureLedger`` 实现**：可见性谓词是共享的，两边各写一份就会
        #: 悄悄漂移。按 ``(控者, 区, 卡号)`` 的多重集张数记，不按实例坐标——
        #: ``field::remove_card`` 会静默重排区序且不发 ``MSG_MOVE``，坐标键
        #: 在对手打出任意一张手牌之后就对不上号，把公开过的卡重新匿名化。
        #: 也不按卡号记：那会把卡组里的同名件一并暴露。
        self.disclosure = DisclosureLedger()
        #: face-up pendulum cards sit at the end of the extra deck and the
        #: insertion point of a face-down one depends on how many there are
        self.extra_faceup = [0, 0]
        # cross_check bookkeeping
        self.checks = 0
        self.mismatches = 0
        self.learned = 0
        self.slot_problems: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self, our_player: int, main: list[int], extra: list[int]) -> None:
        self.our_player = our_player
        self.our_deck_start = Counter(main)
        for key in self.zones:
            self.zones[key] = []
        self.materials.clear()
        self.counters.clear()
        self.positions.clear()
        self.equip_targets.clear()
        self.card_targets.clear()
        self.disclosure.clear()
        self.extra_faceup = [0, 0]

    def zone(self, player: int, location: int) -> list:
        return self.zones.get((player, location), [])

    def our_remaining_deck(self) -> Counter:
        """Multiset of cards still in our deck; order is genuinely unknown."""
        return self.our_deck_start - self.seen_out_of_deck

    # -- message application ----------------------------------------------

    def apply(self, msg: int, body: bytes) -> None:
        if msg == C.MSG_UPDATE_DATA:
            self._update_data(body)
        elif msg == C.MSG_UPDATE_CARD:
            self._update_card(body)
        elif msg == C.MSG_START:
            self._start_msg(body)
        elif msg == C.MSG_DRAW:
            self._draw(body)
        elif msg == C.MSG_MOVE:
            # 先记账再改盘面：账本只吃报文，与盘面重建互不依赖。
            self.disclosure.observe_move_message(body)
            self._move(body)
        elif msg in (C.MSG_SHUFFLE_DECK, C.MSG_SHUFFLE_EXTRA,
                     C.MSG_SWAP_GRAVE_DECK):
            self.disclosure.observe_shuffle(msg, body)
        elif msg == C.MSG_ADD_COUNTER:
            self._counter(body, add=True)
        elif msg == C.MSG_REMOVE_COUNTER:
            self._counter(body, add=False)
        elif msg == C.MSG_SWAP:
            self._swap(body)
        elif msg == C.MSG_POS_CHANGE:
            self._pos_change(body)
        elif msg == C.MSG_EQUIP:
            self._equip(body)
        elif msg == C.MSG_UNEQUIP:
            self._unequip(body)
        elif msg == C.MSG_CARD_TARGET:
            self._card_target(body, add=True)
        elif msg == C.MSG_CANCEL_TARGET:
            self._card_target(body, add=False)
        elif msg == C.MSG_CHAINING:
            self.disclosure.observe_chaining(body)
        elif msg in _CONFIRM_MSGS:
            self.disclosure.observe_confirm(body, msg)
        elif msg == C.MSG_NEW_PHASE:
            self.phase = struct.unpack_from("<H", body)[0]
        elif msg == C.MSG_NEW_TURN:
            self.turn_player = body[0]
        else:
            self.unknown_messages[msg] += 1

    def _start_msg(self, body: bytes) -> None:
        # playertype, duel_rule, lp0, lp1, deck0, extra0, deck1, extra1
        if len(body) < 18:
            return
        d0, e0, d1, e1 = struct.unpack_from("<HHHH", body, 10)
        self.deck_count = [d0, d1]
        self.extra_count = [e0, e1]

    def _update_data(self, body: bytes) -> None:
        if len(body) < 2:
            return
        player, location = body[0], body[1]
        segments = parse_query_segments(body[2:])
        key = (player, location & 0x7F)
        if key not in self.zones:
            return
        self._merge(key, segments)
        self._reconcile_slots(key)
        self.updates += 1

    def _update_card(self, body: bytes) -> None:
        if len(body) < 3:
            return
        player, location, sequence = body[0], body[1], body[2]
        segments = parse_query_segments(body[3:])
        if not segments or segments[0] is None:
            return
        fields = segments[0]
        if fields.get("_nochange"):
            return
        cards = self.zones.get((player, location & 0x7F))
        if cards is None:
            return
        while len(cards) <= sequence:
            cards.append(ShadowCard())
        card = cards[sequence]
        if card is None:
            card = cards[sequence] = ShadowCard()
        if fields.get("_hidden"):
            previous = card
            cards[sequence] = ShadowCard()
            cards[sequence].hidden = True
            if previous is not None:
                cards[sequence].controller = previous.controller
                cards[sequence].location = previous.location
                cards[sequence].sequence = previous.sequence
                cards[sequence].position = previous.position
                cards[sequence].owner = previous.owner
            return
        card.hidden = False
        card.apply(fields)

    def _merge(self, key: tuple[int, int], segments: list[dict | None]) -> None:
        old = self.zones[key]
        new: list = []
        for i, fields in enumerate(segments):
            if fields is None:
                new.append(None)
                continue
            prev = old[i] if i < len(old) else None
            if fields.get("_nochange"):
                new.append(prev if prev is not None else ShadowCard())
                continue
            if fields.get("_hidden"):
                # keep the slot occupied but drop whatever we used to know:
                # the reference client calls ClearData() here, and carrying
                # stale attack/type across a face-down flip is exactly the bug
                # WindBot has (client_card.cpp:51-54).
                card = ShadowCard()
                card.hidden = True
                if prev is not None:
                    card.location = prev.location
                    card.controller = prev.controller
                    card.sequence = prev.sequence
                    card.position = prev.position
                    card.owner = prev.owner
                new.append(card)
                continue
            loc = fields.get("info_location")
            if prev is not None and (loc is None or prev.info_location == loc):
                # a cached delta for a card that did not move: merge in place.
                # QUERY_CODE and QUERY_POSITION are never cache-suppressed, so
                # a missing info_location means "nothing changed at all".
                prev.hidden = False
                prev.apply(fields)
                new.append(prev)
            else:
                card = ShadowCard()
                card.apply(fields)
                new.append(card)
        self.zones[key] = new

    def _draw(self, body: bytes) -> None:
        """A draw moves cards into the hand, and nothing refreshes afterwards.

        ``MSG_DRAW`` is not followed by ``RefreshHand`` (the host refreshes the
        hand from the main message block and from ``MSG_SHUFFLE_HAND``, not
        from the draw), so a hand rebuilt only from refreshes is one card short
        for as long as it takes some other message to trigger one.  The
        reference client adds the cards here too (``duelclient.cpp``
        ``MSG_DRAW`` -> ``AddCard(..., LOCATION_HAND, ...)``).
        """
        if len(body) < 2:
            return
        player, count = body[0], body[1]
        self.deck_count[player] = max(0, self.deck_count[player] - count)
        hand = self.zones.setdefault((player, C.LOCATION_HAND), [])
        for i in range(count):
            off = 2 + 4 * i
            code = 0
            if off + 4 <= len(body):
                (code,) = struct.unpack_from("<I", body, off)
                code &= 0x7FFFFFFF
            card = ShadowCard()
            card.code = code
            card.hidden = code == 0
            card.controller = player
            card.owner = player
            card.location = C.LOCATION_HAND
            card.position = C.POS_FACEDOWN_DEFENSE
            hand.append(card)
            if code and player == self.our_player:
                self.seen_out_of_deck[code] += 1
        self._resequence(player, C.LOCATION_HAND)

    def _move(self, body: bytes) -> None:
        if len(body) < 16:
            return
        code, previous, current, _reason = struct.unpack_from("<IIII", body)
        prev_ctl, prev_loc, prev_seq, prev_pos = (
            previous & 0xFF,
            (previous >> 8) & 0xFF,
            (previous >> 16) & 0xFF,
            (previous >> 24) & 0xFF,
        )
        cur_ctl, cur_loc, cur_seq, cur_pos = (
            current & 0xFF,
            (current >> 8) & 0xFF,
            (current >> 16) & 0xFF,
            (current >> 24) & 0xFF,
        )
        base_prev = prev_loc & 0x7F
        base_cur = cur_loc & 0x7F
        # the four overlay cases of duelclient.cpp:2687-2845
        was_overlay = bool(prev_loc & C.LOCATION_OVERLAY)
        is_overlay = bool(cur_loc & C.LOCATION_OVERLAY)
        src = (prev_ctl, base_prev, prev_seq)
        dst = (cur_ctl, base_cur, cur_seq)

        # Zone counting has to look at the overlay bit, not at the location
        # underneath it.  An xyz summon attaches its materials while the xyz
        # monster is still in the *extra deck*, so the core reports the
        # material's destination as LOCATION_OVERLAY | LOCATION_EXTRA (0xc0):
        # masking the overlay bit off first reads that as "a card entered the
        # extra deck" and inflates the count by one per material.
        left = base_prev if not was_overlay else None
        entered = base_cur if not is_overlay else None
        if left == C.LOCATION_DECK and entered != C.LOCATION_DECK:
            self.deck_count[prev_ctl] = max(0, self.deck_count[prev_ctl] - 1)
            if prev_ctl == self.our_player and code:
                self.seen_out_of_deck[code] += 1
        elif entered == C.LOCATION_DECK and left != C.LOCATION_DECK:
            self.deck_count[cur_ctl] += 1
            if cur_ctl == self.our_player and code:
                self.seen_out_of_deck[code] -= 1
        # RefreshExtra goes to the owner only (single_duel.cpp:1580), so the
        # opponent's extra deck is a count we maintain, not a zone we see
        if left == C.LOCATION_EXTRA and entered != C.LOCATION_EXTRA:
            self.extra_count[prev_ctl] = max(0, self.extra_count[prev_ctl] - 1)
        elif entered == C.LOCATION_EXTRA and left != C.LOCATION_EXTRA:
            self.extra_count[cur_ctl] += 1

        if not was_overlay and not is_overlay:
            self._move_relations(src, dst, base_prev, base_cur)
            self._relocate(code, src, dst, cur_pos)
            # The host blanks a face-down card's whole query segment
            # (``single_duel.cpp`` RefreshMzone / RefreshSzone), position
            # included, but never blanks the position byte of MSG_MOVE, so the
            # stream is the only place the face-down *flavour* survives.
            self.positions.pop(src, None)
            if base_cur & C.LOCATION_ONFIELD:
                self.positions[dst] = cur_pos & 0x0F
            # A card that left the field drops its counters; one that only
            # changed zone inside the same location keeps them, which is why
            # the reference compares the raw location bytes and not the slot.
            if (prev_loc & C.LOCATION_ONFIELD) and cur_loc != prev_loc:
                self.counters.pop(src, None)
            else:
                self._rekey(self.counters, src, dst)
            # Materials follow their host wherever it goes, and the core never
            # says so: the reference client hangs them off the card object
            # (``duelclient.cpp``: ``olcard->overlayed``), which moves for free.
            # The load-bearing case is the xyz summon, where the materials are
            # attached in the extra deck and the monster then walks to the
            # monster zone under its own MSG_MOVE -- with no message per
            # material.  A stack left behind at the old slot would be dropped
            # by the next refresh and the monster would arrive bare.
            #
            # Renumbering after the move is not cosmetic: each material carries
            # the host's location and sequence in its own fields, and those are
            # what the observation encoder reads.  Moving the stack without
            # restating them leaves every material claiming to still sit on a
            # card in the extra deck.
            if src in self.materials:
                self._rekey(self.materials, src, dst)
                self._renumber(dst)
        elif not was_overlay and is_overlay:
            self._move_relations(src, None, base_prev, 0)
            self.counters.pop(src, None)
            self._rekey(self.materials, src, None)
            # take the card out of the zone it came from, so a material pulled
            # from the graveyard or the banished pile does not stay there too
            card = self._take_from(src)
            if card is None:
                card = ShadowCard()
            if code:
                card.code = code & 0x7FFFFFFF
                card.hidden = False
            card.controller = cur_ctl
            if card.owner not in (0, 1):
                card.owner = prev_ctl if base_prev else cur_ctl
            card.location = base_cur
            card.sequence = cur_seq
            self.materials.setdefault(dst, []).append(card)
            self._renumber(dst)
        elif was_overlay and not is_overlay:
            # A detached material lands somewhere real -- usually the
            # graveyard, which nothing refreshes.  Dropping it here instead of
            # filing it would leave that zone one card short and renumber every
            # card behind it for the rest of the duel.
            card = self._detach(src, prev_pos)
            if card is None:
                card = ShadowCard()
            if code:
                card.code = code & 0x7FFFFFFF
                card.hidden = False
            card.status = 0
            card.controller = cur_ctl
            if card.owner not in (0, 1):
                card.owner = cur_ctl
            card.location = base_cur
            card.position = cur_pos & 0x0F
            self._put_into(card, dst, cur_pos)
            if base_cur & C.LOCATION_ONFIELD:
                self.positions[dst] = cur_pos & 0x0F
        else:
            card = self._detach(src, prev_pos)
            if card is not None:
                card.controller = cur_ctl
                card.location = base_cur
                card.sequence = cur_seq
                self.materials.setdefault(dst, []).append(card)
                self._renumber(dst)

    def _relocate(self, code, src, dst, cur_pos) -> None:
        """Move one card between zones, the way the reference client does.

        The module docstring's "the host re-queries a zone after every board
        change" holds for the monster zone, the spell/trap zone and the hand,
        and for nothing else: ``RefreshGrave`` is reachable only from
        ``MSG_SWAP_GRAVE_DECK`` and ``RefreshExtra`` only from
        ``MSG_SHUFFLE_EXTRA`` (``single_duel.cpp:882`` and ``:871``), and the
        banished zone has no refresh at all.  So for the graveyard, the
        banished pile and the extra deck the message stream is not a
        cross-check on the refresh -- it is the *only* source, and without it
        those zones drift the moment a card leaves one of them and every later
        card's sequence is off by one.

        Membership and order follow ``ClientField::AddCard`` /
        ``RemoveCard`` (``client_field.cpp:157`` and ``:216``): list zones
        erase-and-renumber, field zones are fixed slots, and a card returning
        face-down to the extra deck goes in front of the face-up pendulum
        block that lives at its end.  The card object itself travels, so what
        we already knew about it is not thrown away on the way.
        """
        card = self._take_from(src)
        if card is None:
            card = ShadowCard()
        if code:
            card.code = code & 0x7FFFFFFF
            card.hidden = False
        elif dst[1] == C.LOCATION_EXTRA:
            # the reference resets the code here even when it is 0, because a
            # card going back to the extra deck face-down is not identifiable
            card.code = 0
            card.hidden = True
        if card.owner not in (0, 1):
            card.owner = src[0] if src[1] else dst[0]
        if dst[1] != src[1] or not (dst[1] & C.LOCATION_ONFIELD):
            # A card that left the field is no longer negated or forbidden.
            # Only the field zones are refreshed, so a status that went stale
            # in the graveyard would never be corrected and would keep
            # reporting a card as disabled for the rest of the duel.
            card.status = 0
        card.controller = dst[0]
        card.location = dst[1]
        card.position = cur_pos & 0x0F
        self._put_into(card, dst, cur_pos)

    def _take_from(self, src: tuple[int, int, int]):
        player, location, sequence = src
        if location in _SLOT_ZONES:
            zone = self.zones.get((player, location))
            if zone is None or sequence >= len(zone):
                return None
            card, zone[sequence] = zone[sequence], None
            return card
        if not self._tracks_list(player, location):
            return None
        zone = self.zones.get((player, location))
        if zone is None or sequence >= len(zone):
            return None
        card = zone.pop(sequence)
        if location == C.LOCATION_EXTRA and card is not None:
            if card.position & C.POS_FACEUP:
                self.extra_faceup[player] = max(0, self.extra_faceup[player] - 1)
        self._resequence(player, location)
        return card

    def _put_into(self, card, dst: tuple[int, int, int], cur_pos: int) -> None:
        player, location, sequence = dst
        if location in _SLOT_ZONES:
            zone = self.zones.setdefault((player, location), [])
            while len(zone) <= sequence:
                zone.append(None)
            zone[sequence] = card
            card.sequence = sequence
            return
        if not self._tracks_list(player, location):
            return
        zone = self.zones.setdefault((player, location), [])
        if location == C.LOCATION_EXTRA:
            faceup = self.extra_faceup[player]
            if faceup == 0 or (cur_pos & C.POS_FACEUP):
                zone.append(card)
            else:
                zone.insert(max(0, len(zone) - faceup), card)
            if cur_pos & C.POS_FACEUP:
                self.extra_faceup[player] += 1
        else:
            zone.append(card)
        self._resequence(player, location)

    def _tracks_list(self, player: int, location: int) -> bool:
        """Whether we hold the contents of this list zone at all.

        ``RefreshExtra`` is sent to the owner only, so the opponent's extra
        deck was never populated and must not be maintained: removing from an
        empty list would silently eat somebody else's card.  Its size is kept
        in ``extra_count`` instead.
        """
        if location not in _LIST_ZONES:
            return False
        if location == C.LOCATION_EXTRA and player != self.our_player:
            return False
        return True

    def _resequence(self, player: int, location: int) -> None:
        for i, card in enumerate(self.zones.get((player, location), ())):
            if card is not None:
                card.sequence = i


    def _pos_change(self, body: bytes) -> None:
        """``MSG_POS_CHANGE``: the only announcement of a flip we ever get."""
        if len(body) < 9:
            return
        player, location, sequence = body[4], body[5], body[6]
        new_pos = body[8] & 0x0F
        key = (player, location & 0x7F, sequence)
        if location & C.LOCATION_ONFIELD:
            self.positions[key] = new_pos
        zone = self.zones.get((player, location & 0x7F))
        if zone is not None and sequence < len(zone) and zone[sequence] is not None:
            zone[sequence].position = new_pos

    @staticmethod
    def _at_key(at: int) -> tuple[int, int, int]:
        return (at & 0xFF, ((at >> 8) & 0xFF) & 0x7F, (at >> 16) & 0xFF)

    def _equip(self, body: bytes) -> None:
        """``MSG_EQUIP`` publishes the directed equip-card -> host edge."""
        if len(body) < 8:
            return
        source, target = struct.unpack_from("<II", body, 0)
        self.equip_targets[self._at_key(source)] = self._at_key(target)

    def _unequip(self, body: bytes) -> None:
        # MSG_UNEQUIP is reserved/commented in this core, but accepting the
        # legacy four-byte shape costs nothing and keeps compatible servers exact.
        if len(body) >= 4:
            (source,) = struct.unpack_from("<I", body, 0)
            self.equip_targets.pop(self._at_key(source), None)

    def _card_target(self, body: bytes, add: bool) -> None:
        """Maintain the persistent SetCardTarget source -> target relation."""
        if len(body) < 8:
            return
        source_at, target_at = struct.unpack_from("<II", body, 0)
        source, target = self._at_key(source_at), self._at_key(target_at)
        if add:
            self.card_targets.setdefault(source, set()).add(target)
            return
        targets = self.card_targets.get(source)
        if targets is None:
            return
        targets.discard(target)
        if not targets:
            self.card_targets.pop(source, None)

    def _drop_relations_at(self, key: tuple[int, int, int]) -> None:
        self.equip_targets.pop(key, None)
        self.card_targets.pop(key, None)
        self.equip_targets = {
            source: target for source, target in self.equip_targets.items()
            if target != key
        }
        for source, targets in list(self.card_targets.items()):
            targets.discard(key)
            if not targets:
                self.card_targets.pop(source, None)

    def _move_relations(self, src, dst, prev_loc: int, cur_loc: int) -> None:
        """Move an on-field endpoint, or retire every edge when it leaves.

        Core clears equip/target relations on the reset paths that leave the
        field.  A control/zone move that stays on field can keep an endpoint,
        so re-key both forward sources and forward targets.
        """
        if dst is None or not (prev_loc & C.LOCATION_ONFIELD) \
                or not (cur_loc & C.LOCATION_ONFIELD):
            self._drop_relations_at(src)
            return
        if src == dst:
            return
        if src in self.equip_targets:
            self.equip_targets[dst] = self.equip_targets.pop(src)
        if src in self.card_targets:
            self.card_targets[dst] = self.card_targets.pop(src)
        self.equip_targets = {
            source: (dst if target == src else target)
            for source, target in self.equip_targets.items()
        }
        for targets in self.card_targets.values():
            if src in targets:
                targets.remove(src)
                targets.add(dst)

    def _swap_relation_endpoints(self, a, b) -> None:
        def moved(key):
            return b if key == a else (a if key == b else key)

        self.equip_targets = {
            moved(source): moved(target)
            for source, target in self.equip_targets.items()
        }
        self.card_targets = {
            moved(source): {moved(target) for target in targets}
            for source, targets in self.card_targets.items()
        }

    def _detach(self, src: tuple[int, int, int], index: int):
        """Take material ``index`` off the monster at ``src``.

        ``index`` is the *previous position* byte of ``MSG_MOVE``, which the
        core reuses as the material index for a card leaving an overlay
        (``duelclient.cpp:2795``).
        """
        stack = self.materials.get(src)
        if not stack or index >= len(stack):
            self.slot_problems.append(
                f"detach material {index} of {src} but the stack holds "
                f"{0 if not stack else len(stack)}"
            )
            return None
        card = stack.pop(index)
        if stack:
            self._renumber(src)
        else:
            self.materials.pop(src, None)
        return card

    def _renumber(self, key: tuple[int, int, int]) -> None:
        """Restate each material the way ``get_cards_in_location`` would.

        The engine hands a material over with the host's location OR-ed with
        ``LOCATION_OVERLAY``, the host's sequence, and the material's index in
        the stack as its position -- which is also what turns into the ``a`` /
        ``b`` / ``c`` suffix of its spec.
        """
        for i, card in enumerate(self.materials.get(key, ())):
            card.position = i
            card.controller = key[0]
            card.location = key[1] | C.LOCATION_OVERLAY
            card.sequence = key[2]

    @staticmethod
    def _rekey(table: dict, src, dst) -> None:
        value = table.pop(src, None)
        if value is None or dst is None:
            return
        table[dst] = value

    def _counter(self, body: bytes, add: bool) -> None:
        if len(body) < 7:
            return
        ctype = struct.unpack_from("<H", body)[0]
        player, location, sequence = body[2], body[3], body[4]
        count = struct.unpack_from("<H", body, 5)[0]
        key = (player, location & 0x7F, sequence)
        slot = self.counters.setdefault(key, {})
        if add:
            slot[ctype] = slot.get(ctype, 0) + count
        else:
            slot[ctype] = slot.get(ctype, 0) - count
            if slot[ctype] <= 0:
                slot.pop(ctype, None)
        if not slot:
            self.counters.pop(key, None)

    def _swap(self, body: bytes) -> None:
        """Two monsters trade zones; their materials and counters go with them."""
        if len(body) < 16:
            return
        # code(4) controller location sequence position, twice
        a = (body[4], body[5] & 0x7F, body[6])
        b = (body[12], body[13] & 0x7F, body[14])
        self._swap_relation_endpoints(a, b)
        for table in (self.materials, self.counters):
            va, vb = table.pop(a, None), table.pop(b, None)
            if va is not None:
                table[b] = va
            if vb is not None:
                table[a] = vb
        self._renumber(a)
        self._renumber(b)

    def _reconcile_slots(self, key: tuple[int, int]) -> None:
        """Materials and counters may only sit under an occupied slot.

        The refresh is the engine's own view of which slots hold a card, so
        anything left over after it is state we failed to retire -- dropping it
        here keeps a missed message from compounding across a whole duel.
        """
        player, location = key
        occupied = {
            i for i, card in enumerate(self.zones.get(key, ())) if card is not None
        }
        for table in (self.materials, self.counters, self.positions):
            stale = [
                k
                for k in table
                if k[0] == player and k[1] == location and k[2] not in occupied
            ]
            for k in stale:
                table.pop(k, None)
        stale_rel = {
            k for k in set(self.equip_targets) | set(self.card_targets)
            if k[0] == player and k[1] == location and k[2] not in occupied
        }
        stale_rel.update(
            target for target in self.equip_targets.values()
            if target[0] == player and target[1] == location
            and target[2] not in occupied
        )
        stale_rel.update(
            target for targets in self.card_targets.values() for target in targets
            if target[0] == player and target[1] == location
            and target[2] not in occupied
        )
        for key_ in stale_rel:
            self._drop_relations_at(key_)

    # -- observation support -----------------------------------------------

    def cards_in_location(self, player: int, location: int) -> list:
        """``ygopro.h``'s ``get_cards_in_location`` order, minus the engine.

        Each xyz material is emitted *before* the monster that carries it, with
        ``location`` OR-ed with ``LOCATION_OVERLAY``, the monster's sequence and
        its own index as the position -- the exact rows the observation encoder
        expects.  Empty slots contribute nothing, as in the engine, whose query
        buffer skips them.
        """
        out: list = []
        for seq, card in enumerate(self.zone(player, location)):
            if card is None:
                continue
            for material in self.materials.get((player, location, seq), ()):
                out.append(material)
            if card.hidden and not card.position:
                # the refresh blanked the segment; the position we followed
                # through MSG_MOVE / MSG_POS_CHANGE is the only copy left
                card.position = self.positions.get((player, location, seq), 0)
            out.append(card)
        return out

    def counters_of(self, player: int, location: int, sequence: int) -> dict:
        return self.counters.get((player, location & 0x7F, sequence), {})

    @staticmethod
    def _pack_key(key: tuple[int, int, int]) -> int:
        return key[0] | (key[1] << 8) | (key[2] << 16)

    def equip_target_of(self, player: int, location: int, sequence: int) -> int:
        target = self.equip_targets.get((player, location & 0x7F, sequence))
        return self._pack_key(target) if target is not None else 0

    def targets_of(self, player: int, location: int, sequence: int) -> tuple[int, ...]:
        targets = self.card_targets.get((player, location & 0x7F, sequence), ())
        return tuple(self._pack_key(key) for key in sorted(targets))

    # -- consistency -------------------------------------------------------

    def cross_check(self, cards: list[tuple[int, int, int, int]]) -> list[str]:
        """Compare a prompt's card list against the shadow board.

        Every ``MSG_SELECT_*`` prompt names the cards it offers as
        ``(code, controller, location, sequence)``, written by the engine from
        the real board.  Agreeing with those tuples is agreeing with the
        engine, and unlike a replay-based check it costs nothing and runs on
        every live duel.

        Only non-zero codes are checked: the host blanks the code of a card we
        are not allowed to identify.
        """
        problems: list[str] = []
        for code, controller, location, sequence in cards:
            if location & C.LOCATION_OVERLAY:
                continue  # xyz materials are not tracked, see the module doc
            loc = location & 0x7F
            if loc == C.LOCATION_DECK:
                continue  # deck contents are never sent to anyone
            zone = self.zones.get((controller, loc))
            if zone is None:
                continue
            self.checks += 1
            if sequence >= len(zone):
                problems.append(
                    f"code {code} at ({controller}, {loc}, {sequence}) but the "
                    f"shadow zone holds {len(zone)} slots"
                )
                continue
            card = zone[sequence]
            if card is None:
                problems.append(
                    f"code {code} at ({controller}, {loc}, {sequence}) but the "
                    "shadow slot is empty"
                )
            elif card.code and card.code != code:
                problems.append(
                    f"code {code} at ({controller}, {loc}, {sequence}) but the "
                    f"shadow holds {card.code}"
                )
            elif not card.code:
                # we were never told this card's identity; the prompt just did
                self.learned += 1
                card.code = code
                card.hidden = False
        self.mismatches += len(problems)
        return problems

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        def zone_codes(p, loc):
            return [c.code if c else 0 for c in self.zone(p, loc)]

        return {
            "our_player": self.our_player,
            "turn_player": self.turn_player,
            "phase": self.phase,
            "deck_count": list(self.deck_count),
            "hand": [zone_codes(0, C.LOCATION_HAND), zone_codes(1, C.LOCATION_HAND)],
            "mzone": [zone_codes(0, C.LOCATION_MZONE), zone_codes(1, C.LOCATION_MZONE)],
            "szone": [zone_codes(0, C.LOCATION_SZONE), zone_codes(1, C.LOCATION_SZONE)],
            "grave": [zone_codes(0, C.LOCATION_GRAVE), zone_codes(1, C.LOCATION_GRAVE)],
            "extra": [zone_codes(0, C.LOCATION_EXTRA), zone_codes(1, C.LOCATION_EXTRA)],
            "removed": [
                zone_codes(0, C.LOCATION_REMOVED),
                zone_codes(1, C.LOCATION_REMOVED),
            ],
            "updates": self.updates,
            "checks": self.checks,
            "mismatches": self.mismatches,
            "learned": self.learned,
        }
