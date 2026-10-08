"""Shadow board + legal actions -> the observation a PV-1 checkpoint expects.

``ygoenv`` builds its observation by *querying the engine* it owns
(``ygopro.h``: ``_set_obs_cards`` / ``_set_obs_global`` / ``_set_obs_action``).
A network client has no engine, so the same tensors have to be assembled from
:mod:`.board`, which follows the message stream.  Everything below is a
column-for-column port of those three functions; where the two sides can
disagree, the reason is written down rather than smoothed over.

The layout is fixed by the run that produced the checkpoint, and the training
log states it outright rather than leaving it to be inferred::

    obs_space=Dict('cards_': Box(0,255,(160,41)), 'global_': Box(0,255,(23,)),
                   'actions_': Box(0,255,(24,13)), 'h_actions_': Box(0,255,(32,15)),
                   'mask_': Box(0,255,(160,14)))

``mask_`` is deliberately **not** produced.  It is written only under
``oppo_info=true``; PV-1 trained with ``oppo_info=False`` and with
``EnvPreprocess(skip_mask=True)`` replacing it by ``None``
(``ygoai/rl/env.py:86``), and ``CardEncoder`` treats an all-zero mask as "every
card row is the padding row" (``agent.py:133-138``), which blanks the board.
Omitting the key reproduces training; supplying zeros would not.

Where a network client cannot match the engine
----------------------------------------------

1. **The opponent's hidden zones stay hidden.**  ``_set_obs_cards`` opens the
   opponent's whole deck, hand and extra deck whenever ``revealed_`` is
   non-empty (``ygopro.h:3279``, marked ``// check this`` in the source and
   guarded by a ``CountHiddenLeaks`` diagnostic).  That is information no
   client is sent, so those rows are always encoded in the hidden form here.
   The direction is conservative: the policy gets *less* than it did in
   training, and the hidden form is the overwhelmingly common one there.

2. **Face-down cards keep the position the stream reported.**  The host blanks
   the entire query segment of a face-down card, position included, so the
   position comes from ``MSG_MOVE`` / ``MSG_POS_CHANGE`` instead (see
   :mod:`.board`).

3. **Link rating and markers outside the monster zone.**  ``ygoenv`` queries
   every location with ``QUERY_LINK``; the host does not (``0x81fff`` for the
   graveyard, ``0x681fff`` for the hand).  A link monster's rating and markers
   are static card data, so the database value is substituted, which is the
   same number the engine would have sent.
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from . import obsconst as K
from .actions import ActionAct, LegalAction

__all__ = ["ObsEncoder", "ObsGeometry"]

#: the order ``_set_obs_cards`` walks, per player
_LOCATIONS = (
    C.LOCATION_DECK,
    C.LOCATION_HAND,
    C.LOCATION_MZONE,
    C.LOCATION_SZONE,
    C.LOCATION_GRAVE,
    C.LOCATION_REMOVED,
    C.LOCATION_EXTRA,
)

#: the opponent's zones the host never lets us see the contents of
_HIDDEN_FOR_OPPONENT = frozenset(
    {C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_EXTRA}
)

#: locations whose sequence reaches the observation (``ygopro.h:3444``)
_SEQUENCED = frozenset({C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_GRAVE})

#: the host's refresh flags carry QUERY_LINK only for these
_LINK_QUERIED = frozenset({C.LOCATION_MZONE, C.LOCATION_EXTRA})

_MULTI_SELECT = frozenset(
    {
        C.MSG_SELECT_CARD,
        C.MSG_SELECT_TRIBUTE,
        C.MSG_SELECT_SUM,
        C.MSG_SELECT_UNSELECT_CARD,
    }
)

#: messages where ygoenv sets the action's card id itself instead of letting it
#: fall out of the spec: the announced card (``ygopro.h:5153``) and the position
#: prompt, which names its card in the message and carries no spec at all
#: (``ygopro.h:6231-6240``: ``la.cid_ = c_get_card_id(code)``)
_ID_ONLY = frozenset({C.MSG_ANNOUNCE_CARD, C.MSG_SELECT_POSITION})


class ObsGeometry:
    """The four numbers the checkpoint was trained with."""

    max_cards = 80
    max_options = 24
    n_history_actions = 32
    n_action_feats = K.N_ACTION_FEATS

    @property
    def n_card_rows(self) -> int:
        return self.max_cards * 2


class _Row:
    """One card as ``get_cards_in_location`` would have handed it over."""

    __slots__ = (
        "code",
        "location",
        "sequence",
        "controller",
        "position",
        "overlay",
        "level",
        "attack",
        "defense",
        "counter",
        "negated",
        "hide",
    )

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name, 0))


class ObsEncoder:
    """Builds one observation per decision, and keeps the action history.

    One instance belongs to one seat of one duel: ``h_actions_`` is a ring
    buffer of *our own* past actions, which only makes sense per seat.
    """

    def __init__(self, card_pool, geometry: ObsGeometry | None = None):
        self.pool = card_pool
        self.geom = geometry or ObsGeometry()
        self.our_player = 0
        # ygoenv's history_actions_ ring, newest last; rendered newest-first
        self._history: list[tuple[np.ndarray, int]] = []
        #: the card ids ``_encode_actions`` resolved for the current prompt.
        #: ``update_history_actions`` runs *after* ``WriteState`` has filled
        #: ``cid_`` in from the card table, so the history keeps the resolved
        #: value; recomputing it without the table would leave it at 0.
        self._resolved_cids: list[int] = []
        self._deck_rows: dict[int, list[int]] = {}
        self.row_overflows = 0
        self.deck_disagreements = 0
        self.unknown_codes: set[int] = set()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, our_player: int, seed: int = 0) -> None:
        self.our_player = our_player
        self._history.clear()
        self._resolved_cids.clear()
        self._deck_rows.clear()

    # -- the observation ---------------------------------------------------

    def encode(
        self,
        board,
        msg: int,
        actions: list[LegalAction],
        lp: tuple[int, int],
        turn_count: int,
        phase: int,
        turn_player: int,
    ) -> dict:
        cards, spec_index, spec_cid, loc_n_cards = self._encode_cards(board)
        obs = {
            "cards_": cards,
            "global_": self._encode_global(lp, turn_count, phase, turn_player, loc_n_cards),
            "actions_": self._encode_actions(msg, actions, spec_index, spec_cid),
            "h_actions_": self._render_history(turn_count),
        }
        return obs

    def push_history(
        self,
        msg: int,
        action: LegalAction,
        turn_count: int,
        phase: int,
        index: int | None = None,
        forced: bool = False,
    ) -> None:
        """Record a taken action, as ``update_history_actions`` does.

        Cancels are dropped (``ygopro.h:2417``) and the spec column is cleared,
        because a row index only means something against the card table of the
        step it was produced in.  The card id, by contrast, is kept: it was
        already resolved for ``obs:actions_``, so ``index`` names which row of
        the prompt was taken.

        ``forced`` is the single-option branch, which the engine takes without
        ever writing an observation and which resolves the card id straight
        from the board rather than from the card table (``ygopro.h:3934``) --
        so it names even a card the table would have kept anonymous.
        """
        if action.act == ActionAct.CANCEL:
            return
        if forced:
            cid = self._action_cid(msg, action, 0) or self.pool.card_id(action.code)
        elif index is not None and index < len(self._resolved_cids):
            cid = self._resolved_cids[index]
        else:
            cid = self._action_cid(msg, action, 0)
        row = np.zeros(K.N_HISTORY_ACTION_FEATS, dtype=np.uint8)
        self._write_action(row, msg, action, spec_index=0, cid=cid)
        row[0] = 0
        row[K.N_ACTION_FEATS] = turn_count & 0xFF
        row[K.N_ACTION_FEATS + 1] = K.phase_to_id(phase) if phase else 0
        self._history.append((row, turn_count))
        if len(self._history) > self.geom.n_history_actions:
            self._history.pop(0)

    # -- cards -------------------------------------------------------------

    def _encode_cards(self, board):
        geom = self.geom
        out = np.zeros((geom.n_card_rows, 41), dtype=np.uint8)
        spec_index: dict[str, int] = {}
        spec_cid: dict[str, int] = {}
        self._deck_rows = {}
        loc_n_cards: list[int] = []
        offset = 0
        for pi in (0, 1):
            player = (self.our_player + pi) % 2
            opponent = pi == 1
            for location in _LOCATIONS:
                if opponent and location in _HIDDEN_FOR_OPPONENT:
                    n = self._hidden_count(board, player, location)
                    loc_n_cards.append(n)
                    loc_id = K.location_to_id(location)
                    for _ in range(n):
                        if offset >= geom.n_card_rows:
                            self.row_overflows += 1
                            break
                        out[offset, 2] = loc_id
                        out[offset, 4] = 1
                        offset += 1
                    continue
                rows = self._rows_for(board, player, location, opponent)
                loc_n_cards.append(len(rows))
                for row in rows:
                    if offset >= geom.n_card_rows:
                        self.row_overflows += 1
                        break
                    cid = 0 if row.hide else self.pool.card_id(row.code)
                    self._write_card(out, offset, row, cid)
                    offset += 1
                    if location == C.LOCATION_DECK:
                        # Prompt deck numbers are not shuffled-deck positions.
                        # Bind only known own-deck candidates by their card ID.
                        self._deck_rows.setdefault(cid, []).append(offset)
                        continue
                    spec = _spec_of(row, opponent)
                    # recorded after the increment, so the index is 1-based and
                    # 0 can mean "no card" in the action table (ygopro.h:3322)
                    spec_index[spec] = offset
                    spec_cid[spec] = cid
        return out, spec_index, spec_cid, loc_n_cards

    def _hidden_count(self, board, player: int, location: int) -> int:
        if location == C.LOCATION_DECK:
            return max(0, board.deck_count[player])
        if location == C.LOCATION_EXTRA:
            # RefreshExtra only ever reaches the owner, so this is a count we
            # maintained through MSG_MOVE rather than a zone we can look at
            return max(0, board.extra_count[player])
        return len(board.zone(player, location))

    def _rows_for(self, board, player: int, location: int, opponent: bool) -> list[_Row]:
        if location == C.LOCATION_DECK:
            return self._our_deck_rows(board, player)
        rows: list[_Row] = []
        for card in board.cards_in_location(player, location):
            rows.append(self._row_of(card, location, opponent))
        return rows

    def _our_deck_rows(self, board, player: int) -> list[_Row]:
        """Our remaining deck, in canonical card-id order.

        A client knows its submitted cards minus observed departures, not the
        hidden shuffle.  Sorting gives stable rows without inventing order.
        This is not a claim that the native engine uses the same row order.
        """
        remaining = board.our_remaining_deck()
        codes: list[int] = []
        for code, count in remaining.items():
            if count > 0:
                codes.extend([code] * count)
        if len(codes) != board.deck_count[player]:
            # the two are tracked independently (submitted list minus what we
            # have seen, versus MSG_START plus deck crossings); a split is a
            # tracking bug, not a rounding difference
            self.deck_disagreements += 1
        codes.sort(key=lambda c: (self.pool.card_id(c), c))
        rows = []
        for code in codes:
            data = self.pool.cards.get(code)
            rows.append(
                _Row(
                    code=code,
                    location=C.LOCATION_DECK,
                    sequence=0,
                    controller=player,
                    position=C.POS_FACEDOWN,
                    overlay=0,
                    level=data.level if data else 0,
                    attack=data.attack if data else 0,
                    defense=data.defense if data else 0,
                    counter=0,
                    negated=0,
                    hide=False,
                )
            )
        return rows

    def _row_of(self, card, location: int, opponent: bool) -> _Row:
        overlay = 1 if card.location & C.LOCATION_OVERLAY else 0
        code = card.code
        data = self.pool.cards.get(code)
        if code and data is None:
            self.unknown_codes.add(code)
        hide = False
        if opponent and not overlay:
            hide = bool(card.position & C.POS_FACEDOWN)
        if not code:
            hide = True
        if overlay:
            # a material is public even on the opponent's board (ygopro.h:3434)
            hide = False
        printed = not getattr(card, "queried", True)
        level = card.level or card.rank or (data.level if data else 0)
        attack = card.attack
        defense = card.defense
        if printed and data is not None:
            # never queried: the engine would have reported the printed values
            level, attack, defense = data.level, data.attack, data.defense
        if overlay:
            # materials are handed over straight from the card database
            level = data.level if data else 0
            attack = data.attack if data else 0
            defense = data.defense if data else 0
        else:
            if card.link & 0xFF:
                level = card.link & 0xFF
            if card.link_marker:
                defense = card.link_marker
            elif data and (data.type & C.TYPE_LINK) and location not in _LINK_QUERIED:
                # the host did not ask for QUERY_LINK here; rating and markers
                # are static card data, so the database carries the same value
                level = data.level
                defense = data.defense
        counters = 0
        if not overlay:
            slot = card.counters or {}
            if slot:
                ctype = min(slot)
                counters = ctype | (slot[ctype] << 16)
        return _Row(
            code=code,
            location=card.location,
            sequence=card.sequence,
            controller=card.controller,
            position=card.position,
            overlay=overlay,
            level=level,
            attack=attack,
            defense=defense,
            counter=counters,
            # Only a card on the field can be negated, and only the field
            # zones are refreshed.  A status byte captured on the way into the
            # graveyard (MSG_MOVE is followed by a single-card refresh, which
            # carries QUERY_STATUS) goes stale there and nothing ever corrects
            # it, so the disable bits are read only where they can still mean
            # something.
            negated=1
            if (
                not printed
                and (card.location & C.LOCATION_ONFIELD)
                and card.status & (C.STATUS_DISABLED | C.STATUS_FORBIDDEN)
            )
            else 0,
            hide=hide,
        )

    def _write_card(self, out, offset: int, row: _Row, cid: int) -> None:
        """``_set_obs_card_`` (``ygopro.h:3426``), column for column."""
        location = row.location & 0x7F
        hide = row.hide and not row.overlay
        if not hide:
            out[offset, 0] = (cid >> 8) & 0xFF
            out[offset, 1] = cid & 0xFF
        out[offset, 2] = K.location_to_id(location)
        out[offset, 3] = (row.sequence + 1) if location in _SEQUENCED else 0
        out[offset, 4] = 1 if row.controller != self.our_player else 0
        if row.overlay:
            out[offset, 5] = K.position_to_id(C.POS_FACEUP)
            out[offset, 6] = 1
        elif location in (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_EXTRA):
            if hide or (row.position & C.POS_FACEDOWN):
                out[offset, 5] = K.position_to_id(C.POS_FACEDOWN)
        else:
            out[offset, 5] = K.position_to_id(row.position & 0x0F)
        if hide:
            return
        data = self.pool.cards.get(row.code)
        out[offset, 7] = K.attribute_to_id(data.attribute if data else 0)
        out[offset, 8] = K.race_to_id(data.race if data else 0)
        out[offset, 9] = min(row.level, 255)
        out[offset, 10] = min(row.counter, 15)
        out[offset, 11] = row.negated
        out[offset, 12], out[offset, 13] = K.float_transform(row.attack)
        out[offset, 14], out[offset, 15] = K.float_transform(row.defense)
        out[offset, 16:41] = K.type_to_ids(data.type if data else 0)

    # -- global ------------------------------------------------------------

    def _encode_global(self, lp, turn_count, phase, turn_player, loc_n_cards):
        feat = np.zeros(23, dtype=np.uint8)
        me = self.our_player
        feat[0], feat[1] = K.float_transform(lp[me])
        feat[2], feat[3] = K.float_transform(lp[1 - me])
        feat[4] = min(turn_count, 16)
        feat[5] = K.phase_to_id(phase) if phase else 0
        feat[6] = 1 if me == 0 else 0
        feat[7] = 1 if me == turn_player else 0
        for i, n in enumerate(loc_n_cards[:14]):
            feat[8 + i] = min(n, 255)
        return feat

    # -- actions -----------------------------------------------------------

    def _encode_actions(self, msg, actions, spec_index, spec_cid):
        geom = self.geom
        out = np.zeros((geom.max_options, geom.n_action_feats), dtype=np.uint8)
        self._resolved_cids = []
        self._bind_deck_candidates(actions, spec_index, spec_cid)
        for i, action in enumerate(actions[: geom.max_options]):
            idx, spec_id = self._resolve_spec(action, spec_index, spec_cid)
            cid = self._action_cid(msg, action, spec_id)
            self._resolved_cids.append(cid)
            self._write_action(out[i], msg, action, idx, cid)
        return out

    def _bind_deck_candidates(self, actions, spec_index, spec_cid):
        """Bind prompt-local own-deck specs to distinct rows of the right ID."""
        candidates = {}
        for action in actions:
            if not action.spec.isdigit():
                continue
            if int(action.spec) < 1:
                raise ValueError("Invalid deck candidate spec")
            cid = self.pool.card_id(action.code)
            if action.code and not cid:
                raise ValueError("Unknown deck candidate card: " + str(action.code))
            if action.spec in candidates and candidates[action.spec] != cid:
                raise ValueError("Conflicting deck candidate identity: " + action.spec)
            candidates[action.spec] = cid
        used = {row for spec, row in spec_index.items() if spec.isdigit() and row}
        for spec, cid in sorted(candidates.items(), key=lambda item: int(item[0])):
            if spec in spec_index:
                if spec_cid[spec] != cid:
                    raise ValueError("Conflicting deck candidate identity: " + spec)
                continue
            row = 0
            if cid:
                row = next((i for i in self._deck_rows.get(cid, ()) if i not in used), 0)
                if not row:
                    raise ValueError("Deck candidate identity/count disagrees with remaining deck")
                used.add(row)
            spec_index[spec], spec_cid[spec] = row, cid

    def _resolve_spec(self, action: LegalAction, spec_index, spec_cid) -> tuple[int, int]:
        # Hidden opponent candidates and unknown identities retain zero refs.
        return spec_index.get(action.spec, 0), spec_cid.get(action.spec, 0)

    def _action_cid(self, msg: int, action: LegalAction, fallback: int) -> int:
        """``WriteState``: an explicitly set card id wins over the spec's.

        ``ygoenv`` only sets one itself where the description encoded *another*
        card (``la.cid_ = c_get_card_id(code_d)``) or for an announced card;
        everywhere else the id is whatever the card at the spec resolved to,
        which is 0 for a card we are not allowed to identify.  Taking
        ``action.code`` unconditionally would leak a face-down opponent card
        that the engine's own table keeps anonymous.
        """
        explicit = 0
        if msg in _ID_ONLY:
            explicit = self.pool.card_id(action.code)
        elif action.act == ActionAct.ACTIVATE and action.desc >= K.DESCRIPTION_LIMIT:
            explicit = self.pool.card_id(action.code)
        if explicit:
            return explicit
        return fallback if action.spec else 0

    def _write_action(self, row, msg: int, action: LegalAction, spec_index: int, cid: int) -> None:
        """``_set_obs_action`` (``ygopro.h:3645``); untouched columns stay 0."""
        row[3] = K.msg_to_id(msg)
        row[1] = (cid >> 8) & 0xFF
        row[2] = cid & 0xFF
        if msg in _MULTI_SELECT:
            if action.finish:
                row[5] = 1
            else:
                row[0] = spec_index & 0xFF
        elif msg == C.MSG_SELECT_POSITION:
            row[8] = K.position_to_id(action.position & 0x0F)
        elif msg == C.MSG_SELECT_EFFECTYN:
            row[0] = spec_index & 0xFF
            row[4] = int(action.act)
            row[6] = K.effect_to_id(action.effect)
        elif msg in (C.MSG_SELECT_YESNO, C.MSG_SELECT_OPTION):
            row[4] = int(action.act)
            row[6] = K.effect_to_id(action.effect)
        elif msg in (C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_IDLECMD, C.MSG_SELECT_CHAIN):
            row[7] = int(action.phase)
            row[0] = spec_index & 0xFF
            row[4] = int(action.act)
            row[6] = K.effect_to_id(action.effect)
        elif msg in (C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD):
            row[10] = action.place
        elif msg == C.MSG_ANNOUNCE_CARD:
            pass  # the card id above is the whole action
        elif msg == C.MSG_ANNOUNCE_ATTRIB:
            row[11] = K.attribute_to_id(action.attribute)
        elif msg == C.MSG_ANNOUNCE_RACE:
            row[12] = K.race_to_id(action.race)
        elif msg == C.MSG_ANNOUNCE_NUMBER:
            row[9] = action.number & 0xFF
        else:
            raise ValueError(f"message {msg} has no action encoding")

    # -- history -----------------------------------------------------------

    def _render_history(self, turn_count: int):
        """Newest first, with the turn column turned into a delta.

        ``ygoenv`` keeps a ring written backwards and unrolls it from the write
        pointer (``ygopro.h:2670``); a plain list read in reverse is the same
        sequence.  The turn column is rewritten in place to
        ``min(16, now - then)`` and the walk stops at the first row whose
        message id is 0, so trailing rows never get a delta.
        """
        geom = self.geom
        out = np.zeros((geom.n_history_actions, K.N_HISTORY_ACTION_FEATS), dtype=np.uint8)
        for i, (row, turn) in enumerate(reversed(self._history)):
            if i >= geom.n_history_actions:
                break
            out[i] = row
            out[i, K.N_ACTION_FEATS] = min(16, max(0, turn_count - turn))
        return out


def _spec_of(row: _Row, opponent: bool) -> str:
    """``Card::get_spec`` -- ``ls_to_spec(location_, sequence_, position_)``.

    A material's ``position_`` is its index in the stack, which is what turns
    into the ``a``/``b``/``c`` suffix.
    """
    loc = row.location
    seq = row.sequence
    pos = row.position
    letter = ""
    if loc & C.LOCATION_HAND:
        letter = "h"
    elif loc & C.LOCATION_MZONE:
        letter = "m"
    elif loc & C.LOCATION_SZONE:
        letter = "s"
    elif loc & C.LOCATION_GRAVE:
        letter = "g"
    elif loc & C.LOCATION_REMOVED:
        letter = "r"
    elif loc & C.LOCATION_EXTRA:
        letter = "x"
    spec = f"{letter}{seq + 1}"
    if loc & C.LOCATION_OVERLAY:
        spec += chr(ord("a") + pos)
    return ("o" + spec) if opponent else spec
