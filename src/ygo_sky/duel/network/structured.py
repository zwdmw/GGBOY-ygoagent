"""Native structured feature contract applied to one player's network view."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .vendor.netduel import constants as C, obsconst as K
from .vendor.netduel.actions import ActionAct, ActionPhase, LegalAction, ls_to_spec, unpack_desc
from .vendor.netduel.encode import ObsEncoder, ObsGeometry, _Row


CONTRACT = json.loads(Path(__file__).with_name("native_contract.json").read_text())
OBSERVATION_VERSION = "structured-observable-v4-phase-history"
MESSAGES = {int(k): v for k, v in CONTRACT["messages"].items()}
SYSTEMS = {int(k): v for k, v in CONTRACT["system_strings"].items()}
CANDIDATES = {C.MSG_SELECT_CARD, C.MSG_SELECT_TRIBUTE, C.MSG_SELECT_SUM,
              C.MSG_SELECT_UNSELECT_CARD}
SOURCES = {C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_CHAIN,
           C.MSG_SELECT_EFFECTYN}


def effect_id(effect):
    if effect == -1:
        return 0
    if effect == 0:
        return 1
    if effect >= 10010:
        return min(255, effect - 10010 + 2)
    return SYSTEMS.get(effect, 255)


def role(msg):
    return 5 if msg == C.MSG_SELECT_TRIBUTE else 4 if msg == C.MSG_SELECT_SUM else (
        1 if msg in CANDIDATES else 0)


def selection_context(selector):
    ms = getattr(selector, "ms", None)
    if ms is None or ms.done:
        return 0, 0, 0, 0, 0, 0, [], False
    return (ms.idx, ms.min, ms.max, ms.must, len(ms.r_idxs), ms.mode,
            [ms.specs[i] for i in ms.r_idxs if 0 <= i < len(ms.specs)], True)


class Geometry(ObsGeometry):
    max_options = 128
    n_action_feats = 12


@dataclass
class Event:
    actor: int
    msg: int
    turn: int
    phase: int
    action: object = None
    source: str = ""
    candidate: str = ""
    stage: int = 0
    selected: int = 0
    payload: bytes = b""
    candidate_prompt: object = None


class StructuredEncoder(ObsEncoder):
    def __init__(self, pool):
        super().__init__(pool, Geometry())
        self.events = []
        self.pending_spec = ""
        self.pending_cid = 0
        self.pending_effect = -1
        self.refs = {}
        self.cids = {}
        self.last_actions = None
        self._selector = None
        self._turn_player = None
        self._phase_turn = 0
        self._observed_phase = 0
        self.network_residuals = {
            "opponent_private_choices_unavailable": True,
            "own_deck_order_unavailable": True,
            "observable_opponent_actions_reconstructed": True,
            "own_deck_candidates_by_identity": True,
            "native_deck_candidate_mapping_changed": True,
            "opponent_phase_actions_inferred_from_broadcasts": True,
            "opponent_phase_intent_unavailable_until_transition": True,
        }

    def reset(self, our_player, seed=0):
        super().reset(our_player, seed)
        self.events.clear()
        self.pending_spec = ""
        self.pending_cid = 0
        self.pending_effect = -1
        self.refs, self.cids = {}, {}
        self.last_actions = None
        self._selector = None
        self._turn_player = None
        self._phase_turn = 0
        self._observed_phase = 0

    def _write_action(self, row, msg, action, spec_index, cid):
        row[3] = MESSAGES[msg]
        row[1:3] = (cid >> 8, cid & 255)
        if msg in CANDIDATES:
            row[5 if action.finish else 0] = 1 if action.finish else spec_index
        elif msg == C.MSG_SELECT_POSITION:
            row[8] = K.position_to_id(action.position)
        elif msg == C.MSG_SELECT_EFFECTYN:
            row[0], row[4], row[6] = spec_index, int(action.act), effect_id(action.effect)
        elif msg in (C.MSG_SELECT_YESNO, C.MSG_SELECT_OPTION):
            row[4], row[6] = int(action.act), effect_id(action.effect)
        elif msg in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_CHAIN):
            row[7], row[0], row[4], row[6] = (
                int(action.phase), spec_index, int(action.act), effect_id(action.effect))
        elif msg in (C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD):
            row[10] = action.place
        elif msg == C.MSG_ANNOUNCE_ATTRIB:
            row[11] = K.attribute_to_id(action.attribute)
        elif msg == C.MSG_ANNOUNCE_NUMBER:
            row[9] = action.number & 255
        elif msg == C.MSG_ANNOUNCE_RACE:
            row[6] = 128 + max(0, action.race.bit_length() - 1)
        elif msg != C.MSG_ANNOUNCE_CARD:
            raise ValueError("Unsupported native action message: " + str(msg))

    def _render_history(self, turn_count):
        out = np.zeros((32, 14), dtype=np.uint8)
        for i, (row, turn) in enumerate(reversed(self._history[-32:])):
            out[i] = row
            out[i, 12] = min(16, max(0, turn_count - (turn & 255)))
        return out

    def encode_state(self, state, selector):
        if not 1 <= len(state.actions) <= 128:
            raise ValueError("Native action capacity exceeded")
        previous_overflow = self.row_overflows
        previous_disagreements = self.deck_disagreements
        ms = getattr(selector, "ms", None)
        self._deck_candidates = state.actions
        if ms is not None and not ms.done:
            if len(ms.specs) != len(selector.codes):
                raise ValueError("Selection candidate identity count mismatch")
            self._deck_candidates = [LegalAction(spec=spec, code=code)
                                     for spec, code in zip(ms.specs, selector.codes)]
        cards, refs, cids, counts = self._encode_cards(state.board)
        if self.row_overflows != previous_overflow:
            raise ValueError("Native card capacity exceeded")
        if self.unknown_codes:
            raise ValueError("Observed cards outside the model database: " +
                             ",".join(map(str, sorted(self.unknown_codes)[:8])))
        # Validate only visible encoded IDs, never inspect an opponent's hidden cards.
        visible_ids = set((cards[:, 0].astype(np.uint16) * 256 + cards[:, 1]).tolist()) - {0}
        pending_ids = {self.pool.card_id(code): code
                       for code in getattr(self.pool, "semantic_pending", set())}
        self._check_codes([pending_ids[cid] for cid in visible_ids & pending_ids.keys()])
        self._check_codes([action.code for action in state.actions if action.code])
        if self.deck_disagreements != previous_disagreements:
            raise ValueError("Remaining deck count disagrees with the server")
        self._selector = selector
        ms = getattr(selector, "ms", None)
        if ms is not None and not ms.done:
            if len(ms.specs) != len(selector.codes):
                raise ValueError("Selection candidate identity count mismatch")
            # Reserve selected and currently filtered-out copies too, so the
            # remaining candidates cannot reuse their rows between substeps.
            self._bind_deck_candidates(
                [LegalAction(spec=spec, code=code)
                 for spec, code in zip(ms.specs, selector.codes)], refs, cids)
        self.refs, self.cids = refs, cids
        actions = self._encode_actions(state.msg, state.actions, refs, cids)
        for i, action in enumerate(state.actions):
            if action.spec and int(actions[i, 0]):
                refs.setdefault(action.spec, int(actions[i, 0]))
                cids.setdefault(action.spec, self._resolved_cids[i])
        self.last_actions = actions
        stage, minimum, maximum, must, selected, mode, selected_specs, active = selection_context(selector)
        finishable = any(a.finish for a in state.actions)
        cancelable = any(a.act == ActionAct.CANCEL for a in state.actions)
        mandatory = active and selected < max(minimum, must)
        ir = np.zeros((128, 24), dtype=np.uint8)
        singles = np.zeros((128, 4), dtype=np.uint8)
        groups = np.zeros((128, 5, 8), dtype=np.uint8)
        masks = np.zeros_like(groups)
        selected_refs = [refs[s] for s in selected_specs if refs.get(s)][:8]
        selection_role = role(state.msg)
        for i, action in enumerate(state.actions):
            candidate = state.msg in CANDIDATES and not action.finish and action.act != ActionAct.CANCEL
            source_ref, candidate_ref, source_cid, confidence = 0, 0, 0, 0
            source_effect = action.effect
            action_ref = int(actions[i, 0])
            if candidate:
                candidate_ref = action_ref
            elif action.spec:
                source_ref, source_cid = action_ref, self._resolved_cids[i]
                confidence = 2 if source_ref else 0
            if not source_ref and self.pending_spec and state.msg not in SOURCES:
                source_ref = refs.get(self.pending_spec, 0)
                source_cid = cids.get(self.pending_spec, 0)
                if not source_cid and not self.pending_spec.startswith("o"):
                    source_cid = self.pending_cid
                source_effect = self.pending_effect
                confidence = 1 if source_ref else 0
            singles[i, :2] = source_ref, candidate_ref
            if action.act in (ActionAct.ATTACK, ActionAct.DIRECT_ATTACK):
                singles[i, 2] = source_ref
            for j, ref in enumerate(selected_refs):
                groups[i, 4, j], masks[i, 4, j] = ref, 1
                group = {4: 2, 5: 3, 2: 0, 3: 1}.get(selection_role)
                if group is not None:
                    groups[i, group, j], masks[i, group, j] = ref, 1
            number = max(0, action.race.bit_length() - 1) if state.msg == C.MSG_ANNOUNCE_RACE else action.number
            ir[i] = [
                MESSAGES[state.msg], action.act, action.phase, selection_role,
                min(stage, 15), action.finish, action.act == ActionAct.CANCEL,
                mandatory, finishable, cancelable, minimum, maximum, selected,
                must, effect_id(source_effect), source_cid >> 8, source_cid & 255,
                confidence, K.position_to_id(action.position), action.place,
                number & 255, K.attribute_to_id(action.attribute), i + 1, mode]
        selection = np.asarray([
            MESSAGES[state.msg], selection_role, stage, minimum, maximum, must,
            selected, finishable, cancelable, mandatory, mode, len(state.actions)
        ], dtype=np.uint8)
        events, event_refs = self._events(state.turn, refs)
        arrays = {
            "cards_": cards,
            "global_": self._encode_global(state.lp, state.turn, state.phase,
                                          state.board.turn_player, counts),
            "actions_": actions, "h_actions_": self._render_history(state.turn),
            "action_ir_": ir, "action_single_refs_": singles,
            "action_group_refs_": groups, "action_group_mask_": masks,
            "selection_": selection, "public_events_": events,
            "public_event_refs_": event_refs,
        }
        return {**{key: value[None, ...] for key, value in arrays.items()}, "mask_": None}

    def _our_deck_rows(self, board, player):
        from .server_deck import ServerDeckBoard
        if not isinstance(board, ServerDeckBoard):
            return super()._our_deck_rows(board, player)
        codes = board.deck_codes(getattr(self, "_deck_candidates", ()))
        self._check_codes([code for code in codes if code])
        self.network_residuals["own_deck_composition_unavailable"] = not board.complete_deck
        rows = []
        for code in sorted(codes, key=lambda value: (self.pool.card_id(value), value)):
            data = self.pool.cards.get(code)
            rows.append(_Row(code=code, location=C.LOCATION_DECK, sequence=0,
                             controller=player, position=C.POS_FACEDOWN, overlay=0,
                             level=data.level if data else 0, attack=data.attack if data else 0,
                             defense=data.defense if data else 0, counter=0, negated=0,
                             hide=not bool(code)))
        return rows

    def record(self, state, action, selector, index=None, forced=False):
        self._check_codes([action.code] if action.code else [])
        if action.act == ActionAct.CANCEL:
            return
        cid = (self._action_cid(state.msg, action, 0) or self.pool.card_id(action.code)
               if forced else self._resolved_cids[index])
        row = np.zeros(14, dtype=np.uint8)
        self._write_action(row, state.msg, action, 0, cid)
        row[12:14] = state.turn & 255, K.phase_to_id(state.phase) if state.phase else 0
        self._history.append((row, state.turn))
        self._history = self._history[-32:]
        ms = getattr(selector, "ms", None)
        event = Event(self.our_player, state.msg, state.turn, state.phase, action,
                      stage=0 if ms is None or ms.done else ms.idx,
                      selected=len(ms.r_idxs) if ms is not None else 0)
        if state.msg in CANDIDATES:
            event.candidate, event.source = action.spec, self.pending_spec
            event.candidate_prompt = selector
        elif action.spec:
            event.source = action.spec
        self._add_event(event)
        if state.msg in SOURCES and action.spec:
            self.pending_spec, self.pending_cid, self.pending_effect = action.spec, cid, action.effect
        elif action.phase:
            self._clear_pending_source()

    def _clear_pending_source(self):
        self.pending_spec, self.pending_cid, self.pending_effect = "", 0, -1

    def _observe_phase(self, phase, turn):
        previous = self._observed_phase
        same_turn = self._phase_turn == turn
        if same_turn and previous == phase:
            return
        self._observed_phase = phase
        if not same_turn:
            self._turn_player = None
        self._phase_turn = turn
        transition = None
        if previous == C.PHASE_MAIN1 and phase == C.PHASE_BATTLE_START:
            transition = C.MSG_SELECT_IDLECMD, ActionPhase.BATTLE
        elif previous in (C.PHASE_MAIN1, C.PHASE_MAIN2) and phase == C.PHASE_END:
            transition = C.MSG_SELECT_IDLECMD, ActionPhase.END
        elif previous in (C.PHASE_BATTLE_START, C.PHASE_BATTLE):
            if phase == C.PHASE_MAIN2:
                transition = C.MSG_SELECT_BATTLECMD, ActionPhase.MAIN2
            elif phase == C.PHASE_END:
                transition = C.MSG_SELECT_BATTLECMD, ActionPhase.END
        # Own phase choices are already recorded. Peer intent is observable only
        # after the server broadcasts the transition, not during response windows.
        if same_turn and self._turn_player == 1 - self.our_player and transition:
            msg, action_phase = transition
            action = LegalAction(phase=action_phase)
            self._add_event(Event(self._turn_player, msg, turn, previous, action))
        self._clear_pending_source()

    def _add_event(self, event):
        self.events.insert(0, event)
        del self.events[32:]

    def _check_codes(self, codes):
        unknown = {code for code in codes if not self.pool.card_id(code)}
        if unknown:
            raise ValueError("Observed cards outside the model database: " +
                             ",".join(map(str, sorted(unknown))))
        pending = set(codes) & getattr(self.pool, "semantic_pending", set())
        if pending:
            raise ValueError("New-card effect scripts are pending: " +
                             ",".join(map(str, sorted(pending))))

    def observe_message(self, msg, body, turn, phase):
        if msg == C.MSG_NEW_TURN:
            if len(body) != 1 or body[0] not in (0, 1):
                raise ValueError("Invalid new-turn player")
            self._turn_player = body[0]
            self._phase_turn = turn
            self._observed_phase = 0
            self._clear_pending_source()
        elif msg == C.MSG_NEW_PHASE:
            if len(body) != 2:
                raise ValueError("Invalid new-phase payload")
            self._observe_phase(int.from_bytes(body, "little"), turn)
        elif msg in (C.MSG_TOSS_COIN, C.MSG_TOSS_DICE) and len(body) >= 2:
            self._add_event(Event(body[0], msg, turn, phase, payload=body[2:2 + body[1]][:5]))
        elif msg == C.MSG_SWAP_GRAVE_DECK:
            raise ValueError("Deck/grave exchange requires a supported board resynchronization")
        elif msg == C.MSG_CHAINING and len(body) >= 16:
            code = int.from_bytes(body[:4], "little")
            self._check_codes([code] if code else [])
            controller, location, sequence, position = body[4:8]
            desc = int.from_bytes(body[11:15], "little")
            _, effect = unpack_desc(code, desc)
            spec = ls_to_spec(location, sequence, position, controller != self.our_player)
            self.pending_spec, self.pending_effect = spec, effect
            self.pending_cid = self.pool.card_id(code)
            if controller != self.our_player:
                action = LegalAction(spec=spec, code=code, effect=effect, act=ActionAct.ACTIVATE)
                self._add_event(Event(controller, C.MSG_SELECT_CHAIN, turn, phase, action, spec))
        elif msg in (C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING) and len(body) >= 8:
            code = int.from_bytes(body[:4], "little")
            self._check_codes([code] if code else [])
            controller, location, sequence, position = body[4:8]
            if controller != self.our_player:
                spec = ls_to_spec(location, sequence, position, True)
                act = ActionAct.SUMMON if msg == C.MSG_SUMMONING else ActionAct.SPSUMMON
                action = LegalAction(spec=spec, code=code, act=act)
                self._add_event(Event(controller, C.MSG_SELECT_IDLECMD, turn, phase, action, spec))
        elif msg == C.MSG_ATTACK and len(body) >= 8:
            controller, location, sequence, position = body[:4]
            if controller != self.our_player:
                spec = ls_to_spec(location, sequence, position, True)
                act = ActionAct.DIRECT_ATTACK if not body[5] else ActionAct.ATTACK
                action = LegalAction(spec=spec, act=act)
                self._add_event(Event(controller, C.MSG_SELECT_BATTLECMD, turn, phase, action, spec))

    def _events(self, turn, refs):
        events = np.zeros((32, 16), dtype=np.uint8)
        references = np.zeros((32, 4), dtype=np.uint8)
        for i, event in enumerate(self.events):
            action = event.action or LegalAction()
            source, candidate = refs.get(event.source, 0), refs.get(event.candidate, 0)
            if event.candidate.isdigit() and (
                    event.candidate_prompt is not self._selector or
                    self.cids.get(event.candidate, 0) != self.pool.card_id(action.code)):
                # Numeric specs belong to one prompt, not to a persistent card.
                candidate = 0
            events[i, :10] = [
                1, 1 if event.actor == self.our_player else 2, MESSAGES[event.msg],
                action.act, action.phase, role(event.msg), min(16, max(0, turn - event.turn)),
                K.phase_to_id(event.phase) if event.phase else 0,
                action.finish, action.act == ActionAct.CANCEL]
            if event.payload:
                events[i, 10] = len(event.payload)
                events[i, 11:11 + len(event.payload)] = list(event.payload)
            else:
                choice = action.number or (K.attribute_to_id(action.attribute) if action.attribute else 0)
                events[i, 10:] = [effect_id(action.effect), min(event.stage, 15),
                                 min(event.selected, 255), bool(source), bool(candidate), choice & 255]
            references[i, :2] = source, candidate
            if action.act in (ActionAct.ATTACK, ActionAct.DIRECT_ATTACK):
                references[i, 2] = source
        return events, references
