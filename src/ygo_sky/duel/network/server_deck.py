"""Server-assigned decks: known multiplicities and anonymous undisclosed cards."""
from collections import Counter
import struct

from .vendor.netduel import constants as C
from .vendor.netduel.board import ShadowBoard, ShadowCard, parse_query_segments


class ServerDeckBoard(ShadowBoard):
    def __init__(self):
        super().__init__()
        self.known_remaining = Counter()
        self.complete_deck = False
        self.assigned_extra = None

    def assign(self, deck):
        player = self.our_player
        if (len(deck["main"]) != self.deck_count[player]
                or len(deck["extra"]) != self.extra_count[player]):
            raise ValueError("deck_sync_start_count_mismatch")
        self.known_remaining = Counter(deck["main"])
        self.assigned_extra = Counter(deck["extra"])
        self.complete_deck = True

    def require_complete(self):
        if not self.complete_deck:
            raise ValueError("deck_sync_missing_own_deck")
        if sum(self.known_remaining.values()) != self.deck_count[self.our_player]:
            raise ValueError("deck_sync_remaining_count_mismatch")
        extra = self.zone(self.our_player, C.LOCATION_EXTRA)
        codes = [card.code & 0x7FFFFFFF if card is not None else 0 for card in extra]
        if len(codes) != self.extra_count[self.our_player] or not all(codes):
            raise ValueError("deck_sync_missing_extra_deck")
        if self.assigned_extra is not None and Counter(codes) != +self.assigned_extra:
            raise ValueError("deck_sync_extra_mismatch")

    def start(self, our_player, main, extra):
        # Never seed this view with the unrelated local selection.
        self.__init__()
        super().start(our_player, [], [])

    def _start_msg(self, body):
        super()._start_msg(body)
        player = self.our_player
        self.zones[player, C.LOCATION_EXTRA] = [
            ShadowCard(controller=player, owner=player, location=C.LOCATION_EXTRA,
                       sequence=i, position=C.POS_FACEDOWN_DEFENSE, hidden=True)
            for i in range(self.extra_count[player])]

    def _depart(self, code):
        code &= 0x7FFFFFFF
        if not code:
            if self.complete_deck:
                raise ValueError("deck_sync_unknown_departure")
            self.known_remaining.clear()
            self.complete_deck = False
        elif self.known_remaining[code]:
            self.known_remaining[code] -= 1
        elif self.complete_deck:
            raise ValueError("deck_sync_departure_mismatch")

    def _draw(self, body):
        if len(body) >= 2 and body[0] == self.our_player:
            if len(body) != 2 + 4 * body[1]:
                raise ValueError("Invalid assigned-deck draw")
            for offset in range(2, len(body), 4):
                self._depart(struct.unpack_from("<I", body, offset)[0])
        super()._draw(body)

    def _move(self, body):
        if len(body) >= 16:
            code, previous, current, _ = struct.unpack_from("<IIII", body)
            left = previous & 255 == self.our_player and (previous >> 8) & 255 == C.LOCATION_DECK
            entered = current & 255 == self.our_player and (current >> 8) & 255 == C.LOCATION_DECK
            if left and not entered:
                self._depart(code)
            elif entered and not left:
                if code & 0x7FFFFFFF:
                    self.known_remaining[code & 0x7FFFFFFF] += 1
                else:
                    if self.complete_deck:
                        raise ValueError("deck_sync_unknown_return")
                    self.complete_deck = False
            if self.assigned_extra is not None:
                left_extra = previous & 255 == self.our_player and (previous >> 8) & 255 == C.LOCATION_EXTRA
                entered_extra = current & 255 == self.our_player and (current >> 8) & 255 == C.LOCATION_EXTRA
                identity = code & 0x7FFFFFFF
                if left_extra and not entered_extra:
                    if not identity or self.assigned_extra[identity] <= 0:
                        raise ValueError("deck_sync_extra_departure_mismatch")
                    self.assigned_extra[identity] -= 1
                elif entered_extra and not left_extra:
                    if not identity:
                        raise ValueError("deck_sync_extra_unknown_return")
                    self.assigned_extra[identity] += 1
        super()._move(body)

    def _update_data(self, body):
        if len(body) >= 2 and body[:2] == bytes((self.our_player, C.LOCATION_DECK)):
            segments = parse_query_segments(body[2:])
            if len(segments) != self.deck_count[self.our_player]:
                raise ValueError("deck_sync_refresh_count_mismatch")
            codes = [segment.get("code", 0) & 0x7FFFFFFF if segment else 0 for segment in segments]
            if self.complete_deck:
                disclosed = Counter(code for code in codes if code)
                if (sum(self.known_remaining.values()) != len(segments)
                        or any(count > self.known_remaining[code] for code, count in disclosed.items())):
                    raise ValueError("deck_sync_refresh_identity_mismatch")
                self.updates += 1
                return
            # Cache-only slots cannot establish identity in an unordered view.
            if any(segment and segment.get("_nochange") for segment in segments):
                self.complete_deck = False
                return
            self.known_remaining = Counter(code for code in codes if code)
            self.complete_deck = all(codes)
            self.updates += 1
            return
        super()._update_data(body)
        if len(body) >= 2 and body[:2] == bytes((self.our_player, C.LOCATION_EXTRA)):
            for i, card in enumerate(self.zone(self.our_player, C.LOCATION_EXTRA)):
                if card is not None:
                    card.controller, card.location, card.sequence = self.our_player, C.LOCATION_EXTRA, i

    def deck_codes(self, candidates):
        offered = {}
        for action in candidates:
            if action.spec.isdigit() and action.code:
                if action.spec in offered and offered[action.spec] != action.code:
                    raise ValueError("Conflicting assigned-deck candidate identity")
                offered[action.spec] = action.code
        visible = Counter(offered.values())
        if self.complete_deck and any(count > self.known_remaining[code] for code, count in visible.items()):
            raise ValueError("deck_sync_candidate_mismatch")
        remaining = self.known_remaining | visible
        count = self.deck_count[self.our_player]
        if self.complete_deck and sum(self.known_remaining.values()) != count:
            raise ValueError("deck_sync_remaining_count_mismatch")
        if sum(remaining.values()) > count:
            raise ValueError("deck_sync_candidate_count_mismatch")
        return list(remaining.elements()) + [0] * (count - sum(remaining.values()))
