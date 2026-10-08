"""Card-name candidate ordering from the pinned native observation contract."""
from .vendor.netduel.actions import OPCODE_ISCODE, card_is_declarable
from .vendor.netduel.cards import CardPool as BaseCardPool


class CardPool(BaseCardPool):
    def declarable(self, opcodes, preferred=None, limit=128):
        ranked = sorted((identity, code) for code, identity in self.card_ids.items()
                        if code in self.cards and card_is_declarable(self.cards[code], opcodes))
        explicit = [opcodes[i - 1] for i in range(1, len(opcodes)) if opcodes[i] == OPCODE_ISCODE]
        candidates, seen = [], set()
        for code in explicit + [code for _, code in ranked]:
            if len(candidates) >= limit:
                break
            if code not in seen and code in self.cards and card_is_declarable(self.cards[code], opcodes):
                if not self.card_id(code):
                    raise ValueError("Announce-card candidate is outside the semantic pool")
                candidates.append(code)
                seen.add(code)
        if not candidates:
            raise ValueError("No declarable cards match announce-card filter")
        return candidates
