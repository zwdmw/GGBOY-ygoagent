"""Opt-in, connection-bound own-deck manifests. No shuffled order is transmitted."""
from collections import Counter
import hashlib
import json
import re

OPCODE = 0xF0
MAGIC = b"YGO-OWN-DECK/1\0"
SCHEMA = "assigned-own-deck/v1"
MAX_PAYLOAD = 8192


def request(nonce):
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise ValueError("deck_sync_invalid_nonce")
    return MAGIC + bytes.fromhex(nonce)


def read_request(payload):
    if len(payload) != len(MAGIC) + 16 or not payload.startswith(MAGIC):
        raise ValueError("deck_sync_invalid_request")
    return payload[len(MAGIC):].hex()


def canonical_deck(deck):
    if not isinstance(deck, dict) or set(deck) != {"main", "extra", "side"}:
        raise ValueError("deck_sync_invalid_sections")
    result = {}
    for name, limit in (("main", 60), ("extra", 15), ("side", 15)):
        cards = deck[name]
        if (not isinstance(cards, list) or not (40 if name == "main" else 0) <= len(cards) <= limit
                or any(type(code) is not int or not 0 < code < 2**31 for code in cards)):
            raise ValueError("deck_sync_invalid_cards")
        result[name] = sorted(cards)
    return result


def fingerprint(deck):
    raw = json.dumps(canonical_deck(deck), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def manifest(deck, nonce, game):
    request(nonce)
    if type(game) is not int or not 1 <= game <= 5:
        raise ValueError("deck_sync_invalid_game")
    deck = canonical_deck(deck)
    return {"schema": SCHEMA, "nonce": nonce, "game": game,
            "cards": {name: [[code, count] for code, count in sorted(Counter(cards).items())]
                      for name, cards in deck.items()},
            "sha256": fingerprint(deck)}


def encode(deck, nonce, game):
    return json.dumps(manifest(deck, nonce, game), separators=(",", ":")).encode("ascii")


def decode(payload, nonce, game):
    if not 0 < len(payload) <= MAX_PAYLOAD:
        raise ValueError("deck_sync_invalid_size")
    try:
        doc = json.loads(payload)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("deck_sync_invalid_json") from exc
    if (not isinstance(doc, dict) or set(doc) != {"schema", "nonce", "game", "cards", "sha256"}
            or doc["schema"] != SCHEMA or doc["nonce"] != nonce
            or type(doc["game"]) is not int or doc["game"] != game):
        raise ValueError("deck_sync_wrong_connection_or_game")
    if not isinstance(doc["cards"], dict) or set(doc["cards"]) != {"main", "extra", "side"}:
        raise ValueError("deck_sync_invalid_sections")
    deck = {}
    for name, rows in doc["cards"].items():
        if not isinstance(rows, list) or len(rows) > 60:
            raise ValueError("deck_sync_invalid_counts")
        cards, seen = [], set()
        for row in rows:
            if (not isinstance(row, list) or len(row) != 2
                    or type(row[0]) is not int or not 0 < row[0] < 2**31
                    or type(row[1]) is not int or not 1 <= row[1] <= 60
                    or row[0] in seen):
                raise ValueError("deck_sync_invalid_counts")
            seen.add(row[0])
            cards.extend([row[0]] * row[1])
            if len(cards) > 60:
                raise ValueError("deck_sync_invalid_counts")
        deck[name] = cards
    deck = canonical_deck(deck)
    if doc["sha256"] != fingerprint(deck):
        raise ValueError("deck_sync_hash_mismatch")
    return doc, deck
