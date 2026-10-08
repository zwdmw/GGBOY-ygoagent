"""The id tables ``ygopro.h`` bakes into every observation.

Each table is built there by ``make_ids`` (``ygopro.h:694-719``), which numbers
a ``std::map`` by **sorted key** and a ``std::vector`` by **position**, plus a
constant offset.  Both rules are reproduced here from the same key order, so
the only thing that can drift is the key list itself -- and
``tests/test_netduel.py`` re-parses ``ygopro.h`` on every run and fails if it
has.

Why this matters more than a usual constants file: the ids are *learned
embedding indices*.  A checkpoint is welded to the exact table it trained
against, so inserting a key in the middle silently renumbers everything after
it and the weights become meaningless without a single error being raised.
That has already happened once to this project (``pv/core-upgrade-results.md``:
the system-string shift invalidated the upstream public checkpoints).  Append,
never insert.
"""

from __future__ import annotations

__all__ = [
    "LOCATION2ID",
    "POSITION2ID",
    "ATTRIBUTE2ID",
    "RACE2ID",
    "PHASE2ID",
    "MSG2ID",
    "SYSTEM_STRING2ID",
    "TYPE_BITS",
    "N_ACTION_FEATS",
    "N_HISTORY_ACTION_FEATS",
    "CARD_EFFECT_OFFSET",
    "DESCRIPTION_LIMIT",
    "location_to_id",
    "position_to_id",
    "attribute_to_id",
    "race_to_id",
    "phase_to_id",
    "msg_to_id",
    "type_to_ids",
    "float_transform",
    "effect_to_id",
]

N_ACTION_FEATS = 13  # ygopro.h:1145 kNActionFeats
N_HISTORY_ACTION_FEATS = N_ACTION_FEATS + 2  # turn delta, phase
CARD_EFFECT_OFFSET = 10010  # ygopro.h:1136
DESCRIPTION_LIMIT = 10000  # ygopro.h:1135


def _by_sorted_key(keys, offset: int = 0) -> dict[int, int]:
    """``make_ids`` over a ``std::map``: ids follow the sorted key order."""
    return {k: i + offset for i, k in enumerate(sorted(keys))}


def _by_position(keys, offset: int = 0) -> dict[int, int]:
    """``make_ids`` over a ``std::vector``: ids follow the source order."""
    return {k: i + offset for i, k in enumerate(keys)}


# location2str (ygopro.h:767), offset 1
LOCATION2ID = _by_sorted_key([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40], 1)

# position2str (ygopro.h:784); POS_NONE is the xyz-material placeholder
POSITION2ID = _by_sorted_key([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x8, 0xA, 0xC])

# attribute2str (ygopro.h:804): none, earth, water, fire, wind, light, dark, divine
ATTRIBUTE2ID = _by_sorted_key([0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40])

# race2str (ygopro.h:819): none plus the 26 race bits
RACE2ID = _by_sorted_key([0] + [1 << i for i in range(26)])

# phase2str (ygopro.h:891)
PHASE2ID = _by_sorted_key([1 << i for i in range(10)])

# _msgs (ygopro.h:910) is a std::vector, so these ids are positional and
# MSG_ANNOUNCE_RACE was *appended* precisely to keep 1..16 stable.
MSG2ID = _by_position(
    [11, 16, 15, 20, 19, 12, 13, 10, 26, 14, 18, 23, 24, 141, 143, 142, 140], 1
)

# type2str (ygopro.h:854); type_to_ids emits one 0/1 column per bit, sorted
TYPE_BITS = tuple(
    sorted(
        [
            0x1,  # monster
            0x2,  # spell
            0x4,  # trap
            0x10,  # normal
            0x20,  # effect
            0x40,  # fusion
            0x80,  # ritual
            0x100,  # trap monster
            0x200,  # spirit
            0x400,  # union
            0x800,  # dual
            0x1000,  # tuner
            0x2000,  # synchro
            0x4000,  # token
            0x10000,  # quick-play
            0x20000,  # continuous
            0x40000,  # equip
            0x80000,  # field
            0x100000,  # counter
            0x200000,  # flip
            0x400000,  # toon
            0x800000,  # xyz
            0x1000000,  # pendulum
            0x2000000,  # special summon
            0x4000000,  # link
        ]
    )
)

# system_strings (ygopro.h:439) is a std::map with offset 16.  The list below is
# the *unique* sorted key set: 221 appears twice in the source and std::map
# keeps the first, so a duplicate would shift every later id by one.
SYSTEM_STRING2ID = _by_sorted_key(
    [
        1, 30, 31,
        60, 61, 62, 63, 64, 65, 66, 67,
        70, 71, 72,
        80, 81,
        90, 91, 92, 93, 94, 95, 96, 97, 98,
        200, 203, 210, 218, 219, 220, 221, 222,
        1050, 1051, 1052, 1054, 1055, 1056, 1057, 1058, 1059,
        1060, 1061, 1062, 1063, 1064, 1066, 1067, 1068, 1069,
        1070, 1071, 1072, 1073, 1074, 1075, 1076, 1080, 1081,
        1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159,
        1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169,
        1190, 1191, 1192, 1193,
        1213, 1214,
        1621, 1622,
    ],
    16,
)


def _lookup(table: dict[int, int], key: int, what: str) -> int:
    try:
        return table[key]
    except KeyError:  # pragma: no cover - a miss means the tables drifted
        raise KeyError(f"{what} has no id for 0x{key:x}") from None


def location_to_id(location: int) -> int:
    return _lookup(LOCATION2ID, location, "location")


def position_to_id(position: int) -> int:
    return _lookup(POSITION2ID, position, "position")


def attribute_to_id(attribute: int) -> int:
    return _lookup(ATTRIBUTE2ID, attribute, "attribute")


def race_to_id(race: int) -> int:
    return _lookup(RACE2ID, race, "race")


def phase_to_id(phase: int) -> int:
    return _lookup(PHASE2ID, phase, "phase")


def msg_to_id(msg: int) -> int:
    return _lookup(MSG2ID, msg, "message")


def type_to_ids(type_: int) -> list[int]:
    """25 columns, ``min(1, type & bit)`` each (``ygopro.h:882``)."""
    return [1 if type_ & bit else 0 for bit in TYPE_BITS]


def float_transform(x: int) -> tuple[int, int]:
    """``ygopro.h:1075``: a 16 bit big-endian split, negatives wrapped."""
    x = x % 65536
    return (x >> 8) & 0xFF, x & 0xFF


def effect_to_id(effect: int) -> int:
    """``_set_obs_action_effect`` (``ygopro.h:3602``).

    0 is "no effect", 1 is the card's default effect, 2-15 are the card's own
    numbered effects and 16+ are engine system strings.
    """
    if effect == -1:
        return 0
    if effect == 0:
        return 1
    if effect >= CARD_EFFECT_OFFSET:
        return effect - CARD_EFFECT_OFFSET + 2
    return _lookup(SYSTEM_STRING2ID, effect, "system string")
