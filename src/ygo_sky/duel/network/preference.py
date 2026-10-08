"""Turn-order defaults based on deck contents, with explicit owner overrides."""

TENPAI = {39931513, 23657016, 91810826, 65326118}


def is_tenpai(main):
    present = TENPAI.intersection(main)
    return len(present) >= 2 and sum(code in TENPAI for code in main) >= 4


def prefer_first(main, preference="auto"):
    if preference not in ("auto", "first", "second", "random"):
        raise ValueError("Invalid turn-order preference")
    if preference == "random":
        return None
    if preference == "auto":
        return not is_tenpai(main)
    return preference == "first"
