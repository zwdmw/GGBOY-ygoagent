# Protocol Components

Source: a user-provided Mirrorforce snapshot, copied 2026-09-17.
The exact upstream repository and its redistribution license remain unresolved.

Only the TCP framing, legal response parser, board tracker, card database reader,
baseline card encoder, and shared selection decomposition are included.
`sources.json` records the source file hashes.

Local encoder correction, 2026-09-17: own-deck candidate references use the
prompt's card identity and deterministic duplicate-copy allocation. The old
seeded shuffle approximation is removed. The source hash remains the original
upstream snapshot; the vendored hash records this local modification.

The upstream PV-1 observation encoder is NOT a compatible policy adapter by
itself. `outbound/structured.py` supplies the deployed model's 12-column actions,
14-column history, native system-description IDs, action IR, selection groups
and observable public-event tensors.

Network clients never receive an opponent's private choices, full deck order,
or every engine query field. They do not invent those values or read a second
player's connection. Such differences must be retained in evaluation metadata.
