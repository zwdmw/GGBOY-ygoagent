# Component notices

Sky Striker Expert 0.1.0 is published at https://github.com/zwdmw/GGBOY-ygoagent. The root MIT LICENSE applies only to the original integration code and documentation contributed for this project. It does not relicense third-party code, trained weights, game assets, derived semantic resources or native dependencies. Existing upstream terms continue to apply to their code and derived modifications.

We thank ygo-agent (https://github.com/sbl1996/ygo-agent), its authors and contributors for making its model, reinforcement learning and environment implementations available. The `ygoai` library and modified `ygoenv` sources originate from the user-provided ygo-agent snapshot. Its MIT notice (Copyright 2024 Hastur) and Apache-2.0 notice for EnvPool (Copyright 2021 Garena Online Private Limited) are retained in `third_party/ygo-agent/LICENSE`. An identical copy is supplied as `third_party/ygo-agent/YGO-AGENT-LICENSE.txt` to avoid a filename collision with the integration LICENSE in wheel metadata. Complete Apache-2.0 terms are supplied in `third_party/ygo-agent/APACHE-2.0.txt`. The wheel includes the integration LICENSE, both upstream notice files and this notice.

We also thank 海之中道, the author of MirrorForce. We learned a great deal from our discussions with him.

The vendored network and replay utilities originate from a user-provided MirrorForce snapshot. Their upstream license remains unresolved. Retaining the known ygo-agent notices does not grant a blanket license over those components.

Card scripts, card data, semantic tables, checkpoints, decks and native reference binaries are supplied separately. Their provenance and applicable notices are described in `docs/第三方资源说明.md` and `third_party/SOURCES.json`; resource redistribution terms and the complete linked native dependency license bundle remain to be settled.
