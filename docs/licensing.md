# Licensing — maintainer decision required

**Status: MLBreak has no software license.** The repository contains no `LICENSE` file, and neither
`pyproject.toml` nor `frontend/package.json` declares one. Under default copyright, others may view and fork
the public repository (GitHub's terms of service) but have **no right to use, modify or redistribute** the code.
This must be decided by the maintainer before the project is presented as open source. **No license has been
chosen on the maintainer's behalf.**

## What is already settled (independent of the choice)

| Component | License | Obligation |
|---|---|---|
| Bundled **SMS Spam Collection v.1** (`src/mlbreak/text/data/sms_spam_collection/`) | CC BY 4.0 (UCI) | Keep the attribution in `NOTICE.md` and the original `readme`; it is a data license, separate from the code license, and compatible with every option below |
| `moons-v1` dataset | none needed | Generated at runtime by scikit-learn's `make_moons`; nothing is redistributed |
| Python runtime deps | NumPy (BSD-3-Clause + 0BSD/MIT/Zlib/CC0 parts), SciPy (BSD), scikit-learn, pandas, uvicorn, Starlette, joblib, threadpoolctl (BSD-3-Clause), Pydantic, FastAPI (MIT) | All permissive: compatible with every option below |
| Frontend deps | React, React DOM, React Router, Plotly.js, Vite (MIT), TypeScript (Apache-2.0, build-time) | All permissive: compatible with every option below |
| Design reference [AgentBreak](https://github.com/urcuqui/AgentBreak) and [SpamShieldAI](https://github.com/urcuqui/SpamShieldAI) | no license file | **No code was copied** from either (both are referenced as methodology); nothing to inherit |

(Checked from installed package metadata on 2026-11-15; re-check when dependencies change.)

## Options

| License | Kind | Patent grant | Main consequence for MLBreak |
|---|---|---|---|
| **MIT** | permissive | no explicit grant | Simplest; anyone may reuse the labs in courses, products or other tools with attribution |
| **Apache-2.0** | permissive | **yes**, with patent-retaliation clause | Like MIT plus an explicit patent licence and a `NOTICE` convention (already used for the dataset); common in security tooling and ML |
| **BSD-3-Clause** | permissive | no explicit grant | Same family as NumPy/scikit-learn; adds a no-endorsement clause |
| **MPL-2.0** | weak (file-level) copyleft | yes | Modified MLBreak files must stay open; can be combined with proprietary code at file granularity |
| **GPL-3.0-or-later** | strong copyleft | yes | Distributed derivatives must be GPL; running MLBreak as a hosted service does **not** trigger source sharing |
| **AGPL-3.0-or-later** | strong copyleft incl. network use | yes | Like GPL, and anyone offering a modified MLBreak **as a network service** must publish their changes — relevant because MLBreak is a web application |

Considerations specific to this project:

* It is an **intentionally vulnerable** educational tool. Whatever the license, keep the "use at your own risk /
  no warranty" disclaimer that every standard license above includes, and the existing
  [responsible-use](responsible-use.md) guidance (a license cannot restrict *how* the code is used for any of
  the OSI options; field-of-use restrictions would make it non-open-source).
* Course and workshop reuse is easiest under MIT / Apache-2.0 / BSD-3-Clause.
* If contributions are expected, decide whether a DCO sign-off or a CLA is wanted; Apache-2.0 includes an
  inbound=outbound contribution clause by default.

## After the decision (checklist)

1. Add `LICENSE` at the repository root with the full license text (and, for Apache-2.0, a root `NOTICE`).
2. Declare it: `license = "<SPDX id>"` in `pyproject.toml`, `"license": "<SPDX id>"` in `frontend/package.json`.
3. Replace the "License" section of the README and this document's status line.
4. Keep `src/mlbreak/text/data/sms_spam_collection/NOTICE.md` (CC BY 4.0 attribution) unchanged.
