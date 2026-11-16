# Datasets and attribution

MLBreak downloads nothing at runtime. Every dataset is either generated deterministically or bundled with its
attribution and verified before use.

| Dataset id | Used by | Origin | License / terms | Integrity |
|---|---|---|---|---|
| `moons-v1` | ML01 2-D evasion, ML02 poisoning | Generated: `sklearn.datasets.make_moons(n_samples=500, noise=0.25, random_state=seed)`, stratified 70/30 split | No third-party data | Train/test split SHA-256 recorded in every record |
| `synthetic-account-risk-v1` | ML03 model inversion | Generated in-process from a documented logistic generator (`default_rng([seed, 3003])`): 4 public features, a binary confidential attribute, a binary risk label; train 1200 / auxiliary 600 / evaluation 400 | No third-party data; **fictional records only** | Train, auxiliary and evaluation SHA-256 recorded in every record |
| `sms-spam-collection-v1` | ML01 spam-filter evasion | **SMS Spam Collection v.1**, Almeida & Gómez Hidalgo (2011), UCI Machine Learning Repository, DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84) | **CC BY 4.0** (UCI); the authors' `readme` (bundled unchanged) grants free use subject to their copyright and a warranty disclaimer | File SHA-256 `7d039a24…c239d` checked at load; a modified file is refused; split hashes recorded |

**Attribution for the SMS Spam Collection** (required by CC BY 4.0):

> Almeida, T. A., & Gómez Hidalgo, J. M. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C5CC84 — licensed under CC BY 4.0.
> Paper: Almeida, Gómez Hidalgo & Yamakami (2011), *Contributions to the Study of SMS Spam Filtering: New
> Collection and Results*, DocEng'11. https://doi.org/10.1145/2034691.2034742

The bundled files are byte-for-byte as distributed; MLBreak removes 415 exact duplicates **in memory** before the
split (see `src/mlbreak/text/data/sms_spam_collection/NOTICE.md`). The corpus contains real-world spam and
informal personal messages and is used only to train and attack a local classifier.

**Adding a dataset.** Bundle it under the package with a `NOTICE.md` (source, license, retrieval date, file
hashes), verify its SHA-256 at load time, de-duplicate before splitting, record split hashes in evidence, and add
it to this table. Do not add data whose license forbids redistribution.
