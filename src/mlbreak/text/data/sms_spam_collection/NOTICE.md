# SMS Spam Collection v.1 — bundled copy

| | |
|---|---|
| Dataset | SMS Spam Collection v.1 |
| Authors | Tiago A. Almeida (UFSCar) and José María Gómez Hidalgo |
| Source | UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> |
| DOI | [10.24432/C5CC84](https://doi.org/10.24432/C5CC84) |
| License | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) as published by UCI; the authors' `readme` (included unchanged) states the corpus is provided free and without limitations other than their copyright and the warranty disclaimer |
| Retrieved | 2026-11-12, `sms+spam+collection.zip` (sha256 `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3`) |
| Files | `SMSSpamCollection` (UTF-8, tab-separated `label<TAB>text`, 5,574 lines) and `readme` (ISO-8859-1), both byte-for-byte as distributed |

**Citation.** Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). *Contributions to the Study of SMS
Spam Filtering: New Collection and Results.* Proceedings of the 2011 ACM Symposium on Document Engineering
(DocEng'11). <https://doi.org/10.1145/2034691.2034742>

**Changes.** None to the files. MLBreak verifies the SHA-256 of `SMSSpamCollection` at load time
(`src/mlbreak/text/data.py`), then removes exact duplicates (case- and whitespace-insensitive) **in memory**
before the train/test split, so no message appears in both splits.

**Content warning.** The corpus contains real-world spam (premium-rate numbers, prize scams) and informal
personal messages. It is used here only to train and attack a local spam classifier.
