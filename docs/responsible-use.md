# Responsible use

MLBreak is an **intentionally vulnerable** educational and defensive-research tool.

* Use it to learn how evasion attacks work, how to measure them honestly, and how to evaluate
  defenses against adaptive adversaries.
* Run it locally. The server binds to `127.0.0.1` by default; do not expose it to untrusted networks.
* The techniques here are well-known, published attacks applied to a synthetic toy problem. Do not
  apply them to systems you do not own or are not explicitly authorised to test.
* Report results honestly. Include failed attacks, the denominators, and whether an evaluation was a
  replay or an adaptive attack. A defense evaluated only against replayed perturbations has not been
  shown to be robust.
* If you extend MLBreak with real datasets or models, make sure you have the right to use them, and
  keep the no-upload / no-pickle posture: add them as bundled, reviewed, approved configurations.
