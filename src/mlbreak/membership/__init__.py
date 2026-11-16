"""ML04:2023 Membership Inference — training-set membership inference laboratory (synthetic data only).

The attack code in :mod:`.attacks` and :mod:`.shadow` only receives an :class:`~.attacks.AttackerKnowledge` and a
counting query function. Which candidate is a training member is known only to :mod:`.engine` (the evaluation
engine), which attaches the ground truth after the attack returns.
"""
