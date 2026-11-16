"""ML03:2023 Model Inversion — private-attribute inversion laboratory (synthetic data only).

The attack code in :mod:`.attacks` only receives an :class:`~.attacks.AttackerView` and a counting query
function: ground truth, model parameters and private training records stay inside :mod:`.engine` (the
evaluation engine) and :mod:`.defenses` (the prediction service).
"""
