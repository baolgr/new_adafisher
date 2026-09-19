"""The shared benchmark harness: everything a bench does not choose for itself.

One training loop (``loop``), one optimizer factory (``optimizers``), one record and report schema
(``records``), one data pipeline (``data``), one pair of learning-rate schedules (``schedules``),
one checkpointer (``checkpoints``) and one runner (``runner``). A model's ``bench.py`` names the
pieces it wants and adds nothing of its own.
"""
