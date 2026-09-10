"""The shared benchmark harness (``docs/reports/plan_exp_step1.md`` §3).

One training loop (``loop``), one optimizer factory (``optimizers``), one record/report schema
(``records``), one data pipeline (``data``), one checkpointer (``checkpoints``) and one runner
(``runner``) — the merge of what lots 1, 7 and 8 grew as three separate flat modules.
"""
