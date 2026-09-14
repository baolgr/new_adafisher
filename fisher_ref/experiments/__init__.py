"""Measurement drivers for the Fisher-drift campaign — not tests, not part of any lot's API.

Each script here answers one protocol question with numbers, and its answer is recorded in
``docs/reports/plan_exp_draft.md``. They are kept in the repository because they are the *evidence*
behind a protocol decision, and because a protocol decision that cannot be re-measured is an
assumption.

Run them by hand; none is fast enough for the test suite (10-15 min on CPU):

    PYTHONPATH=src:. .venv/bin/python -u fisher_ref/experiments/rewarm_fidelity.py
    PYTHONPATH=src:. .venv/bin/python -u fisher_ref/experiments/identity_seed_residual.py
"""
