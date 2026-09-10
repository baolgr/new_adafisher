"""Training benches: one shared harness in ``common/``, one folder per tested model.

Layout and rules are ``docs/reports/plan_exp_step1.md`` §1-§2. A model folder holds a ``model.py``
only if it introduces a new architecture (D1); its ``bench.py`` is a ``Benchmark`` literal plus
``main(BENCH)`` and contains no control flow (D3).
"""
