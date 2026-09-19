"""Measurement drivers: one protocol question each, answered with printed numbers.

These are not tests and not part of any module's API. Each script here settles one protocol
decision -- how long a re-warm has to be, how large the damping is next to the curvature, whether a
validation curve's wobble is signal -- and its answer is pasted back into its own module docstring
so the decision can be re-read without re-running it. They are kept in the repository because a
protocol decision that cannot be re-measured is an assumption.

Conventions shared by all of them:

* **no command line.** Every knob is a module-level constant, overridable by an environment
  variable with a default, so a batch script can set them without the file growing an argument
  parser it does not want. Each script's docstring lists its own variables.
* **run them by hand.** None is fast enough for the test suite; several take tens of minutes on a
  laptop and some need a GPU::

      PYTHONPATH=src:. .venv/bin/python -u fisher_ref/experiments/<name>.py

* **outputs** go to ``fisher_ref/outputs/`` as JSON where a later script reads them, and to standard
  output otherwise.

Two of these scripts reuse a sibling's machinery (``curvature_max_per_layer.py`` and
``early_curvature.py`` both import from ``lambda_vs_curvature``). They import it by its full
package path and put the repository root on the import path first, so both invocation forms above
work.
"""
