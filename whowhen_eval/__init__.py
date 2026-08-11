"""Who&When Pro — failure-attribution evaluation harness.

Given the transcript of a multi-agent (or single-agent) system that failed
a task, an LLM judge must answer three questions:

* **Who** — which agent took the turn that first introduced the decisive
  error?
* **When** — at which step coordinate did that turn occur?
* **What** — which taxonomy error mode does it instantiate?

The package ships:

* :mod:`whowhen_eval.render` — one renderer per released framework, turning
  a release JSON trace into the transcript the judge sees.
* :mod:`whowhen_eval.prompts` — the all-at-once prompt builder.
* :mod:`whowhen_eval.parse` — a lenient parser for the judge's response.
* :mod:`whowhen_eval.score` — per-framework scoring of the three axes.
* :mod:`whowhen_eval.run` — the CLI runner (``python -m whowhen_eval.run``).
* :mod:`whowhen_eval.leaderboard` — the Who/When/What/All metric.

The taxonomy is *not* baked into this package: it is read at runtime from
``taxonomy.yaml`` at the root of the dataset checkout you point
``--data-root`` at, so the prompt always enumerates exactly the modes the
data can be labelled with.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
