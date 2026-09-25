"""Usage: python examples/patterns/composite_score.py CHECKPOINT
Several Score rubrics on one state, combined by weights in plain code."""
import sys

from janus.patterns import LocalEvaluator, composite_score

state = "Six years of Python; led a team of four; designed the event pipeline; some Go and SQL."
levels = ["none", "basic", "working", "strong", "expert"]
rubrics = {name: {"instructions": f"Rate the candidate's {name.replace('_', ' ')}.", "criteria": levels}
           for name in ("python_depth", "leadership", "architecture", "general")}
weights = {"python_depth": .4, "leadership": .1, "architecture": .4, "general": .1}
out = composite_score(LocalEvaluator(sys.argv[1]), state, rubrics, weights)
for name, a in out["answers"].items():
    print(f"{name}: expected level {a.score:.2f} of {len(levels) - 1}")
print(f"composite {out['score']:.3f}")
