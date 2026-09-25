"""Usage: python examples/patterns/extract_candidate.py CHECKPOINT
A regex over-finds candidate spans; the model only chooses among them (or none), so the value is verbatim."""
import re
import sys

from janus.patterns import LocalEvaluator, extract_candidate

state = "Invoice 4471. Subtotal $118.00, shipping $6.50, tax $9.44. Amount due: $133.94. Pay by 2026-10-01."
candidates = re.findall(r"\$\d+(?:\.\d{2})?", state)
out = extract_candidate(LocalEvaluator(sys.argv[1]), state, "total amount due", candidates)
print(f"candidates={candidates}")
print(f"value={out['value']!r} confidence={out['confidence']:.3f}")
