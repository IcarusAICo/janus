"""Usage: python examples/patterns/speculative_fanout.py CHECKPOINT
One call asks every triage question; the code below only reads the ones the category makes relevant."""
import sys

from janus.patterns import LocalEvaluator, speculative_fanout

state = "The app crashes every time I open settings. Here are the steps: 1) open app 2) tap settings. Please fix!!!"
questions = {"category": {"type": "choice", "instructions": "What kind of ticket is this?",
                          "criteria": {"bug_report": "A defect", "billing": "Billing or refunds", "other": "Anything else"}},
             "bug_severity": {"type": "score", "instructions": "How severe is the bug?", "criteria": ["minor", "major", "critical"]},
             "has_repro_steps": {"type": "noul", "instructions": "Does the ticket give steps to reproduce?"},
             "refund_requested": {"type": "noul", "instructions": "Is a refund requested?"}}
answers = speculative_fanout(LocalEvaluator(sys.argv[1]), state, questions)
for qid, a in answers.items():
    print(f"{qid}: {a.probabilities}")
if answers["category"].choice == "bug_report":
    print("escalate" if answers["bug_severity"].score > 1.5 and answers["has_repro_steps"].noul > .6 else "queue for engineering")
else:
    print("not a bug; severity and repro answers ignored")
