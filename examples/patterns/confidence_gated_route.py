"""Usage: python examples/patterns/confidence_gated_route.py CHECKPOINT [THRESHOLD]
Act on the classification only when its confidence clears the threshold; otherwise hand off."""
import sys

from janus.patterns import LocalEvaluator, confidence_gated_route

state = "Please move 500 from savings to checking today."
question = {"type": "choice", "instructions": "What does the customer want done?",
            "criteria": {"check_balance": "See a balance", "approve_transfer": "Move money between accounts", "other": "Something else"}}
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else .85
out = confidence_gated_route(LocalEvaluator(sys.argv[1]), state, question, threshold, lambda answer: "ask_user_to_confirm")
print(f"confidence {out['confidence']:.3f} threshold {threshold}: accepted={out['accepted']} -> {out['result']}")
