"""Usage: python examples/patterns/intent_route.py CHECKPOINT
Choice over intents plus a none option; None means no listed handler applies."""
import sys

from janus.patterns import LocalEvaluator, intent_route

intents = {"order_status": "Where an order is", "product_question": "A question about a product",
           "return_exchange": "Returning or exchanging an item", "complaint": "A complaint"}
out = intent_route(LocalEvaluator(sys.argv[1]), "Where is my package? It was due Monday.", intents,
                   "None of the listed intents")
print(f"intent={out['intent']} probability={out['probability']:.3f}")
print({k: round(v, 3) for k, v in out["answer"].probabilities.items()})
