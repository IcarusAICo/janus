"""Deterministic renderers and the Luna paraphrase path with round-trip acceptance."""

import json

from .worlds import FEATURES

EVIDENCE_SCHEMAS = {
    "rel": {"type": "object", "additionalProperties": False, "required": ["customer_request", "reference"],
            "properties": {"customer_request": {"type": "string"}, "reference": {"type": ["string", "null"]}}},
    "ord": {"type": "object", "additionalProperties": False,
            "required": ["users_affected", "duration_minutes", "data_loss", "workaround_available"],
            "properties": {"users_affected": {"type": "integer"}, "duration_minutes": {"type": "integer"},
                           "data_loss": {"type": "boolean"}, "workaround_available": {"type": "boolean"}}},
    "post": {"type": "object", "additionalProperties": False, "required": ["message_features"],
             "properties": {"message_features": {"type": "object", "additionalProperties": False, "required": list(FEATURES),
                                                 "properties": {name: {"type": "boolean"} for name in FEATURES}}}}}
REWRITE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["text"], "properties": {"text": {"type": "string"}}}


def evidence(world):
    family = world["family"]
    if family == "rel":
        return {"customer_request": world_request_text(world), "reference": world["reference"]}
    if family == "ord":
        return {"users_affected": world["users"], "duration_minutes": world["minutes"],
                "data_loss": world["data_loss"], "workaround_available": world["workaround"]}
    if family == "post":
        return {"message_features": dict(world["features"])}
    raise ValueError(family)


def world_request_text(world):
    kind, reference = world["kind"], world["reference"]
    return {"second_cheapest": "The customer wants the second cheapest option.",
            "median_rating": "The customer wants the option whose rating is the median of all the options.",
            "closest_price_to_named": f"The customer wants the option whose price is closest to the option named {reference}.",
            "cheapest_above_named_rating": f"The customer wants the cheapest option that is rated higher than the option named {reference}."}[kind]


class Paraphraser:
    REWRITE = ("Rewrite the text in a different register and sentence structure. Keep every fact, number, name, and "
               "yes/no detail exactly as given. Do not add or remove information. Return JSON with one field, text.")
    EXTRACT = "Extract the requested fields from the text exactly. Return JSON matching the schema; do not guess missing fields."

    def __init__(self, completer):
        self.completer = completer
        self.accepted = self.rejected = 0

    def paraphrase(self, world, prose_state):
        family = world["family"]
        rewritten, _, _ = self.completer.complete(self.REWRITE, "REWRITE: " + prose_state, "rewrite", REWRITE_SCHEMA)
        text = rewritten["text"]
        extracted, _, _ = self.completer.complete(self.EXTRACT, text, f"evidence_{family}", EVIDENCE_SCHEMAS[family])
        if extracted == evidence(world):
            self.accepted += 1
            return text
        self.rejected += 1
        return None
