"""Cached, cost-accounted structured outputs over the OpenAI Responses API."""

import hashlib
import json
from pathlib import Path

from ..remote import _atomic_json, load_env_key

PRICES = {"gpt-5.6-luna": (.20, 1.20), "gpt-5.6-terra": (2.00, 12.00), "gpt-5.6-sol": (5.00, 30.00)}


class StructuredCompleter:
    def __init__(self, model="gpt-5.6-luna", env_file="env.sh", cache_dir=".cache/synth/openai", effort="none", client=None):
        if model not in PRICES:
            raise ValueError(f"Unknown model {model!r}; known: {sorted(PRICES)}")
        self.model, self.effort = model, effort
        self.cache_dir = Path(cache_dir)
        if client is None:
            import openai
            client = openai.OpenAI(api_key=load_env_key("OPENAI_API_KEY", env_file))
        self.client = client
        self.new_usage = {"input_tokens": 0, "output_tokens": 0}
        self.cached_usage = {"input_tokens": 0, "output_tokens": 0}
        self.calls = {"new": 0, "cached": 0}

    def complete(self, instructions, user, schema_name, schema):
        body = {"model": self.model, "instructions": instructions, "input": user,
                "reasoning": {"effort": self.effort},
                "text": {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
                "store": False}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            envelope = json.loads(path.read_text())
            self.calls["cached"] += 1
            for key in self.cached_usage:
                self.cached_usage[key] += int(envelope["usage"][key])
            return json.loads(envelope["output_text"]), envelope["usage"], True
        response = self.client.responses.create(**body)
        usage = {"input_tokens": int(response.usage.input_tokens), "output_tokens": int(response.usage.output_tokens)}
        envelope = {"body": body, "output_text": response.output_text, "usage": usage}
        _atomic_json(path, envelope)
        for key in usage:
            self.new_usage[key] += usage[key]
        self.calls["new"] += 1
        return json.loads(response.output_text), usage, False

    def cost_usd(self, include_cached=False):
        """Spend on this run's new calls; with include_cached, what every call used by this run once cost."""
        price_in, price_out = PRICES[self.model]
        usage = dict(self.new_usage)
        if include_cached:
            usage = {key: usage[key] + self.cached_usage[key] for key in usage}
        return usage["input_tokens"] * price_in / 1e6 + usage["output_tokens"] * price_out / 1e6
