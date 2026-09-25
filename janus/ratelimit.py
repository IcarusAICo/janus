"""Token-bucket rate limiter keyed by caller (bearer token or client address), stdlib only.

Each key refills at `rate` tokens per second up to `burst`; `take(key)` spends one token and returns 0.0 when the
call is allowed, else the seconds until a token is available (the server turns that into 429 + Retry-After).
"""

import threading
import time


class TokenBucket:
    # ponytail: keys are never evicted; there is one per bearer token (one) or per client address, so the dict stays
    # small on a local server. Upgrade path: drop entries whose bucket has been full for a while, inside `take`.
    def __init__(self, rate, burst, clock=time.monotonic):
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self.rate, self.burst, self.clock = float(rate), float(burst), clock
        self.buckets = {}  # key -> [tokens, last refill time]
        self.lock = threading.Lock()

    def take(self, key):
        now = self.clock()
        with self.lock:
            tokens, last = self.buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.:
                self.buckets[key] = [tokens - 1., now]
                return 0.
            self.buckets[key] = [tokens, now]
            return (1. - tokens) / self.rate

    @classmethod
    def per_minute(cls, rpm, clock=time.monotonic):
        """`rpm` sustained, with a burst of one second's worth (at least one request)."""
        return cls(rpm / 60., max(1., rpm / 60.), clock)
