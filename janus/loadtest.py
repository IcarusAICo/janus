"""Closed-loop HTTP load test for the local server: N client threads each POST the request(s) back to back for S seconds.

    python -m janus.loadtest --url http://127.0.0.1:8080 --clients 16 --seconds 30 --requests examples/request.json

Reports QPS, latency p50/p95/p99 over the successful requests (ms), error counts by status, input tokens per second
and the server's batching stats (`GET /v1/models` `stats`) as one JSON object. Stdlib only (threading + urllib):
every request opens its own connection, so the numbers include connection setup, about 0.1 ms on localhost.
`--requests` is one JSON request body or a JSONL file of them (clients cycle through it); the `model` field is
filled from `--model`, else from the first entry of `GET /v1/models`. The bearer token comes from
`--token`, else `JANUS_SERVER_TOKEN`.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_bodies(path):
    text = Path(path).read_text()
    try:
        value = json.loads(text)
    except ValueError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    return value if isinstance(value, list) else [value]


def percentile(sorted_values, p):
    """Nearest-rank percentile of an ascending list; None when empty."""
    return sorted_values[min(len(sorted_values) - 1, int(p * len(sorted_values)))] if sorted_values else None


def models(url, headers):
    with urlopen(Request(url + "/v1/models", headers=headers), timeout=60) as response:
        return json.load(response)


def run(url, clients, seconds, bodies, token=None, model=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if model is None:
        model = models(url, headers)["models"][0]["name"]
    payloads = [json.dumps({**body, "model": model}).encode() for body in bodies]
    latencies, errors, tokens, lock = [], {}, [0], threading.Lock()
    deadline = time.monotonic() + seconds

    def client(index):
        while time.monotonic() < deadline:
            started = time.perf_counter()
            try:
                with urlopen(Request(url + "/v1/systemone", data=payloads[index % len(payloads)], headers=headers), timeout=60) as response:
                    status, used = 200, json.load(response)["usage"]["input_tokens"]
            except HTTPError as error:
                status, used = error.code, 0
                error.read()
            except (URLError, OSError) as error:  # connection refused, timeout
                status, used = type(error).__name__, 0
            elapsed = (time.perf_counter() - started) * 1e3
            with lock:
                if status == 200:
                    latencies.append(elapsed)
                    tokens[0] += used
                else:
                    errors[str(status)] = errors.get(str(status), 0) + 1
            index += clients

    threads = [threading.Thread(target=client, args=(i,)) for i in range(clients)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    ordered = sorted(latencies)
    report = {"url": url, "model": model, "clients": clients, "seconds": round(wall, 3), "requests": len(ordered), "errors": errors,
              "qps": len(ordered) / wall, "p50_ms": percentile(ordered, .5), "p95_ms": percentile(ordered, .95),
              "p99_ms": percentile(ordered, .99), "mean_ms": statistics.fmean(ordered) if ordered else None,
              "input_tokens_per_s": tokens[0] / wall}
    try:
        report["server_stats"] = models(url, headers).get("stats")
    except (HTTPError, URLError, OSError, ValueError, KeyError):
        pass
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--requests", default="examples/request.json", help="A JSON request body or a JSONL file of them")
    parser.add_argument("--model", help="Sent as the `model` field; default: the first entry of GET /v1/models")
    parser.add_argument("--token", default=os.environ.get("JANUS_SERVER_TOKEN"), help="Bearer token (default: $JANUS_SERVER_TOKEN)")
    args = parser.parse_args(argv)
    report = run(args.url, args.clients, args.seconds, load_bodies(args.requests), args.token, args.model)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
