# Demos

Live System One demos. They call TypeSafe Jev (or this repo's local `/v1/systemone` server) over HTTP. They do not load `janus.model`.

```bash
python -m pip install -e '.[demos]'
# keys: TYPESAFE_API_KEY and OPENAI_API_KEY in env.sh (never committed)
```

Artifacts go to `demos/artifacts/` (gitignored). Every demo accepts `--backend typesafe|local` and `--base-url`.

## Doom

Live HUD at http://127.0.0.1:8765. Type standing orders or edit the JSON `state` while the game runs.

```bash
python -m demos.doom.server --backend typesafe
python -m demos.doom.record --seconds 90 --script demos/doom/scripts/orders_midrun.json
```

Freedoom only (BSD). No commercial Doom WAD.

## Wikiracing

Four-way race: Jev vs GPT-5.6 Luna / Terra / Sol. Live at http://127.0.0.1:8767.

```bash
python -m demos.wikiracing.server --challenge baseball-sun
python -m demos.wikiracing.record --challenge baseball-sun
```

Wikipedia excerpts are CC BY-SA 4.0.

## Browser Use (Google Flights)

Vendored [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT). Own `uv` env; needs Chrome + Browser Harness. This host has no sudo for Google Chrome, so the wrapper starts Chrome for Testing under Xvfb and attaches over CDP (`BU_CDP_URL=http://127.0.0.1:9222`). Do not attach to the kiosk Chromium on port 9202.

```bash
python -m demos.browser.chrome          # start Chrome for Testing + print CDP URL
python -m demos.browser.record --env-file env.sh
```

```bash
cd demos/browser
uv sync
uv run browser-harness --doctor         # needs BU_CDP_URL if Chrome for Testing is already up
```

## Choice benches (jevlike + MMLU-Pro)

JSONL Choice evals for TypeSafe Jev and GPT-5.6. Numbers: [artifacts/bench.md](artifacts/bench.md).

```bash
python -m demos.bench.run --task mmlu-pro --backend typesafe --limit 1000
python -m demos.bench.run --task mmlu-pro --backend gpt --model gpt-5.6-luna --limit 100
python -m demos.bench.run --task synthetic --backend typesafe
scripts/jevlike_recreate.sh    # GPU: recreate jevlike README train/eval
```
