"""The public TypeSafe SDKs against our local /v1/systemone server (tiny CPU checkpoint, free port).

Python: `typesafe_sdk` (skipped when not installed). JavaScript: `@typesafe-ai/sdk` via `node` and
`tests/sdk-js/conformance.mjs` (skipped when node or `tests/sdk-js/node_modules` is missing; run
`npm install` in tests/sdk-js). Versions tested are recorded in docs/api-compat.md.
"""

import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading

import pytest
import torch

from janus.model import DecisionModel, ModelConfig
from janus.server import make_server
from janus.training import checkpoint

TOKEN = "test-token-never-print"
EXAMPLE = json.load(open("examples/request.json"))
JS_DIR = Path(__file__).parent / "sdk-js"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    torch.set_num_threads(2)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, layers=1, head_rank=8, max_tokens=1024))
    checkpoint(model, path, {"step": 0})
    server = make_server(path, port=0, model_id="janus-local-test", queue_size=2)
    server.token = TOKEN
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{server.server_address[1]}", "server": server}
    server.shutdown()


@pytest.fixture
def sdk():
    return pytest.importorskip("typesafe_sdk")


@pytest.fixture
def client(sdk, served):
    with sdk.TypeSafeClient(api_key=TOKEN, base_url=served["url"], model="janus-local-test",
                            retry=sdk.RetryPolicy(max_retries=0)) as client:
        yield client


def test_python_sdk_three_primitives_and_typed_accessors(sdk, client):
    result = client.system_one(state=EXAMPLE["state"], questions={
        "route": sdk.Choice(instructions=EXAMPLE["questions"]["route"]["instructions"], criteria=EXAMPLE["questions"]["route"]["criteria"]),
        "frustration": sdk.Score(instructions=EXAMPLE["questions"]["frustration"]["instructions"], criteria=EXAMPLE["questions"]["frustration"]["criteria"]),
        "transfer_problem": sdk.Noul(instructions=EXAMPLE["questions"]["transfer_problem"]["instructions"])})
    assert isinstance(result, sdk.SystemOneResponse)
    assert result.model == "janus-local-test"
    assert result.usage.input_tokens > 0 and result.usage.output_tokens == 0
    assert set(result.answers) == set(EXAMPLE["questions"])
    assert set(result.choices) == {"route"} and set(result.scores) == {"frustration"} and set(result.nouls) == {"transfer_problem"}
    route = result.choices["route"]
    assert set(route.probabilities) == {"payments", "access", "other"}
    assert route.choice == max(route.probabilities, key=route.probabilities.get)
    assert route.confidence == max(route.probabilities.values())
    score = result.scores["frustration"]
    assert score.legend == {0: "calm", 1: "concerned", 2: "very frustrated"}  # SDK coerces our string keys to int
    assert list(score.probabilities) == [0, 1, 2]
    assert abs(score.score - sum(i * p for i, p in score.probabilities.items())) < 1e-9
    assert score.confidence == max(score.probabilities.values())
    assert 0 <= result.nouls["transfer_problem"].noul <= 1
    assert len(result.request_id) == 32  # our x-typesafe-request-id header
    assert result.raw_http_response.status_code == 200


def test_python_sdk_dict_questions_and_null_choice_criteria(client):
    result = client.system_one(state={"message": "hello"}, questions={
        "tone": {"type": "choice", "instructions": "Tone?", "criteria": {"calm": None, "angry": None}}})
    assert result.choices["tone"].choice in {"calm", "angry"}


def test_python_sdk_validation_errors(sdk, client):
    with pytest.raises(sdk.TypeSafeUnprocessableEntityError) as caught:
        client.system_one(state="x", questions={"q": sdk.Choice(instructions="i", criteria={})})
    assert caught.value.status == 422 and caught.value.body["error"]["type"] == "validation_error"
    assert "1" in str(caught.value)  # our message reaches the SDK's error text
    with pytest.raises(sdk.TypeSafeUnprocessableEntityError):
        client.system_one(state="x", questions={"q": {"type": "score", "instructions": "i", "criteria": ["only"]}})
    with pytest.raises(sdk.TypeSafeError):  # rejected client-side, never sent
        client.system_one(state="x", questions={})


def test_python_sdk_auth_error(sdk, served):
    with sdk.TypeSafeClient(api_key="wrong", base_url=served["url"], retry=sdk.RetryPolicy(max_retries=0)) as client:
        with pytest.raises(sdk.TypeSafeAuthenticationError) as caught:
            client.system_one(state="x", questions={"q": sdk.Noul(instructions="i")})
        assert caught.value.status == 401
        with pytest.raises(sdk.TypeSafeAuthenticationError):
            client.models.list()


def test_python_sdk_models_list(sdk, client):
    listing = client.models.list()
    assert isinstance(listing, sdk.ListModelsResponse)
    assert [m.name for m in listing.models] == ["janus-local-test"]
    assert len(listing.models[0].release_date) == 10
    assert "limits" in listing.raw_http_response.json()  # our extra field is ignored, not rejected


def test_python_sdk_overloaded_maps_to_5xx_error_with_retry_after(sdk, client, served):
    worker = served["server"].worker
    original = worker.queue
    worker.queue = queue.Queue(maxsize=1)
    worker.queue.put(None)
    try:
        with pytest.raises(sdk.TypeSafeInternalServerError) as caught:  # 529 (the docs' overload status) is a 5xx to the SDK
            client.system_one(state="x", questions={"q": sdk.Noul(instructions="i")})
    finally:
        worker.queue = original
    assert caught.value.status == 529 and caught.value.headers["retry-after"] == "1"
    # The SDK's default policy retries 5xx; with one retry the restored queue answers.
    result = client.system_one(state="x", questions={"q": sdk.Noul(instructions="i")}, retry=sdk.RetryPolicy(max_retries=1, backoff_initial=0))
    assert 0 <= result.nouls["q"].noul <= 1


def test_js_sdk_three_primitives_errors_and_models(served):
    node = shutil.which("node")
    if node is None or not (JS_DIR / "node_modules" / "@typesafe-ai" / "sdk").is_dir():
        pytest.skip("node or tests/sdk-js/node_modules/@typesafe-ai/sdk missing (npm install in tests/sdk-js)")
    env = {**os.environ, "TYPESAFE_API_KEY": TOKEN, "TYPESAFE_BASE_URL": served["url"], "TYPESAFE_DEFAULT_MODEL": "janus-local-test"}
    run = subprocess.run([node, str(JS_DIR / "conformance.mjs")], env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    result = out["result"]
    assert result["model"] == "janus-local-test" and result["usage"]["output_tokens"] == 0
    assert result["answers"]["route"]["choice"] in {"payments", "access", "other"}
    assert result["answers"]["frustration"]["legend"]["2"] == "very frustrated"
    assert 0 <= result["answers"]["transfer_problem"]["noul"] <= 1
    assert [m["name"] for m in out["models"]] == ["janus-local-test"] and set(out["models"][0]) == {"name", "description", "release_date"}
    assert out["auth"] == {"name": "AuthenticationError", "status": 401}
    assert out["invalid"] == {"name": "UnprocessableEntityError", "status": 422}
    assert len(out["request_id"]) == 32
