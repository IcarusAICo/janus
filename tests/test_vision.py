"""Image states (janus.vision) on the Qwen3.5 hybrid runner: equivalence with the stock multimodal forward, isolation,
text-only states unchanged, the schema round trip, the hash, the server parser and the local evaluator. CPU, fp32,
tiny random weights."""

import base64
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request as HTTPRequest, urlopen

from PIL import Image as PILImage
import pytest
import torch

from janus.data import state_hash
from janus.hybrid import TINY, TINY_VL_VOCAB, HybridBackbone
from janus.model import DecisionModel, ModelConfig
from janus.packing import ByteTokenizer, pack_request
from janus.patterns import LocalEvaluator
from janus.schema import ImageState, Request
from janus.server import make_server
from janus.training import checkpoint
from janus.vision import Vision, state_embeddings, tiny_full_model, tiny_processor

QUESTIONS = {
    "color": {"type": "choice", "instructions": "Color?", "criteria": {"r": "red", "b": "blue", "g": "green"}, "target": [1, 0, 0]},
    "red": {"type": "noul", "instructions": "Is it red?", "target": [0, 1]},
    "intensity": {"type": "score", "instructions": "Intensity?", "criteria": ["low", "medium", "high"], "target": [0, 0, 1]}}


def png(width, height, seed=0):
    torch.manual_seed(seed)
    pixels = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8).numpy()
    buffer = io.BytesIO()
    PILImage.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def b64(data):
    return {"base64": base64.b64encode(data).decode(), "media_type": "image/png"}


def request(text="A card: [image:0] is it red?", images=(png(24, 16),)):
    return Request.from_dict({"state": {"text": text, "images": [b64(i) for i in images]}, "questions": QUESTIONS})


def full_and_backbone(seed=0, attention="sdpa"):
    torch.manual_seed(seed)
    full = tiny_full_model(HybridBackbone._tiny_config(64, 4, 512, attention, TINY_VL_VOCAB), attention).eval()
    vision = Vision(full.model.visual, tiny_processor(), full.config.image_token_id,
                    full.config.vision_start_token_id, full.config.vision_end_token_id)
    return full, HybridBackbone(full.model.language_model, ByteTokenizer(), attention, vision=vision).eval()


def paths(packed):
    state = list(range(packed.state_length))
    for branch in packed.branches:
        block = list(range(branch.start, branch.end))
        for start, end in branch.leaves:
            yield branch, (start, end), state + block + list(range(start, end))


@pytest.mark.parametrize("tree_positions", ["shared", "continue"])
def test_runner_matches_the_stock_multimodal_forward_on_every_root_to_leaf_path(tree_positions):
    """Under `shared` every path is contiguous, so the stock model computes its own 3-axis positions (get_rope_index);
    under `continue` the packed positions are passed explicitly. Hidden states agree at every block and leaf position."""
    full, bb = full_and_backbone()
    packed = pack_request(request(), ByteTokenizer(), "tree", tree_positions=tree_positions, vision=bb.vision)
    pixel_values, grid_thw = packed.images
    n_image = int((packed.input_ids[0] == bb.vision.image_token_id).sum())
    assert grid_thw.tolist() == [[1, 4, 6]] and n_image == 6 and packed.position_ids.shape[0] == 3
    assert packed.state_length > int(packed.position_ids[0, :packed.state_length].max()) + 1  # the image is 6 tokens, 3 positions wide
    with torch.no_grad():
        ours = bb.encode(packed)
        for branch, (start, end), positions in paths(packed):
            ids = packed.input_ids[:, positions]
            kwargs = {"position_ids": packed.position_ids[:, None, positions]} if tree_positions == "continue" else \
                     {"mm_token_type_ids": (ids == bb.vision.image_token_id).int()}
            reference = full.model(input_ids=ids, pixel_values=pixel_values, image_grid_thw=grid_thw, use_cache=False, **kwargs).last_hidden_state[0]
            torch.testing.assert_close(ours[start:end], reference[-(end - start):], atol=1e-4, rtol=1e-4)
            length = branch.end - branch.start
            torch.testing.assert_close(ours[branch.start:branch.end], reference[packed.state_length:packed.state_length + length], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(ours[:packed.state_length], reference[:packed.state_length], atol=1e-4, rtol=1e-4)


def test_state_positions_and_layout_follow_the_stock_processor_and_rope_index():
    full, bb = full_and_backbone()
    packed = pack_request(request("[image:0] then text", (png(16, 16),)), ByteTokenizer(), "tree", vision=bb.vision)
    ids = packed.input_ids[:, :packed.state_length]
    v = bb.vision
    head = ByteTokenizer().encode("State:\n")
    assert ids[0, len(head)].item() == v.vision_start_token_id and ids[0, len(head) + 5].item() == v.vision_end_token_id
    assert (ids[0, len(head) + 1:len(head) + 5] == v.image_token_id).all()
    stock, _ = full.model.get_rope_index(ids, (ids == v.image_token_id).int(), packed.images[1])
    assert torch.equal(packed.position_ids[:, :packed.state_length], stock[:, 0])
    s = len(head) + 1
    assert packed.position_ids[:, s:s + 4].tolist() == [[s] * 4, [s, s, s + 1, s + 1], [s, s + 1, s, s + 1]]
    assert packed.position_ids[0, s + 4].item() == s + 2  # the text after a 2x2 grid continues at start + 2
    # The branches continue from the position after the state's last position, on all three axes.
    block = packed.branches[0]
    assert (packed.position_ids[:, block.start] == packed.position_ids[0, packed.state_length - 1] + 1).all()
    embeds, positions, mask = state_embeddings(bb, request("[image:0] then text", (png(16, 16),)).state, ByteTokenizer())
    assert embeds.shape == (1, packed.state_length - len(head) - 1, 64) and positions.shape[0] == 3 and int(mask.sum()) == 4


def test_two_images_and_marker_rules():
    _, bb = full_and_backbone()
    two = request("first [image:1] second [image:0] end", (png(16, 16, 1), png(32, 16, 2)))
    packed = pack_request(two, ByteTokenizer(), "tree", vision=bb.vision)
    ids = packed.input_ids[0, :packed.state_length].tolist()
    assert packed.images[1].tolist() == [[1, 4, 4], [1, 4, 8]] and ids.count(bb.vision.image_token_id) == 4 + 8
    # image 1 (8 tokens) comes first in the text, and the features are placed per marker, not per list order.
    starts = [i for i, t in enumerate(ids) if t == bb.vision.vision_start_token_id]
    assert ids[starts[0] + 1:starts[0] + 9] == [bb.vision.image_token_id] * 8 and ids[starts[0] + 9] == bb.vision.vision_end_token_id
    with torch.no_grad():
        hidden = bb.encode(packed)
    assert torch.isfinite(hidden).all()
    # Without markers every image precedes the text.
    plain = pack_request(request("just text", (png(16, 16, 1), png(32, 16, 2))), ByteTokenizer(), "tree", vision=bb.vision)
    head = len(ByteTokenizer().encode("State:\n"))
    assert plain.input_ids[0, head].item() == bb.vision.vision_start_token_id
    # A repeated marker places the image once; an unmentioned image goes first; a missing image is an error.
    two = (png(16, 16, 1), png(32, 16, 2))
    starts = lambda packed: (packed.input_ids[0] == bb.vision.vision_start_token_id).sum().item()
    assert starts(pack_request(request("[image:0] [image:0]", two), ByteTokenizer(), "tree", vision=bb.vision)) == 2
    assert starts(pack_request(request("[image:1]", two), ByteTokenizer(), "tree", vision=bb.vision)) == 2
    with pytest.raises(ValueError, match="missing image"):
        pack_request(request("[image:0] [image:2]", two), ByteTokenizer(), "tree", vision=bb.vision)


def test_sibling_isolation_with_images():
    _, bb = full_and_backbone()
    a = pack_request(request(), ByteTokenizer(), "tree", vision=bb.vision)
    changed = {"state": {"text": "A card: [image:0] is it red?", "images": [b64(png(24, 16))]}, "questions": json.loads(json.dumps(QUESTIONS))}
    changed["questions"]["intensity"]["instructions"] = "Secret blue! " * 6
    b = pack_request(Request.from_dict(changed), ByteTokenizer(), "tree", vision=bb.vision)
    boundary = a.branches[2].start
    with torch.no_grad():
        ha, hb = bb.encode(a), bb.encode(b)
    torch.testing.assert_close(hb[:boundary], ha[:boundary], atol=1e-6, rtol=0)
    assert not torch.allclose(hb[-1], ha[-1], atol=1e-3)
    # The image reaches every leaf: other pixels, other leaf outputs.
    c = pack_request(request(images=(png(24, 16, seed=5),)), ByteTokenizer(), "tree", vision=bb.vision)
    with torch.no_grad():
        hc = bb.encode(c)
    for branch in a.branches:
        for start, end in branch.leaves:
            assert not torch.allclose(hc[end - 1], ha[end - 1], atol=1e-4)


def test_text_only_states_are_unchanged_by_the_image_path():
    _, bb = full_and_backbone()
    text_only = Request.from_dict({"state": "A red card.", "questions": QUESTIONS})
    with_vision = pack_request(text_only, ByteTokenizer(), "tree", tree_positions="continue", vision=bb.vision)
    without = pack_request(text_only, ByteTokenizer(), "tree", tree_positions="continue")
    assert with_vision.images is None and torch.equal(with_vision.input_ids, without.input_ids)
    assert torch.equal(with_vision.position_ids, without.position_ids) and torch.equal(with_vision.segment_ids, without.segment_ids)
    plain = HybridBackbone(bb.lm, ByteTokenizer(), "sdpa").eval()
    with torch.no_grad():
        assert torch.equal(bb.encode(with_vision), plain.encode(without))


def test_hash_and_schema_round_trip(tmp_path):
    a, b = png(16, 16, 1), png(16, 16, 2)
    assert a != b
    assert state_hash(ImageState("same text", [])) == state_hash("same text")
    assert state_hash(request("t", (a,)).state) != state_hash(request("t", (b,)).state)
    assert state_hash(request("t", (a,)).state) == state_hash(request("t", (a,)).state)
    # base64 form
    r = request("hello [image:0]", (a,))
    assert isinstance(r.state, ImageState) and r.state == "hello [image:0]" and "State:\n" + r.state == "State:\nhello [image:0]"
    assert Request.from_dict(r.to_dict()) == r and r.to_dict()["state"]["images"][0]["base64"] == base64.b64encode(a).decode()
    # path form, relative to the jsonl directory
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.png").write_bytes(a)
    raw = {"state": {"text": "hello", "images": [{"path": "img/a.png"}]}, "questions": QUESTIONS}
    r = Request.from_dict(raw, base_dir=tmp_path)
    assert r.state.images[0].data == a and r.to_dict()["state"] == raw["state"]
    from janus.data import load_requests, write_jsonl
    write_jsonl(tmp_path / "set.jsonl", [raw])
    assert load_requests(tmp_path / "set.jsonl")[0].state.images[0].data == a
    # a plain JSON-object state stays what it was
    assert Request.from_dict({"state": {"message": "hi"}, "questions": QUESTIONS}).state == '{"message": "hi"}'
    for bad in ({"text": "x", "images": []}, {"text": "x", "images": [{"path": "p", "base64": "aa"}]},
                {"text": "x", "images": [{"base64": "not base64!"}]}, {"text": "x", "images": [{"base64": "aGk=", "media_type": "text/plain"}]}):
        with pytest.raises(ValueError):
            Request.from_dict({"state": bad, "questions": QUESTIONS})


def test_decision_model_config_checks_and_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="images"):
        DecisionModel(ModelConfig(backbone="tiny", images=True))
    settings = dict(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4, hidden_size=64, layers=2,
                    dtype="float32", attention="sdpa", max_tokens=512, tree_positions="continue")
    torch.manual_seed(0)
    without = DecisionModel(ModelConfig(**settings)).eval()
    with pytest.raises(ValueError, match="ModelConfig.images"):
        pack_request(request(), without.tokenizer, "tree", **without.packing_kwargs)
    model = DecisionModel(ModelConfig(**settings, images=True)).eval()
    assert model.packing_kwargs["vision"] is model.backbone.vision
    assert not any(p.requires_grad for p in model.backbone.vision.parameters())
    packed = pack_request(request(), model.tokenizer, "tree", **model.packing_kwargs)
    logits = model(packed)
    assert [len(z) for z in logits] == [3, 2, 3]
    sum(z.square().sum() for z in logits).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.backbone.named_parameters() if "lora_" in n)
    checkpoint(model, tmp_path / "m.pt", {"step": 1})
    evaluator = LocalEvaluator(tmp_path / "m.pt")
    answers = evaluator.evaluate(request())
    assert [a.question.id for a in answers] == list(QUESTIONS)
    with torch.no_grad():
        for x, y in zip(model(packed), evaluator.model(packed)):
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(torch.tensor([answers[0].probabilities[k] for k in "rbg"]), logits[0].detach().softmax(-1), atol=1e-5, rtol=1e-4)


def test_server_parses_base64_images(tmp_path):
    torch.manual_seed(0)
    model = DecisionModel(ModelConfig(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4, hidden_size=32,
                                      layers=2, dtype="float32", attention="sdpa", max_tokens=1024, images=True))
    checkpoint(model, tmp_path / "best.pt", {"step": 0})
    server = make_server(tmp_path / "best.pt", port=0, model_id="jev-local-test", queue_size=2)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/systemone"

    def post(body):
        data = json.dumps(body).encode()
        try:
            with urlopen(HTTPRequest(url, data=data, headers={"Content-Type": "application/json"}, method="POST"), timeout=60) as r:
                return r.status, json.load(r)
        except HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        status, body = post(request().to_dict())
        assert status == 200 and set(body["answers"]) == set(QUESTIONS)
        text_tokens = post({"state": "A card:  is it red?", "questions": QUESTIONS})[1]["usage"]["input_tokens"]  # the marker leaves two spaces
        assert body["usage"]["input_tokens"] == text_tokens + 6 + 2  # six image tokens plus vision_start/vision_end
        status, body = post({"state": {"text": "x", "images": [{"path": "/etc/hostname"}]}, "questions": QUESTIONS})
        assert status == 422 and "inline" in body["error"]["message"]
        status, _ = post({"state": {"text": "x", "images": [b64(png(16, 16))] * 9}, "questions": QUESTIONS})
        assert status == 422
        status, _ = post({"state": {"text": "x", "images": [{"base64": "A" * (5 << 20)}]}, "questions": QUESTIONS})
        assert status == 413
        status, _ = post({"state": {"text": "x", "images": [{"base64": "!!!"}]}, "questions": QUESTIONS})
        assert status == 422
    finally:
        server.shutdown()
