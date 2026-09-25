"""Public dataset conversions to Choice/Score/Noul with verification tiers (spec WP3 section 3.2)."""

import hashlib
import io
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen
import zipfile

from ..data import assert_disjoint, file_hash, state_hash, write_json, write_jsonl
from ..schema import Request
from ..study_data import balanced_rows

NONE_TEXT = "None of these."
NLI_DESCRIPTIONS = ("Entailed: the premise establishes that the hypothesis is true.",
                    "Unknown: the premise establishes neither the hypothesis nor its negation.",
                    "Contradicted: the premise establishes that the hypothesis is false.")


def group(text):
    return state_hash(text)


def menu(rng, gold, pool, cardinalities=(2, 4, 8), omit=.2, none_text=NONE_TEXT):
    candidates = [x for x in pool if x != gold]
    cardinalities = tuple(k for k in cardinalities if k <= len(candidates) + 1)
    if not cardinalities:
        raise ValueError("Pool too small for a menu")
    k = rng.choice(cardinalities)
    omitted = rng.random() < omit
    words = set(re.findall(r"[a-z]+", gold.lower()))
    rng.shuffle(candidates)
    hard = sorted(candidates, key=lambda c: len(words & set(re.findall(r"[a-z]+", c.lower()))), reverse=True)
    needed = k - 1 if omitted else k - 2
    chosen = hard[:max(0, needed // 2)]
    chosen += [x for x in candidates if x not in chosen][:needed - len(chosen)]
    if not omitted:
        chosen.append(gold)
    chosen.append(None)
    rng.shuffle(chosen)
    options = {f"o{i}": (c.replace("_", " ") if c else none_text) for i, c in enumerate(chosen)}
    target = [float((c is None) if omitted else (c == gold)) for c in chosen]
    return options, target


def _row(state, questions, group_id, tier, family, source):
    raw = {"state": state, "questions": questions, "group_id": group_id}
    Request.from_dict(raw)  # validate
    return {**raw, "tier": tier, "family": family, "source": source}


def convert_massive(rows, rng, label_names=None):
    if not label_names:
        raise ValueError("MASSIVE needs intent label names")
    out = []
    for r in rows:
        gold = label_names[int(r["intent"])]
        options, target = menu(rng, gold, list(label_names))
        positive = rng.random() < .5
        probe = gold if positive else rng.choice([x for x in label_names if x != gold])
        out.append(_row(r["utt"], {
            "massive:intent": {"type": "choice", "instructions": "Which intent best describes the user's request?",
                               "criteria": options, "target": target},
            "massive:matches": {"type": "noul", "instructions": f"Does this request express the intent: {probe.replace('_', ' ')}?",
                                "target": [float(not positive), float(positive)]}},
            f"massive:{r.get('locale', 'xx')}:{group(r['utt'])}", "T1", "massive", "AmazonScience/massive"))
    return out


def convert_wildguard(rows, rng, label_names=None):
    categories = sorted({r["subcategory"] for r in rows if r.get("subcategory")})
    out = []
    for r in rows:
        if r.get("prompt_harm_label") not in ("harmful", "unharmful") or not r.get("prompt"):
            continue
        harmful = r["prompt_harm_label"] == "harmful"
        questions = {"wildguard:harmful": {"type": "noul", "instructions": "Is this user prompt harmful?",
                                           "target": [float(not harmful), float(harmful)]}}
        if r.get("subcategory") and len(categories) >= 3:
            options, target = menu(rng, r["subcategory"], categories)
            questions["wildguard:category"] = {"type": "choice", "instructions": "Which category best describes this prompt?",
                                               "criteria": options, "target": target}
        out.append(_row(r["prompt"], questions, f"wildguard:{group(r['prompt'])}", "T1", "wildguard", "allenai/wildguardmix"))
    return out


def convert_civil(rows, rng, label_names=None):
    out = []
    for r in rows:
        t = r.get("toxicity")
        if not isinstance(t, (int, float)) or not 0 <= t <= 1 or not r.get("text"):
            continue
        out.append(_row(r["text"], {"civil:toxic": {"type": "noul", "instructions": "Is this comment toxic?",
                                                    "target": [1 - float(t), float(t)]}},
                        f"civil:{group(r['text'])}", "T2", "civil", "google/civil_comments"))
    return out


HELPFULNESS = ["Not helpful at all", "Slightly helpful", "Moderately helpful", "Very helpful", "Extremely helpful"]
COHERENCE = ["Incoherent", "Mostly incoherent", "Somewhat coherent", "Mostly coherent", "Fully coherent"]


def convert_helpsteer(rows, rng, label_names=None):
    out = []
    for r in rows:
        h, c = r.get("helpfulness"), r.get("coherence")
        if h not in range(5) or c not in range(5):
            continue
        state = json.dumps({"prompt": r["prompt"], "response": r["response"]}, ensure_ascii=False)
        out.append(_row(state, {
            "helpsteer:helpfulness": {"type": "score", "instructions": "How helpful is the response to the prompt?",
                                      "criteria": HELPFULNESS, "target": [float(i == h) for i in range(5)]},
            "helpsteer:coherence": {"type": "score", "instructions": "How coherent is the response?",
                                    "criteria": COHERENCE, "target": [float(i == c) for i in range(5)]}},
            f"helpsteer:{group(r['prompt'])}", "T1", "helpsteer", "nvidia/HelpSteer2"))
    return out


def _decode_list(value):
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return []
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def convert_arena(rows, rng, label_names=None):
    out = []
    for r in rows:
        prompt, a, b = _decode_list(r["prompt"]), _decode_list(r["response_a"]), _decode_list(r["response_b"])
        if not prompt or not a or not b:
            continue
        winner = [int(r.get("winner_model_a", 0)), int(r.get("winner_model_b", 0)), int(r.get("winner_tie", 0))]
        if sum(winner) != 1:
            continue
        state = json.dumps({"prompt": prompt[0], "response_a": a[0], "response_b": b[0]}, ensure_ascii=False)
        out.append(_row(state, {
            "arena:better": {"type": "choice", "instructions": "Which response better answers the prompt?",
                             "criteria": {"a": "Response A is better", "b": "Response B is better", "tie": "Tie or both bad"},
                             "target": [float(w) for w in winner]},
            "arena:a_better": {"type": "noul", "instructions": "Is response A better than response B?",
                               "target": [float(winner[0] == 0), float(winner[0] == 1)]}},
            f"arena:{group(prompt[0])}", "T1", "arena", "lmarena-ai/arena-human-preference-55k"))
    return out


def convert_mmlu_pro(rows, rng, label_names=None):
    out = []
    for r in rows:
        options = [o for o in r["options"] if isinstance(o, str) and o.strip() and o != "N/A"]
        index = int(r["answer_index"])
        if index >= len(options):
            continue
        keys = [chr(ord("A") + i) for i in range(len(options))]
        out.append(_row(r["question"], {"mmlu_pro:answer": {"type": "choice", "instructions": f"Answer this {r.get('category', '')} question.".replace("  ", " "),
                                                            "criteria": dict(zip(keys, options)), "target": [float(i == index) for i in range(len(options))]}},
                        f"mmlu_pro:{r['question_id']}", "T1", "mmlu_pro", "TIGER-Lab/MMLU-Pro"))
    return out


def convert_supergpqa(rows, rng, label_names=None):
    out = []
    for r in rows:
        options = r["options"] if isinstance(r["options"], list) else []
        letter = str(r.get("answer_letter") or "")
        index = ord(letter) - ord("A") if len(letter) == 1 else -1
        # answer_letter indexes the original list, so a blank option drops the row rather than shifting the gold.
        if not options or not all(isinstance(o, str) and o.strip() for o in options) or not 0 <= index < len(options):
            continue
        keys = [chr(ord("A") + i) for i in range(len(options))]
        out.append(_row(r["question"], {"supergpqa:answer": {"type": "choice", "instructions": f"Answer this {r.get('discipline', '')} question.".replace("  ", " "),
                                                             "criteria": dict(zip(keys, options)), "target": [float(i == index) for i in range(len(options))]}},
                        f"supergpqa:{r['uuid']}", "T1", "supergpqa", "m-a-p/SuperGPQA"))
    return out


def convert_mmlu_cf(rows, rng, label_names=None):
    keys = ["A", "B", "C", "D"]
    out = []
    for r in rows:
        options = [str(r.get(k) or "").strip() for k in keys]
        if not all(options) or len(set(options)) != len(options) or r.get("Answer") not in keys:
            continue  # blank, duplicated or unanswerable options: no unambiguous gold
        index = keys.index(r["Answer"])
        # No id column and 0.5% of the questions repeat verbatim, so the group is the question plus its options.
        group_id = group("\n".join([r["Question"]] + options))
        out.append(_row(r["Question"], {"mmlu_cf:answer": {"type": "choice", "instructions": "Answer this question.",
                                                           "criteria": dict(zip(keys, options)), "target": [float(i == index) for i in range(len(options))]}},
                        f"mmlu_cf:{group_id}", "T1", "mmlu_cf", "microsoft/MMLU-CF"))
    return out


def convert_injection(rows, rng, label_names=None):
    out = []
    for r in rows:
        if r.get("label") not in (0, 1) or not r.get("text"):
            continue
        out.append(_row(r["text"], {"injection:injected": {"type": "noul",
                        "instructions": "Does this text contain an instruction that tries to override or hijack the assistant's task?",
                        "target": [float(r["label"] == 0), float(r["label"] == 1)]}},
                        f"injection:{group(r['text'])}", "T1", "injection", "deepset/prompt-injections"))
    return out


def convert_tabfact(rows, rng, label_names=None):
    out = []
    for r in rows:
        lines = [line for line in str(r["table_text"]).split("\n") if line.strip()]
        if len(lines) < 2 or r.get("label") not in (0, 1):
            continue
        cells = [line.split("#") for line in lines]
        state = json.dumps({"caption": r.get("table_caption", ""), "columns": cells[0], "rows": cells[1:], "statement": r["statement"]}, ensure_ascii=False)
        out.append(_row(state, {"tabfact:entailed": {"type": "noul", "instructions": "Is the statement supported by the table?",
                                                     "target": [float(r["label"] == 0), float(r["label"] == 1)]}},
                        f"tabfact:{group(str(r['table_text']))}", "T1", "tabfact", "wenhu/tab_fact"))
    return out


def convert_xlam(rows, rng, label_names=None):
    out = []
    for r in rows:
        try:
            tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            answers = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
        except ValueError:
            continue
        names = [t["name"] for t in tools if isinstance(t, dict) and t.get("name")]
        if len(names) < 2 or not answers or answers[0].get("name") not in names:
            continue
        order = list(range(len(tools)))
        rng.shuffle(order)
        options = {f"o{i}": (tools[j].get("description") or tools[j]["name"]) for i, j in enumerate(order)}
        target = [float(tools[j]["name"] == answers[0]["name"]) for j in order]
        out.append(_row(r["query"], {"xlam:tool": {"type": "choice", "instructions": "Which tool should be called first for this request?",
                                                   "criteria": options, "target": target}},
                        f"xlam:{group(r['query'])}", "T1", "xlam", "Salesforce/xlam-function-calling-60k"))
    return out


def _nli_question(rng, dist):
    """Entailed/unknown/contradicted choice over `dist` (ordered entailment, neutral, contradiction), options shuffled."""
    order = [0, 1, 2]
    rng.shuffle(order)
    return {"type": "choice", "instructions": "Assume the premise is true. What relationship does the hypothesis have to the premise?",
            "criteria": {f"o{i}": NLI_DESCRIPTIONS[j] for i, j in enumerate(order)}, "target": [float(dist[j]) for j in order]}


def convert_chaosnli(rows, rng, label_names=None):
    out = []
    for r in rows:
        dist = r.get("label_dist")
        if not isinstance(dist, list) or len(dist) != 3 or abs(sum(dist) - 1) > 1e-3:
            continue
        example = r["example"]
        state = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
        out.append(_row(state, {"chaosnli:relation": _nli_question(rng, dist)},
                        f"chaosnli:{r['uid']}", "T2", "chaosnli", "ChaosNLI (github.com/easonnie/ChaosNLI)"))
    return out


def convert_xnli(rows, rng, label_names=None):
    """XNLI rows ({premise, hypothesis, label 0/1/2 = entailment/neutral/contradiction, lang}) as one 3-way choice."""
    out = []
    for r in rows:
        if r.get("label") not in (0, 1, 2):
            continue
        state = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
        out.append(_row(state, {"xnli:relation": _nli_question(rng, [float(r["label"] == j) for j in range(3)])},
                        f"xnli:{r['lang']}:{group(state)}", "T1", "xnli", "facebook/xnli"))
    return out


def convert_massive_k(rows, rng, label_names, k=20):
    """Laya's MASSIVE protocol: one k-way intent choice, gold plus k-1 uniformly drawn distractors, no "None of these"."""
    out = []
    for r in rows:
        gold = label_names[int(r["intent"])]
        chosen = rng.sample([x for x in label_names if x != gold], k - 1) + [gold]
        rng.shuffle(chosen)
        out.append(_row(r["utt"], {"massive:intent": {"type": "choice", "instructions": "Which intent best describes the user's request?",
                                                      "criteria": {f"o{i}": c.replace("_", " ") for i, c in enumerate(chosen)},
                                                      "target": [float(c == gold) for c in chosen]}},
                        f"massive:{r.get('locale', 'xx')}:{group(r['utt'])}", "T1", "massive", "AmazonScience/massive"))
    return out


# --- Registry, download, sampling, splitting, manifest -----------------------------------------

PARQUET_BRANCH = "refs/convert/parquet"  # script-based Hub datasets are unsupported by `datasets` 4.x; read the Hub's parquet export
SOURCES = {
    "massive": {"repo": "AmazonScience/massive", "configs": ["en-US", "de-DE", "fr-FR", "es-ES"], "splits": {"train": "train", "test": "test"},
                "converter": convert_massive, "licence": "CC-BY-4.0", "needs_label_names": True, "eval_only": False,
                "revision": PARQUET_BRANCH, "data_files": "{config}/{split}/*.parquet"},
    "wildguard": {"repo": "allenai/wildguardmix", "configs": ["wildguardtrain"], "test_config": "wildguardtest", "splits": {"train": "train", "test": "test"},
                  "converter": convert_wildguard, "licence": "ODC-BY (see dataset card)", "needs_label_names": False, "eval_only": False},
    "civil": {"repo": "google/civil_comments", "configs": [None], "splits": {"train": "train", "test": "test"},
              "converter": convert_civil, "licence": "CC0-1.0", "needs_label_names": False, "eval_only": False,
              "max_rows": {"train": 60_000, "test": 20_000}},  # ~1.8M train rows; deterministic subsample before conversion
    "helpsteer": {"repo": "nvidia/HelpSteer2", "configs": [None], "splits": {"train": "train", "test": "validation"},
                  "converter": convert_helpsteer, "licence": "CC-BY-4.0", "needs_label_names": False, "eval_only": False},
    "arena": {"repo": "lmarena-ai/arena-human-preference-55k", "configs": [None], "splits": {"train": "train"},
              "converter": convert_arena, "licence": "Apache-2.0 (see dataset card)", "needs_label_names": False, "eval_only": False},
    # eval_only sources are knowledge benchmarks: they go to their own test_<name>.jsonl and NEVER into train,
    # dev, calibration or any production mix builder. The committed 1,000-item samples (test_<name>_1000.jsonl)
    # are random.Random(17).sample(converted, 1000) over the converted test rows, in dataset order.
    "mmlu_pro": {"repo": "TIGER-Lab/MMLU-Pro", "configs": [None], "splits": {"test": "test"},
                 "converter": convert_mmlu_pro, "licence": "MIT", "needs_label_names": False, "eval_only": True},
    "supergpqa": {"repo": "m-a-p/SuperGPQA", "configs": [None], "splits": {"test": "train"},  # one file, exposed as "train"
                  "converter": convert_supergpqa, "licence": "ODC-BY (composite; see dataset card)", "needs_label_names": False, "eval_only": True},
    "mmlu_cf": {"repo": "microsoft/MMLU-CF", "configs": [None], "splits": {"test": "val"},  # the test split is deliberately closed
                "converter": convert_mmlu_cf, "licence": "CDLA-Permissive-2.0", "needs_label_names": False, "eval_only": True},
    "injection": {"repo": "deepset/prompt-injections", "configs": [None], "splits": {"train": "train", "test": "test"},
                  "converter": convert_injection, "licence": "Apache-2.0", "needs_label_names": False, "eval_only": False},
    "tabfact": {"repo": "wenhu/tab_fact", "configs": ["tab_fact"], "splits": {"train": "train", "test": "test"},
                "converter": convert_tabfact, "licence": "CC-BY-4.0", "needs_label_names": False, "eval_only": False,
                "revision": PARQUET_BRANCH, "data_files": "{config}/{split}/*.parquet"},
    "xlam": {"repo": "Salesforce/xlam-function-calling-60k", "configs": [None], "splits": {"train": "train"},
             "converter": convert_xlam, "licence": "CC-BY-NC-4.0 (gated; research use)", "needs_label_names": False, "eval_only": False},
    "chaosnli": {"loader": "chaosnli", "url": "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1", "splits": {"test": "test"},
                 "converter": convert_chaosnli, "licence": "CC-BY-4.0 (see repository)", "needs_label_names": False, "eval_only": False},
}
DEFAULT_CAPS = {"train": 5000, "dev": 300, "calibration": 300, "test": 1000}


def _load_hf(repo, config, split, cache_dir, revision=None, data_files=None):
    from datasets import load_dataset
    if data_files:
        # Hub parquet export: configs are directories, not builder configs.
        pattern = data_files.format(config=config, split=split)
        dataset = load_dataset(repo, data_files={split: pattern}, split=split, revision=revision, cache_dir=cache_dir)
    else:
        dataset = load_dataset(repo, config, split=split, revision=revision, cache_dir=cache_dir)
    return dataset, _hub_revision(repo)


def _hub_revision(repo):
    """The Hub commit sha of the repository's main branch, or None when unreachable."""
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def load_source(name, cache_dir=".cache/study", seed=17):
    spec = SOURCES[name]
    if spec.get("loader") == "chaosnli":
        path = Path(cache_dir) / "chaosNLI_v1.0.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urlopen(spec["url"], timeout=120) as response:
                path.write_bytes(response.read())
        if not zipfile.is_zipfile(path):
            path.unlink()  # an HTML error page, not the archive; do not leave it cached
            raise ValueError(f"{spec['url']} did not return a zip archive (Dropbox reports the file deleted)")
        rows = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith("chaosNLI_snli.jsonl") or member.endswith("chaosNLI_mnli_m.jsonl"):
                    rows += [json.loads(line) for line in io.TextIOWrapper(archive.open(member), encoding="utf-8") if line.strip()]
        return {"train": [], "test": rows, "label_names": None, "revision": file_hash(path)[:16]}
    result = {"train": [], "test": [], "label_names": None, "revision": None}
    for config in spec["configs"]:
        for ours, theirs in spec["splits"].items():
            dataset, revision = _load_hf(spec["repo"], config, theirs, cache_dir, spec.get("revision"), spec.get("data_files"))
            result["revision"] = result["revision"] or revision
            if spec["needs_label_names"] and result["label_names"] is None:
                result["label_names"] = dataset.features["intent"].names
            limit = (spec.get("max_rows") or {}).get(ours)
            if limit and len(dataset) > limit:
                dataset = dataset.shuffle(seed=seed).select(range(limit))
            result[ours].extend(dict(row) for row in dataset)
    if spec.get("test_config"):
        dataset, _ = _load_hf(spec["repo"], spec["test_config"], "test", cache_dir, spec.get("revision"), spec.get("data_files"))
        result["test"].extend(dict(row) for row in dataset)
    return result


def _label_of(row):
    if row.get("tier") == "T2":
        return "all"  # distributional targets: never threshold-balance, keep the annotator base rate
    q = next(iter(row["questions"].values()))
    t = q["target"]
    index = max(range(len(t)), key=t.__getitem__)
    if q["type"] == "choice" and isinstance(q["criteria"], dict):
        return str(list(q["criteria"].values())[index])  # balance by gold description (intent), not by slot
    return str(index)


def split_rows(converted, rng, caps, has_official_test, seed=17, official_test=None):
    by_split = {"train": [], "dev": [], "calibration": [], "test": []}
    for row in converted:
        bucket = int(hashlib.sha256(f"{seed}:{row['group_id']}".encode()).hexdigest(), 16) % 10
        by_split["dev" if bucket == 8 else "calibration" if bucket == 9 else "train"].append(row)
    if has_official_test:
        by_split["test"] = list(official_test or [])
    else:
        holdout = []
        keep = []
        for row in by_split["train"]:
            bucket = int(hashlib.sha256(f"{seed}:test:{row['group_id']}".encode()).hexdigest(), 16) % 10
            (holdout if bucket < 2 else keep).append(row)
        by_split["train"], by_split["test"] = keep, holdout
    out = {}
    for name, rows in by_split.items():
        rows = [dict(r, state=r["state"]) for r in rows]
        for r in rows:
            r.setdefault("label", _label_of(r))
        selected = balanced_rows([{"text": r["state"], "label": r["label"], "_row": r} for r in rows], caps.get(name), seed)
        out[name] = [item["_row"] for item in selected]
        for r in out[name]:
            r.pop("label", None)
    return out


def _dedupe(rows, exclude_states=(), exclude_groups=()):
    """Keep the first row per normalised state; drop rows whose state or group is already taken."""
    seen, out = set(), []
    for r in rows:
        key = state_hash(r["state"])
        if key in seen or key in exclude_states or r["group_id"] in exclude_groups:
            continue
        seen.add(key)
        out.append(r)
    return out


def _drop_cross_source_states(splits):
    """Remove every occurrence of a normalised state that appears in more than one source."""
    owners = {}
    for rows in splits.values():
        for r in rows:
            owners.setdefault(state_hash(r["state"]), set()).add(r["family"])
    shared = {key for key, families in owners.items() if len(families) > 1}
    dropped = {}
    for name, rows in splits.items():
        kept = [r for r in rows if state_hash(r["state"]) not in shared]
        dropped[name] = len(rows) - len(kept)
        splits[name] = kept
    return {"normalized_states": len(shared), "rows_dropped": dropped}


def prepare_public(output, sources=None, seed=17, caps=None, cache_dir=".cache/study"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    caps = caps or DEFAULT_CAPS
    sources = sources or list(SOURCES)
    splits = {name: [] for name in ("train", "dev", "calibration", "test")}
    eval_only = {}
    manifest = {"dataset": "JEV_PUBLIC_V1", "seed": seed, "caps": caps, "counts": {}, "skipped": {}, "revisions": {},
                "licences": {name: SOURCES[name]["licence"] for name in sources},
                "subsampled_before_conversion": {name: SOURCES[name]["max_rows"] for name in sources if SOURCES[name].get("max_rows")},
                "duplicates_dropped": {}}
    for name in sources:
        try:
            loaded = load_source(name, cache_dir)
        except Exception as error:  # download, gating, schema changes
            manifest["skipped"][name] = f"{type(error).__name__}: {error}"[:300]
            continue
        spec = SOURCES[name]
        rng = random.Random(f"{seed}:{name}")
        converted_train = spec["converter"](loaded["train"], rng, loaded["label_names"])
        converted_test = spec["converter"](loaded["test"], rng, loaded["label_names"])
        manifest["revisions"][name] = loaded["revision"]
        if spec["eval_only"]:  # its own file; never merged into the training/calibration splits
            eval_only[name] = converted_test
            manifest["counts"][name] = {f"test_{name}": len(converted_test)}
            continue
        # One row per normalised state; official test rows win over train rows sharing a state or group.
        raw_train, raw_test = len(converted_train), len(converted_test)
        converted_test = _dedupe(converted_test)
        converted_train = _dedupe(converted_train, {state_hash(r["state"]) for r in converted_test},
                                  {r["group_id"] for r in converted_test})
        manifest["duplicates_dropped"][name] = {"train": raw_train - len(converted_train), "test": raw_test - len(converted_test)}
        parts = split_rows(converted_train or converted_test, rng, caps, bool(converted_train) and bool(converted_test), seed,
                           official_test=converted_test)
        manifest["counts"][name] = {k: len(v) for k, v in parts.items()}
        for k, v in parts.items():
            splits[k].extend(v)
    manifest["cross_source_deduplication"] = _drop_cross_source_states(splits)
    for name in manifest["counts"]:
        if not SOURCES[name]["eval_only"]:
            manifest["counts"][name] = {k: sum(r["family"] == name for r in v) for k, v in splits.items()}
    assert_disjoint({k: [Request.from_dict(r) for r in v] for k, v in splits.items() if v})
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    for name, rows in eval_only.items():
        write_jsonl(output / f"test_{name}.jsonl", rows)
    manifest["tiers"] = {tier: sum(r["tier"] == tier for rows in splits.values() for r in rows) for tier in ("T1", "T2")}
    manifest["files"] = {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}
    write_json(output / "manifest.json", manifest)
    return manifest


# --- Multilingual set: MASSIVE (51 locales) and XNLI (15 languages), Laya's benchmark protocol -------------------------

MASSIVE_LOCALES = (  # MASSIVE 1.0, the 51 Laya reports; 1.1 adds ca-ES
                   "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "cy-GB", "da-DK", "de-DE", "el-GR", "en-US", "es-ES", "fa-IR",
                   "fi-FI", "fr-FR", "he-IL", "hi-IN", "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID", "ka-GE", "km-KH",
                   "kn-IN", "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY", "my-MM", "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU",
                   "sl-SL", "sq-AL", "sv-SE", "sw-KE", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR", "ur-PK", "vi-VN", "zh-CN", "zh-TW")
XNLI_LANGUAGES = ("ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh")
MULTILINGUAL_SOURCES = {  # pinned Hub commits: MASSIVE's parquet export branch, XNLI's main branch
    "massive": {"repo": "AmazonScience/massive", "revision": "ed58ac423a2f4121720918bf5301577edce4ffd3", "data_files": "{config}/{split}/*.parquet",
                "configs": MASSIVE_LOCALES, "licence": "CC-BY-4.0"},
    "xnli": {"repo": "facebook/xnli", "revision": "b8dd5d7af51114dbda02c0e3f6133f332186418e", "data_files": "{config}/{split}-*.parquet",
             "configs": XNLI_LANGUAGES, "licence": "CC-BY-NC-4.0 (see dataset card)"},
}
# Per locale/language: 51*80 + 15*128 = 6000 train; 51*4 + 15*7 = 309 dev and calibration each; tests mirror Laya.
MULTILINGUAL_SIZES = {"massive": {"train": 80, "dev": 4, "calibration": 4, "test": 100},
                      "xnli": {"train": 128, "dev": 7, "calibration": 7, "test": 200}}
TEST_FILES = {"massive": "test_massive51", "xnli": "test_xnli15"}


def _take(rows, n, convert, taken):
    """Convert rows in order until n rows whose normalised state is not yet taken; claims each kept state."""
    out = []
    for r in rows:
        if len(out) == n:
            break
        for row in convert([r]):
            key = state_hash(row["state"])
            if key not in taken:
                taken.add(key)
                out.append(dict(row, pool="multilingual"))
    return out


def build_multilingual(massive, xnli, label_names, seed=17, sizes=None):
    """`massive`/`xnli`: {locale: {"train", "validation", "test": raw rows}}. Returns {file stem: rows}, pairwise state-disjoint.

    Tests come only from official test splits, dev/calibration only from validation, train only from train; every official
    test state (sampled or not) is barred from the other files. Test rows follow one shared permutation per dataset, so the
    parallel corpora give (mostly) the same items in every language."""
    sizes = sizes or MULTILINGUAL_SIZES
    data = {"massive": massive, "xnli": xnli}
    train_convert = {"massive": convert_massive, "xnli": convert_xnli}
    test_convert = {"massive": convert_massive_k, "xnli": convert_xnli}
    out = {"train": [], "dev": [], "calibration": [], **{TEST_FILES[f]: [] for f in data}}
    taken = set()
    for family, locales in data.items():
        for locale in sorted(locales):
            rows = locales[locale]["test"]
            order = list(range(len(rows)))
            random.Random(f"{seed}:{family}:order").shuffle(order)
            rng = random.Random(f"{seed}:{family}:{locale}:test")
            out[TEST_FILES[family]] += _take([rows[i] for i in order], sizes[family]["test"],
                                             lambda r: test_convert[family](r, rng, label_names), taken)
    for family, locales in data.items():  # bar every official test state, not only the sampled ones
        for locale in locales:
            taken |= {state_hash(row["state"]) for row in test_convert[family](locales[locale]["test"], random.Random(0), label_names)}
    for family, locales in data.items():
        for locale in sorted(locales):
            rng = random.Random(f"{seed}:{family}:{locale}")
            validation, train = list(locales[locale]["validation"]), list(locales[locale]["train"])
            rng.shuffle(validation)
            rng.shuffle(train)
            for split, source in (("dev", validation), ("calibration", validation), ("train", train)):
                out[split] += _take(source, sizes[family][split], lambda r: train_convert[family](r, rng, label_names), taken)
    assert_disjoint({name: [Request.from_dict(r) for r in rows] for name, rows in out.items() if rows})
    return out


def _load_multilingual(family, cache_dir, train_cap, seed):
    from datasets import load_dataset
    spec = MULTILINGUAL_SOURCES[family]
    result, label_names = {}, None
    for config in spec["configs"]:
        result[config] = {}
        for split in ("train", "validation", "test"):
            dataset = load_dataset(spec["repo"], data_files={split: spec["data_files"].format(config=config, split=split)}, split=split,
                                   revision=spec["revision"], cache_dir=cache_dir)
            if family == "massive":
                label_names = label_names or dataset.features["intent"].names
            elif dataset.features["label"].names != ["entailment", "neutral", "contradiction"]:
                raise ValueError(f"Unexpected XNLI labels: {dataset.features['label'].names}")
            if split == "train" and len(dataset) > train_cap:  # XNLI train is 392k pairs per language; subsample before dict conversion
                dataset = dataset.shuffle(seed=seed).select(range(train_cap))
            result[config][split] = [dict(row, lang=config) for row in dataset]
    return result, label_names


def prepare_multilingual(output, seed=17, cache_dir=".cache/multilingual"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    massive, label_names = _load_multilingual("massive", cache_dir, 100_000, seed)
    xnli, _ = _load_multilingual("xnli", cache_dir, 20 * MULTILINGUAL_SIZES["xnli"]["train"], seed)
    splits = build_multilingual(massive, xnli, label_names, seed)
    counts = {}
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
        counts[name] = {}
        for r in rows:
            key = ":".join(r["group_id"].split(":")[:2])
            counts[name][key] = counts[name].get(key, 0) + 1
    manifest = {"dataset": "JEV_MULTILINGUAL_V1", "seed": seed, "sizes_per_locale": MULTILINGUAL_SIZES,
                "sources": {k: {"repo": v["repo"], "revision": v["revision"], "licence": v["licence"], "locales": list(v["configs"]),
                                "splits": {"train": "train", "dev": "validation", "calibration": "validation", "test": "test"}}
                            for k, v in MULTILINGUAL_SOURCES.items()},
                "protocol": {"test_massive51": "one 20-way intent choice per utterance: gold + 19 uniform distractors, shuffled (Laya)",
                             "test_xnli15": "one 3-way entailed/unknown/contradicted choice per premise-hypothesis pair",
                             "disjointness": "no normalised state of any official test split appears in train/dev/calibration"},
                "totals": {name: len(rows) for name, rows in splits.items()}, "counts": counts,
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return {k: manifest[k] for k in ("dataset", "totals")}
