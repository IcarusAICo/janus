"""Reference tree packing. Dense masking shares state computation, not sparse FLOPs."""

from dataclasses import dataclass, field

import torch

from .attention import dense_mask
from .schema import Request


class ByteTokenizer:
    """Offline integration tokenizer, not a pretrained semantic model."""
    vocab_size = 257

    def encode(self, value, add_special_tokens=False):
        return [b + 1 for b in value.encode("utf-8")]


@dataclass
class Branch:
    start: int
    end: int
    question_index: int
    option_indices: tuple[int, ...]
    option_positions: tuple[int, ...]
    decision_position: int
    kind: str = "choice"
    leaves: tuple[tuple[int, int], ...] = ()


MODES = ("listwise", "independent", "tree", "decoder")
TREE_BLOCKS = ("full", "keys")
TREE_POSITIONS = ("shared", "restart", "continue")
# Score in tree mode: "independent" (the Phase 1 layout) keeps the level list out of the shared block, so a level leaf
# sees the state, the instructions and only its own description; "full" lists every level with its index in the block.
SCORE_BLOCKS = ("independent", "full")


@dataclass
class PackedRequest:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    segment_ids: torch.Tensor
    parents: torch.Tensor
    branches: list[Branch]
    state_length: int
    option_counts: tuple[int, ...]
    # Decoder mode only: one state-free listwise pack per question, encoded as separate sequences.
    branch_packs: list["PackedRequest"] = field(default_factory=list)
    # Image state (janus.vision): (pixel_values, image_grid_thw) for the placeholder tokens in segment 0; position_ids
    # is then [3, n] (temporal, height, width) instead of [1, n].
    images: tuple | None = None

    @property
    def token_count(self):
        return self.input_ids.shape[1]

    @property
    def allowed(self):
        return dense_mask(self.segment_ids, self.parents)

    def mask(self, directionality="causal"):
        """Dense attention mask under the given directionality; `allowed` is the causal one."""
        return dense_mask(self.segment_ids, self.parents, directionality)


def _question_tokens(encode, q, indices):
    """Listwise branch text for the given option indices; returns (tokens, option_positions)."""
    tokens = encode(f"Question ({q.kind}):\n{q.instructions}\nOptions:\n")
    option_positions = []
    for i in indices:
        option = q.options[i]
        tokens.extend(encode(f"[{option.key}] {option.description}\n"))
        option_positions.append(len(tokens) - 1)
    tokens.extend(encode("Decision:\n"))
    return tokens, option_positions


def _pack_branch_only(q, question_index, encode):
    """A question as its own causal sequence with no state: positions from 0, one segment under the (absent) state."""
    indices = tuple(range(len(q.options)))
    tokens, option_positions = _question_tokens(encode, q, indices)
    n = len(tokens)
    branch = Branch(0, n, question_index, indices, tuple(option_positions), n - 1, q.kind)
    return PackedRequest(torch.tensor([tokens]), torch.tensor([list(range(n))]), torch.ones(n, dtype=torch.long),
                         torch.zeros(2, dtype=torch.long), [branch], 0, (len(q.options),))


def pack_request(request: Request, tokenizer, mode="listwise", max_tokens=2048, tree_block="full",
                 tree_positions="shared", score_block="independent", max_state_plus_question=None, vision=None):
    """Two budgets: `max_tokens` bounds the whole packed request; `max_state_plus_question` (None: same as
    `max_tokens`) bounds the state plus the tokens of any one question, which is also the largest position used.
    `vision` (janus.vision.Vision) lays out an image state: its placeholder tokens count like text tokens, and the
    branches continue from the position after the state's last position (which is below the token count)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if tree_block not in TREE_BLOCKS:
        raise ValueError(f"tree_block must be one of {TREE_BLOCKS}")
    if tree_positions not in TREE_POSITIONS:
        raise ValueError(f"tree_positions must be one of {TREE_POSITIONS}")
    if score_block not in SCORE_BLOCKS:
        raise ValueError(f"score_block must be one of {SCORE_BLOCKS}")
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False)
    images = getattr(request.state, "images", None)
    axes = None
    if images is None:
        tokens = encode("State:\n" + request.state + "\n")
        state_length = len(tokens)
        positions = list(range(state_length))
    else:
        if vision is None:
            raise ValueError("the state has images but the model was loaded without them (ModelConfig.images)")
        tokens = encode("State:\n")
        body, axes, images = vision.layout(request.state, encode, start=len(tokens))
        tokens += body
        axes = [list(range(len(tokens) - len(body))) + axis for axis in axes]
        tail = encode("\n")
        tokens += tail
        axes = [axis + list(range(axis[-1] + 1, axis[-1] + 1 + len(tail))) for axis in axes]
        state_length = len(tokens)
        positions = axes[0]
    state_next = positions[-1] + 1  # the position after the state: its token count for text, less with images
    branches, segments = [], []  # segments: (start, end, parent_segment)
    branch_packs = []
    question_tokens = [0] * len(request.questions)
    for qi, q in enumerate(request.questions):
        question_start = len(tokens)
        if mode == "decoder":
            branch_packs.append(_pack_branch_only(q, qi, encode))
            question_tokens[qi] = branch_packs[-1].token_count
            continue
        if mode == "tree":
            block_start = len(tokens)
            header = f"Question ({q.kind}):\n{q.instructions}\n"
            if q.kind == "choice":
                line = (lambda o: f"[{o.key}] {o.description}\n") if tree_block == "full" else (lambda o: f"[{o.key}]\n")
                header += "Options:\n" + "".join(line(o) for o in q.options)
            elif q.kind == "noul":
                header += f"False means: {q.options[0].description}\nTrue means: {q.options[1].description}\n"
            elif score_block == "full":
                header += "Levels:\n" + "".join(f"[{o.key}] {o.description}\n" for o in q.options)
            tokens.extend(encode(header))
            block_end = len(tokens)
            block_length = block_end - block_start
            positions.extend(range(state_next, state_next + block_length))
            block_segment = len(segments) + 1
            segments.append((block_start, block_end, 0))
            leaf_texts = (["Decision:\n"] if q.kind == "noul" else
                          [f"Candidate: [{o.key}]\nDecision:\n" for o in q.options] if q.kind == "choice" else
                          [f"Level: {o.description}\nDecision:\n" for o in q.options])
            leaves, decisions = [], []
            next_leaf_position = state_next + block_length
            for text in leaf_texts:
                start = len(tokens)
                tokens.extend(encode(text))
                end = len(tokens)
                first = {"shared": state_next + block_length, "restart": state_next,
                         "continue": next_leaf_position}[tree_positions]
                positions.extend(range(first, first + end - start))
                next_leaf_position += end - start
                segments.append((start, end, block_segment))
                leaves.append((start, end))
                decisions.append(end - 1)
            branches.append(Branch(block_start, block_end, qi, tuple(range(len(q.options))), tuple(decisions),
                                   block_end - 1, q.kind, tuple(leaves)))
            question_tokens[qi] = len(tokens) - question_start
            continue
        groups = ([tuple(range(len(q.options)))] if mode == "listwise" else [(i,) for i in range(len(q.options))])
        for indices in groups:
            start = len(tokens)
            branch_tokens, option_positions = _question_tokens(encode, q, indices)
            tokens.extend(branch_tokens)
            option_positions = [start + p for p in option_positions]
            end = len(tokens)
            positions.extend(range(state_next, state_next + end - start))
            segments.append((start, end, 0))
            branches.append(Branch(start, end, qi, indices, tuple(option_positions), end - 1, q.kind))
        question_tokens[qi] = len(tokens) - question_start
    n = len(tokens) + sum(p.token_count for p in branch_packs)
    if n > max_tokens:
        # janus.server maps "exceeding max_tokens" to 413; keep that phrase.
        raise ValueError(f"Packed request has {n} tokens, exceeding max_tokens={max_tokens} (the per-request budget); "
                         "shorten the request or split questions explicitly")
    cap = max_tokens if max_state_plus_question is None else max_state_plus_question
    longest = max(question_tokens)
    if state_length + longest > cap:
        raise ValueError(f"State ({state_length} tokens) plus the longest question ({longest} tokens, question "
                         f"{request.questions[question_tokens.index(longest)].id!r}) is {state_length + longest} tokens, "
                         f"exceeding max_state_plus_question={cap} (the state-plus-question budget); shorten the state")
    n = len(tokens)
    segment_ids = torch.zeros(n, dtype=torch.long)
    parents = torch.zeros(len(segments) + 1, dtype=torch.long)
    for index, (start, end, parent) in enumerate(segments, start=1):
        segment_ids[start:end] = index
        parents[index] = parent
    if axes is not None:  # height and width axes: the image grid inside the state, the text positions after it
        positions = [positions] + [axis + positions[state_length:] for axis in axes[1:]]
        positions = torch.tensor(positions)
    else:
        positions = torch.tensor([positions])
    return PackedRequest(torch.tensor([tokens]), positions, segment_ids, parents,
                         branches, state_length, tuple(len(q.options) for q in request.questions), branch_packs, images)
