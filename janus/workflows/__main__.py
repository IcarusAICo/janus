"""CLI for WP4: `python -m janus.workflows data|train|evaluate|env|report`."""

import argparse
import json
from pathlib import Path
import random


def _config_from_args(args):
    from .rl import WorkflowRLConfig
    base = json.loads(Path(args.config).read_text()) if args.config else {}
    config = WorkflowRLConfig(arm=args.arm, estimator=args.estimator, group=args.group, states_per_step=args.states_per_step,
                              steps=args.steps, epochs=args.epochs, temperature=args.temperature, proper_score=args.proper_score,
                              mask_proper_from_policy=not args.no_mask, seed=args.seed, log_every=args.log_every,
                              backbone_lr=base.get("backbone_lr", 2e-4), head_lr=base.get("head_lr", 1e-3),
                              weight_decay=base.get("weight_decay", .01), device=args.device or base.get("device", "cpu"),
                              noul_target=args.noul_target)
    return config


def _add_training_args(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Training config whose learning rates, weight decay and device are reused")
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="rl", choices=("rl", "rl_only", "supervised"))
    parser.add_argument("--estimator", default="rloo", choices=("rloo", "group_mean", "grpo"))
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--states-per-step", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--proper-score", default="brier", choices=("brier", "ce"))
    parser.add_argument("--no-mask", action="store_true", help="Let the proper score train the backbone too")
    parser.add_argument("--noul-target", default="bfs", choices=("bfs", "realised"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device")
    parser.add_argument("--log-every", type=int, default=10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Write disjoint train and held-out workflow worlds")
    data.add_argument("--output", required=True)
    data.add_argument("--train", type=int, default=2000)
    data.add_argument("--heldout", type=int, default=500)
    data.add_argument("--seed", type=int, default=17)
    train = commands.add_parser("train", help="WP4a: policy gradient through the five workflows from a checkpoint")
    _add_training_args(train)
    train.add_argument("--worlds", required=True)
    evaluation = commands.add_parser("evaluate", help="Greedy workflow-level evaluation on a world file")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--worlds", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="cpu")
    env = commands.add_parser("env", help="WP4b: gridworld episodes with REINFORCE on the return and Brier on the success Nouls")
    _add_training_args(env)
    env.add_argument("--eval-starts", type=int, default=200)
    report = commands.add_parser("report", help="Markdown table over the arms in a run directory")
    report.add_argument("root")
    args = parser.parse_args(argv)
    from ..data import write_json
    if args.command == "data":
        from .programs import prepare_workflow_data
        result = prepare_workflow_data(args.output, args.train, args.heldout, args.seed)
    elif args.command in {"train", "env"}:
        from ..training import load_checkpoint
        from .rl import train_env_rl, train_workflow_rl, evaluate_env
        model, start_metadata = load_checkpoint(args.checkpoint, args.device or "cpu")
        model.start_metadata = start_metadata
        config = _config_from_args(args)
        if args.command == "train":
            from .programs import WORKFLOWS, load_worlds
            history = train_workflow_rl(model, WORKFLOWS, config, worlds=load_worlds(args.worlds), output=args.output)
            result = history[-1]
        else:
            from .env import budgeted_grid
            history = train_env_rl(model, config, output=args.output)
            rng = random.Random(config.seed + 1)
            evaluation_result = evaluate_env(model, [budgeted_grid(rng, max_distance=config.env_max_distance) for _ in range(args.eval_starts)])
            write_json(Path(args.output) / "evaluation.json", evaluation_result)
            result = {"final": history[-1], "evaluation": evaluation_result}
    elif args.command == "evaluate":
        from ..training import load_checkpoint
        from .programs import load_worlds
        from .rl import evaluate_workflows
        model, metadata = load_checkpoint(args.checkpoint, args.device)
        model.start_metadata = metadata
        worlds = load_worlds(args.worlds)
        from ..data import state_hash
        from .programs import WORKFLOW_BY_NAME
        used = set(metadata.get("training_group_ids", [])) | set(metadata.get("training_state_hashes", []))
        requests = [WORKFLOW_BY_NAME[w.workflow].request(w) for w in worlds]
        if any(r.group_id in used or state_hash(r.state) in used for r in requests):
            raise ValueError("Held-out worlds overlap the checkpoint's training states")
        result = evaluate_workflows(model, worlds)
        write_json(Path(args.output) / "summary.json", result)
    else:
        root = Path(args.root)
        lines = ["| arm | held-out greedy reward | per-workflow | held-out question accuracy | held-out Brier | held-out ECE | panel accuracy | panel NLL | panel ECE |",
                 "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm in sorted(p.name for p in root.iterdir() if p.is_dir()):
            def load(name):
                path = root / arm / name
                return json.loads(path.read_text()) if path.exists() else None
            heldout, questions, panel = load("heldout/summary.json"), load("heldout_questions/metrics.json"), load("panel/metrics.json")
            cell = lambda value, key: f"{value['raw'][key]:.4f}" if value else "n/a"
            per_workflow = ", ".join(f"{k} {v:.3f}" for k, v in heldout["per_workflow"].items()) if heldout else "n/a"
            lines.append(f"| {arm} | {heldout['greedy_reward_mean']:.4f} | {per_workflow} | " if heldout else f"| {arm} | n/a | n/a | ")
            lines[-1] += f"{cell(questions, 'accuracy')} | {cell(questions, 'brier')} | {cell(questions, 'ece')} | {cell(panel, 'accuracy')} | {cell(panel, 'nll')} | {cell(panel, 'ece')} |"
        lines.append("")
        lines.append("Gate (spec WP4): advance only if an RL arm beats `supervised` on held-out greedy reward without a worse held-out "
                     "question Brier or ECE; `rl_only` shows whether calibration degrades under workflow-only reward.")
        print("\n".join(lines))
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
