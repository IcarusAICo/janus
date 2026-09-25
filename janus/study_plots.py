"""Headless, standalone study figures generated only from an analysis report."""

from pathlib import Path

import numpy as np


def plot_study(report, output_dir, reliability_models=None):
    """Save three PNG/SVG figures without changing the report or its source files.

    Returns ``{figure_name: {"png": absolute_path, "svg": absolute_path}}``.
    Existing figure files are never overwritten. Supply ``reliability_models`` to
    select exact display names for the primary small model, large model, and Jev
    in both reliability and NLL-sensitivity plots; NLL sensitivity additionally
    includes any present Frozen models. The default prefers listwise/larger
    names and always includes the reference.
    """
    models = report.get("models", {})
    reference = report.get("reference")
    if not models or reference not in models:
        raise ValueError("A study report with models and a reference is required")
    names = list(models)
    if reliability_models is None:
        primary = [name for name in names if name != reference
                   and any(token in name.lower() for token in ("listwise", "1.7", "larger"))
                   and not any(token in name.lower() for token in ("small-data", "small_data", "repeat"))]
        reliability_models = (primary or [name for name in names if name != reference])[:2] + [reference]
    if (not reliability_models or len(set(reliability_models)) != len(reliability_models)
            or any(name not in models for name in reliability_models)):
        raise ValueError("Reliability model names must be distinct known models")
    nll_models = list(reliability_models) + [name for name in names
                                           if "frozen" in name.lower() and name not in reliability_models]
    domains = list(models[reference]["by_domain"])
    if not domains or any(set(model["by_domain"]) != set(domains) for model in models.values()):
        raise ValueError("Every model must contain the same nonempty domain set")
    output = Path(output_dir).resolve()
    saved = {name: {kind: str(output / f"{name}.{kind}") for kind in ("png", "svg")}
             for name in ("domain_accuracy", "reliability", "nll_sensitivity")}
    for formats in saved.values():
        for path in formats.values():
            if Path(path).exists():
                raise FileExistsError(f"Refusing to overwrite study figure: {path}")

    # FigureCanvasAgg avoids a GUI backend and pyplot's global figure registry.
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output.mkdir(parents=True, exist_ok=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
              "#555555", "#8C564B", "#9467BD", "#17BECF")
    model_colors = {name: colors[i % len(colors)] for i, name in enumerate(names)}

    def save(figure, name):
        for kind, path in saved[name].items():
            figure.savefig(path, format=kind, dpi=180, bbox_inches="tight", facecolor="white")
        figure.clear()

    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.titlepad": 14}):
        figure = Figure(figsize=(8.2, max(3.2, .52*len(names)+1.8)), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        values = np.array([[models[name]["by_domain"][domain]["raw"]["accuracy"]
                            for domain in domains] for name in names])
        artist = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(domains)), [domain.upper() for domain in domains])
        ax.set_yticks(range(len(names)), names)
        ax.set_title("Raw accuracy on the same held-out decisions, by domain")
        for i in range(len(names)):
            for j in range(len(domains)):
                ax.text(j, i, f"{values[i, j]:.1%}", ha="center", va="center",
                        color="white" if values[i, j] < .6 else "black", fontweight="medium")
        colorbar = figure.colorbar(artist, ax=ax, shrink=.85)
        colorbar.set_label("Accuracy")
        save(figure, "domain_accuracy")

        figure = Figure(figsize=(7.5, 6.0), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1.3,
                label="Perfect calibration", zorder=1)
        for name in reliability_models:
            for mode, linestyle in (("raw", "-"), ("calibrated", "--")):
                bins = [b for b in models[name][mode]["reliability"] if b["count"]]
                x, y = [b["confidence"] for b in bins], [b["accuracy"] for b in bins]
                total = sum(b["count"] for b in bins)
                ax.plot(x, y, linestyle=linestyle, linewidth=1.6,
                        color=model_colors[name], label=f"{name} · {mode}")
                ax.scatter(x, y, s=[14+130*b["count"]/total for b in bins],
                           color=model_colors[name], alpha=.75, zorder=3)
        ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Mean confidence in populated bin",
               ylabel="Accuracy in populated bin", title="Reliability before and after calibration")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -.14), ncol=2, frameon=False, fontsize=8)
        figure.supxlabel("Marker area scales with bin count; empty bins are omitted.", fontsize=9)
        save(figure, "reliability")

        figure = Figure(figsize=(8.2, 6.2 if len(nll_models) <= 4 else 7.3), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        floors = report["nll_sensitivity_floors"]
        for name in nll_models:
            for mode, linestyle in (("raw", "-"), ("calibrated", "--")):
                values = [models[name][mode]["nll_sensitivity"][str(f)] for f in floors]
                ax.plot(floors, values, marker="o", markersize=3.5, linestyle=linestyle,
                        linewidth=1.6, color=model_colors[name], label=f"{name} · {mode}")
        ax.axvline(report["nll_probability_floor"], color="#555555", alpha=.5, linewidth=1)
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set(xlabel="Common probability floor inside log(p), no renormalization",
               ylabel="Mean NLL (nats; lower is better)",
               title="NLL sensitivity to zero and very small probabilities")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -.17), ncol=2, frameon=False, fontsize=8)
        zeros = [f"{mode}: {models[reference][mode]['target_zero_probability_count']} zero-probability target answers"
                 for mode in ("raw", "calibrated") if models[reference][mode]["target_zero_probability_count"]]
        caption = (f"{reference} " + "; ".join(zeros) + ". Literal NLL is infinite."
                   if zeros else "All curves use an identical floor; accuracy and Brier are unaffected.")
        figure.supxlabel(caption, fontsize=9)
        save(figure, "nll_sensitivity")
    return saved
