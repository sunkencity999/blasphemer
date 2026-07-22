# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Optimization report exporter.

After an abliteration optimization run, this module writes a machine-readable
CSV of every completed trial plus a human-readable Markdown report highlighting
the Pareto-optimal trials (the best trade-offs between KL divergence and
refusal count). This gives users a durable, shareable record of a run instead
of only the ephemeral terminal output.

The core functions operate on plain ``TrialRecord`` objects so they can be
unit-tested without a live Optuna study, a GPU, or a downloaded model.
"""

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrialRecord:
    """A single completed optimization trial, decoupled from Optuna."""

    index: int
    kl_divergence: float
    refusals: int
    total_prompts: int
    direction_index: Optional[float] = None
    parameters: dict = field(default_factory=dict)

    @property
    def refusal_rate(self) -> float:
        """Refusal rate as a percentage of evaluation prompts."""
        if self.total_prompts <= 0:
            return 0.0
        return self.refusals / self.total_prompts * 100.0

    def flat_parameters(self) -> dict[str, str]:
        """Flatten nested per-component parameters into ``component.name`` keys."""
        flat: dict[str, str] = {}
        direction = self.direction_index
        flat["direction_index"] = (
            "per layer" if direction is None else f"{direction:.4f}"
        )
        for component, params in self.parameters.items():
            if isinstance(params, dict):
                for name, value in params.items():
                    try:
                        flat[f"{component}.{name}"] = f"{float(value):.4f}"
                    except (TypeError, ValueError):
                        flat[f"{component}.{name}"] = str(value)
            else:
                flat[str(component)] = str(params)
        return flat


def build_records_from_study(study) -> list[TrialRecord]:
    """
    Build ``TrialRecord`` objects from a completed Optuna study.

    Only trials that completed and carry the user attributes Blasphemer sets
    (``kl_divergence`` / ``refusals``) are included.
    """
    import optuna

    records: list[TrialRecord] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        attrs = trial.user_attrs
        if "kl_divergence" not in attrs or "refusals" not in attrs:
            continue
        records.append(
            TrialRecord(
                index=attrs.get("index", trial.number),
                kl_divergence=float(attrs["kl_divergence"]),
                refusals=int(attrs["refusals"]),
                total_prompts=int(attrs.get("total_prompts", 0)),
                direction_index=attrs.get("direction_index"),
                parameters=attrs.get("parameters", {}) or {},
            )
        )
    return records


def compute_pareto_front(records: list[TrialRecord]) -> list[TrialRecord]:
    """
    Return the Pareto-optimal trials, minimizing both KL divergence and refusals.

    A trial is Pareto-optimal when no other trial is at least as good on both
    objectives and strictly better on at least one.
    """
    front: list[TrialRecord] = []
    for candidate in records:
        dominated = False
        for other in records:
            if other is candidate:
                continue
            not_worse = (
                other.kl_divergence <= candidate.kl_divergence
                and other.refusals <= candidate.refusals
            )
            strictly_better = (
                other.kl_divergence < candidate.kl_divergence
                or other.refusals < candidate.refusals
            )
            if not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)

    # Present the front ordered by fewest refusals, then lowest KL divergence.
    front.sort(key=lambda r: (r.refusals, r.kl_divergence))
    return front


def _markdown_report(
    records: list[TrialRecord],
    pareto: list[TrialRecord],
    model_name: str,
    base_refusals: int,
    total_prompts: int,
    elapsed_seconds: Optional[float],
) -> str:
    lines: list[str] = []
    lines.append(f"# Blasphemer optimization report: `{model_name}`")
    lines.append("")
    lines.append(f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Completed trials:** {len(records)}")
    lines.append(
        f"- **Baseline refusals:** {base_refusals}/{total_prompts}"
        if total_prompts
        else f"- **Baseline refusals:** {base_refusals}"
    )
    if elapsed_seconds is not None:
        mins, secs = divmod(int(elapsed_seconds), 60)
        lines.append(f"- **Total time:** {mins}m {secs}s")
    lines.append("")

    if records:
        best_kl = min(records, key=lambda r: r.kl_divergence)
        best_ref = min(records, key=lambda r: r.refusals)
        lines.append("## Highlights")
        lines.append("")
        lines.append(
            f"- Lowest KL divergence: **{best_kl.kl_divergence:.3f}** "
            f"(trial {best_kl.index}, {best_kl.refusals}/{best_kl.total_prompts} refusals)"
        )
        lines.append(
            f"- Fewest refusals: **{best_ref.refusals}/{best_ref.total_prompts}** "
            f"(trial {best_ref.index}, KL {best_ref.kl_divergence:.3f})"
        )
        lines.append("")

    lines.append("## Pareto-optimal trials")
    lines.append("")
    lines.append(
        "These trials represent the best trade-offs found; no other trial beats "
        "them on both refusals and KL divergence. Lower is better on both axes. "
        "_KL divergence above ~1.0 usually indicates meaningful capability loss._"
    )
    lines.append("")
    lines.append("| Trial | Refusals | Refusal rate | KL divergence |")
    lines.append("| ----: | -------: | -----------: | ------------: |")
    for r in pareto:
        lines.append(
            f"| {r.index} | {r.refusals}/{r.total_prompts} "
            f"| {r.refusal_rate:.1f}% | {r.kl_divergence:.3f} |"
        )
    lines.append("")

    if pareto:
        rec = pareto[0]
        lines.append("## Recommended trial")
        lines.append("")
        lines.append(
            f"Trial **{rec.index}** has the fewest refusals on the Pareto front "
            f"({rec.refusals}/{rec.total_prompts}, KL {rec.kl_divergence:.3f})."
        )
        lines.append("")
        lines.append("Parameters:")
        lines.append("")
        for name, value in rec.flat_parameters().items():
            lines.append(f"- `{name}` = {value}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Generated by [Blasphemer](https://github.com/sunkencity999/blasphemer)._"
    )
    lines.append("")
    return "\n".join(lines)


def export_report(
    records: list[TrialRecord],
    model_name: str,
    output_dir: str | Path,
    base_refusals: int = 0,
    total_prompts: int = 0,
    elapsed_seconds: Optional[float] = None,
) -> Path:
    """
    Write ``trials.csv`` and ``report.md`` into ``output_dir`` and return the dir.

    Safe to call with an empty ``records`` list (produces empty-but-valid files).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pareto = compute_pareto_front(records)

    # --- CSV ---
    # Collect the union of flattened parameter keys so every row lines up.
    param_keys: list[str] = []
    for r in records:
        for key in r.flat_parameters():
            if key not in param_keys:
                param_keys.append(key)

    fieldnames = [
        "index",
        "refusals",
        "total_prompts",
        "refusal_rate_pct",
        "kl_divergence",
        "pareto_optimal",
    ] + param_keys

    pareto_indices = {r.index for r in pareto}
    csv_path = out / "trials.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(records, key=lambda x: x.index):
            row = {
                "index": r.index,
                "refusals": r.refusals,
                "total_prompts": r.total_prompts,
                "refusal_rate_pct": f"{r.refusal_rate:.2f}",
                "kl_divergence": f"{r.kl_divergence:.6f}",
                "pareto_optimal": r.index in pareto_indices,
            }
            row.update(r.flat_parameters())
            writer.writerow(row)

    # --- Markdown ---
    md_path = out / "report.md"
    md_path.write_text(
        _markdown_report(
            records, pareto, model_name, base_refusals, total_prompts, elapsed_seconds
        ),
        encoding="utf-8",
    )

    return out
