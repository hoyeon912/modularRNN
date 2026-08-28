"""Collect DQN experiment logs and build architecture-level report inputs."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Run:
    path: Path
    architecture: str
    rows: list[dict[str, Any]]


def classify_architecture(path: Path) -> str | None:
    name = path.as_posix().lower().replace("-", "_").replace(" ", "_")
    if "cnn_modular_rnn_dqn" in name:
        return "cnn_modular_rnn_dqn"
    if "cnn_rnn_dqn" in name:
        return "cnn_rnn_dqn"
    if "cnn_dqn" in name:
        return "cnn_dqn"
    if "dqn" in name:
        return "dqn"
    return None


def discover_runs(results_root: Path) -> list[Run]:
    runs: list[Run] = []
    for results_path in sorted(results_root.rglob("results.json")):
        run_dir = results_path.parent
        architecture = classify_architecture(run_dir)
        if architecture is None or run_dir.name.lower().startswith("reinforce_"):
            continue
        if any((child / "results.json").exists() for child in run_dir.iterdir() if child.is_dir()):
            continue
        try:
            rows = json.loads(results_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rows, list):
            continue
        runs.append(Run(run_dir, architecture, rows))
    return runs


def summarize_rewards(
    rows: list[dict[str, Any]], recent_window: int = 100
) -> dict[str, float | int] | None:
    rewards = [row.get("reward") for row in rows]
    numeric = [float(value) for value in rewards if isinstance(value, (int, float))]
    if not numeric:
        return None
    recent = numeric[-recent_window:]
    return {
        "episodes": len(numeric),
        "best_reward": max(numeric),
        "final_reward": numeric[-1],
        "recent_mean_reward": sum(recent) / len(recent),
    }


ARCHITECTURE_TITLES = {
    "dqn": "DQN",
    "cnn_dqn": "CNN-DQN",
    "cnn_rnn_dqn": "CNN-RNN-DQN",
    "cnn_modular_rnn_dqn": "CNN-Modular-RNN-DQN",
}


def infer_environment(path: Path) -> str:
    return "LunarLander" if "lunarlander" in path.as_posix().lower() else "CartPole"


def _sample_rows(rows: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    step = max(1, len(rows) // (limit - 1))
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled[:limit]


def build_artifact(
    architecture: str, runs: list[Run], results_root: Path
) -> dict[str, Any]:
    display_name = ARCHITECTURE_TITLES[architecture]
    title = f"{display_name} Architecture Report"
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    curve_points_per_run = max(2, 2000 // max(1, len(runs)))
    for run in runs:
        summary = summarize_rewards(run.rows)
        if summary is None:
            continue
        relative = run.path.relative_to(results_root)
        environment = infer_environment(relative)
        summaries.append(
            {
                "experiment": relative.parts[0],
                "run": relative.name,
                "environment": environment,
                "episodes": summary["episodes"],
                "best_reward": summary["best_reward"],
                "final_reward": summary["final_reward"],
                "recent_mean_reward": summary["recent_mean_reward"],
                "source_path": (relative / "results.json").as_posix(),
            }
        )
        for row in _sample_rows(run.rows, limit=curve_points_per_run):
            episode, reward = row.get("episode"), row.get("reward")
            if isinstance(episode, (int, float)) and isinstance(reward, (int, float)):
                curves.append(
                    {
                        "run": relative.name,
                        "environment": environment,
                        "episode": episode,
                        "reward": reward,
                    }
                )
    summaries.sort(key=lambda row: row["recent_mean_reward"], reverse=True)
    environments = sorted({row["environment"] for row in summaries})
    episode_total = sum(row["episodes"] for row in summaries)
    source_id = f"{architecture}_logs"
    summary_text = (
        f"## Executive Summary\n\n"
        f"- **{len(summaries)} valid runs** were collected across "
        f"{', '.join(environments) if environments else 'no identified environment'}.\n"
        f"- The reports cover **{episode_total:,} logged episodes**. Rankings use the mean "
        f"reward over each run's last 100 valid episodes.\n"
        f"- Reward scales differ by environment, so cross-environment ranking should not be "
        f"treated as an architecture comparison."
    )
    ranking_lines = [
        "## Strongest recent runs",
        "",
        "The table ranks runs by mean reward over their last 100 valid episodes. Compare rows within the same environment.",
        "",
        "| Rank | Experiment / run | Environment | Episodes | Best | Final | Recent mean |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summaries[:25], 1):
        ranking_lines.append(
            f"| {rank} | {row['experiment']} / {row['run']} | {row['environment']} | "
            f"{row['episodes']} | {row['best_reward']:.3g} | {row['final_reward']:.3g} | "
            f"{row['recent_mean_reward']:.3g} |"
        )
    inventory_lines = [
        "## Complete log inventory",
        "",
        "Every included training log is listed below. These paths are relative to `results/`.",
        "",
    ]
    inventory_lines.extend(f"- `{row['source_path']}`" for row in summaries)
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": f"Repository-wide training-log review for the {display_name} architecture.",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": [],
            "tables": [],
            "sources": [{"id": source_id, "label": "Repository DQN result logs", "path": "results"}],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {"id": "summary", "type": "markdown", "body": summary_text, "sourceId": source_id},
                {"id": "ranking", "type": "markdown", "body": "\n".join(ranking_lines), "sourceId": source_id},
                {"id": "inventory", "type": "markdown", "body": "\n".join(inventory_lines), "sourceId": source_id},
                {
                    "id": "caveats",
                    "type": "markdown",
                    "body": "## Limitations\n\n- Runs have different training budgets and hyperparameters.\n- The last-100 mean uses fewer observations when a run has fewer than 100 valid episodes.\n- This is descriptive log aggregation, not a controlled architecture comparison.",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "run_summary": summaries,
                "reward_curves": curves,
            },
            "accessIssues": [],
        },
        "sources": [
            {
                "id": source_id,
                "query": {
                    "engine": "python",
                    "description": "Discovers results/**/results.json, classifies paths by architecture, excludes reinforce_* comparisons, and computes reward summaries.",
                    "language": "python",
                    "tables_used": sorted({row["source_path"] for row in summaries}),
                    "executed_at": generated_at,
                },
            }
        ],
    }


def write_artifacts(results_root: Path, output_dir: Path) -> list[Path]:
    grouped: dict[str, list[Run]] = {name: [] for name in ARCHITECTURE_TITLES}
    for run in discover_runs(results_root):
        grouped[run.architecture].append(run)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for architecture, runs in grouped.items():
        output_path = output_dir / f"{architecture}.artifact.json"
        output_path.write_text(
            json.dumps(build_artifact(architecture, runs, results_root), indent=2) + "\n"
        )
        outputs.append(output_path)
    return outputs


def _reward_chart(runs: list[Run], environment: str, results_root: Path) -> str:
    selected = [run for run in runs if infer_environment(run.path) == environment][:8]
    series = []
    for run in selected:
        points = [
            (float(row["episode"]), float(row["reward"]))
            for row in _sample_rows(run.rows, 60)
            if isinstance(row.get("episode"), (int, float))
            and isinstance(row.get("reward"), (int, float))
        ]
        if points:
            series.append((run.path.relative_to(results_root).as_posix(), points))
    if not series:
        return ""
    all_points = [point for _, points in series for point in points]
    x_min, x_max = min(x for x, _ in all_points), max(x for x, _ in all_points)
    y_min, y_max = min(y for _, y in all_points), max(y for _, y in all_points)
    x_span, y_span = max(1.0, x_max - x_min), max(1.0, y_max - y_min)
    colors = ("#2563eb", "#d97706", "#7c3aed", "#0891b2", "#be123c", "#4d7c0f", "#475569", "#c2410c")
    lines, legend = [], []
    for index, (label, points) in enumerate(series):
        coords = " ".join(
            f"{48 + 684 * (x - x_min) / x_span:.1f},{18 + 214 * (y_max - y) / y_span:.1f}"
            for x, y in points
        )
        color = colors[index]
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.6" opacity=".82"/>')
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(label)}</span>'
        )
    return (
        f'<div class="chart"><h3>{environment} reward trajectories</h3>'
        '<p>Up to eight runs, sampled to at most 60 points per run.</p>'
        '<svg viewBox="0 0 760 260" role="img" aria-label="Reward by episode">'
        '<line x1="48" y1="232" x2="732" y2="232" stroke="#94a3b8"/>'
        '<line x1="48" y1="18" x2="48" y2="232" stroke="#94a3b8"/>'
        + "".join(lines)
        + f'<text x="48" y="252">episode {x_min:g}</text><text x="680" y="252">{x_max:g}</text>'
        + f'<text x="4" y="28">{y_max:.3g}</text><text x="4" y="232">{y_min:.3g}</text></svg>'
        + '<div class="legend">' + "".join(legend) + "</div></div>"
    )


def _architecture_html(architecture: str, runs: list[Run], results_root: Path) -> str:
    rows = []
    for run in runs:
        summary = summarize_rewards(run.rows)
        if summary is None:
            continue
        relative = run.path.relative_to(results_root)
        rows.append({
            **summary,
            "experiment": relative.parts[0],
            "run": relative.name,
            "environment": infer_environment(relative),
            "source_path": (relative / "results.json").as_posix(),
        })
    rows.sort(key=lambda row: row["recent_mean_reward"], reverse=True)
    total_episodes = sum(row["episodes"] for row in rows)
    table_rows = "".join(
        "<tr>"
        f"<td>{index}</td><td>{html.escape(row['experiment'])}</td><td>{html.escape(row['run'])}</td>"
        f"<td>{row['environment']}</td><td>{row['episodes']:,}</td>"
        f"<td>{row['best_reward']:.3g}</td><td>{row['final_reward']:.3g}</td>"
        f"<td>{row['recent_mean_reward']:.3g}</td></tr>"
        for index, row in enumerate(rows[:25], 1)
    )
    inventory = "".join(
        f"<li><code>{html.escape(row['source_path'])}</code></li>" for row in rows
    )
    charts = "".join(
        _reward_chart(runs, environment, results_root)
        for environment in ("CartPole", "LunarLander")
    )
    title = ARCHITECTURE_TITLES[architecture]
    return f"""
    <section id="{architecture}">
      <h2>{title}</h2>
      <div class="metrics"><div><strong>{len(rows)}</strong><span>valid runs</span></div><div><strong>{total_episodes:,}</strong><span>logged episodes</span></div></div>
      <p>Recent mean is calculated over the last 100 valid reward observations. Compare values within the same environment because task reward scales differ.</p>
      {charts}
      <h3>Top runs by recent mean reward</h3>
      <div class="table-wrap"><table><thead><tr><th>#</th><th>Experiment</th><th>Run</th><th>Environment</th><th>Episodes</th><th>Best</th><th>Final</th><th>Recent mean</th></tr></thead><tbody>{table_rows}</tbody></table></div>
      <details><summary>All included logs ({len(rows)})</summary><ul class="inventory">{inventory}</ul></details>
    </section>"""


def write_combined_html(results_root: Path, output_path: Path) -> Path:
    grouped: dict[str, list[Run]] = {name: [] for name in ARCHITECTURE_TITLES}
    for run in discover_runs(results_root):
        grouped[run.architecture].append(run)
    sections = "".join(
        _architecture_html(architecture, grouped[architecture], results_root)
        for architecture in ARCHITECTURE_TITLES
    )
    navigation = "".join(
        f'<a href="#{key}">{title}</a>' for key, title in ARCHITECTURE_TITLES.items()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DQN Architecture Log Report</title><style>
:root{{--bg:#f8fafc;--card:#fff;--ink:#172033;--muted:#64748b;--line:#dbe3ee;--accent:#1d4ed8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}}main{{max-width:1180px;margin:auto;padding:40px 22px 80px}}header{{margin-bottom:30px}}h1{{font-size:clamp(2rem,5vw,3.5rem);margin:.1em 0}}h2{{font-size:2rem;margin-top:0}}h3{{margin-top:28px}}.sub{{color:var(--muted);max-width:75ch}}nav{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}}nav a{{color:var(--accent);background:#eaf0ff;padding:7px 12px;border-radius:999px;text-decoration:none}}section{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:28px;margin:24px 0;box-shadow:0 12px 30px #0f172a0a}}.metrics{{display:flex;gap:14px;flex-wrap:wrap}}.metrics div{{min-width:170px;padding:15px 18px;background:#f1f5f9;border-radius:12px}}.metrics strong{{display:block;font-size:1.6rem}}.metrics span{{color:var(--muted)}}.chart{{border:1px solid var(--line);border-radius:14px;padding:16px;margin:20px 0}}.chart h3{{margin:0}}.chart p{{color:var(--muted);margin:2px 0 8px}}svg{{width:100%;height:auto}}svg text{{font-size:11px;fill:var(--muted)}}.legend{{display:flex;gap:8px 16px;flex-wrap:wrap;font-size:12px}}.legend span{{display:flex;align-items:center;gap:5px}}.legend i{{width:10px;height:10px;border-radius:50%}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right}}th:nth-child(2),th:nth-child(3),th:nth-child(4),td:nth-child(2),td:nth-child(3),td:nth-child(4){{text-align:left}}th{{color:var(--muted);font-size:12px}}details{{margin-top:24px}}summary{{cursor:pointer;font-weight:650}}.inventory{{columns:2;word-break:break-all}}code{{font-size:12px}}@media(max-width:700px){{section{{padding:20px}}.inventory{{columns:1}}}}
</style></head><body><main><header><p class="sub">Repository-wide training-log inventory</p><h1>DQN Architecture Log Report</h1><p class="sub">Generated from every valid <code>results/**/results.json</code> path classified as a DQN-style architecture. <code>reinforce_*</code> comparison runs and duplicated root summaries are excluded.</p><nav>{navigation}</nav></header>{sections}<section><h2>Limitations</h2><ul><li>Training budgets and hyperparameters differ between runs.</li><li>Charts show a bounded sample of runs and episodes; the tables and inventories use all valid logs.</li><li>This is descriptive aggregation, not a controlled architecture benchmark.</li></ul></section></main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("reports/dqn_architectures"))
    parser.add_argument("--combined-html", type=Path)
    args = parser.parse_args()
    if args.combined_html:
        print(write_combined_html(args.results, args.combined_html))
        return
    for path in write_artifacts(args.results, args.output):
        print(path)


if __name__ == "__main__":
    main()
