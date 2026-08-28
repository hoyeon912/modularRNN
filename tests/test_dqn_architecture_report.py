import json
from pathlib import Path

import pytest

from scripts.dqn_architecture_report import (
    Run,
    build_artifact,
    classify_architecture,
    discover_runs,
    summarize_rewards,
    write_combined_html,
    write_artifacts,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("results/lunarlander_dqn/seed1", "dqn"),
        ("results/dqn_cartpole_hparam_search/trial1", "dqn"),
        ("results/best CNN-DQN", "cnn_dqn"),
        ("results/lunarlander_cnn_dqn/seed2", "cnn_dqn"),
        ("results/cnn_rnn_dqn_search/trial1", "cnn_rnn_dqn"),
        ("results/lunarlander_cnn_rnn_dqn/seed1", "cnn_rnn_dqn"),
        ("results/cnn_modular_rnn_dqn_search/trial1", "cnn_modular_rnn_dqn"),
    ],
)
def test_classify_architecture_uses_the_most_specific_model_name(path, expected):
    assert classify_architecture(Path(path)) == expected


def test_discover_runs_excludes_reinforce_comparisons_and_root_summaries(tmp_path):
    td = tmp_path / "cnn_modular_rnn_dqn_goal_search" / "td_h150"
    reinforce = tmp_path / "cnn_modular_rnn_dqn_goal_search" / "reinforce_h150"
    root = tmp_path / "lunarlander_dqn"
    for directory in (td, reinforce, root / "seed1"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.json").write_text(
            json.dumps([{"episode": 1, "reward": 10.0}])
        )
    (root / "results.json").write_text(json.dumps([{"episode": 1, "reward": 5.0}]))

    runs = discover_runs(tmp_path)

    assert [run.path.relative_to(tmp_path).as_posix() for run in runs] == [
        "cnn_modular_rnn_dqn_goal_search/td_h150",
        "lunarlander_dqn/seed1",
    ]


def test_summarize_rewards_reports_best_final_and_recent_window():
    rows = [
        {"episode": 1, "reward": 1.0},
        {"episode": 2, "reward": 3.0},
        {"episode": 3, "reward": 2.0},
    ]

    summary = summarize_rewards(rows, recent_window=2)

    assert summary == {
        "episodes": 3,
        "best_reward": 3.0,
        "final_reward": 2.0,
        "recent_mean_reward": 2.5,
    }


def test_summarize_rewards_ignores_rows_without_numeric_rewards():
    assert summarize_rewards([{"episode": 1, "reward": None}, {"episode": 2}]) is None


def test_build_artifact_preserves_report_contract_and_source_paths(tmp_path):
    run_path = tmp_path / "lunarlander_dqn" / "seed1"
    run = Run(
        path=run_path,
        architecture="dqn",
        rows=[
            {"episode": 1, "reward": -10.0},
            {"episode": 2, "reward": 20.0},
        ],
    )

    artifact = build_artifact("dqn", [run], tmp_path)

    assert artifact["surface"] == "report"
    assert artifact["manifest"]["blocks"][0]["body"].startswith("# DQN Architecture Report")
    assert artifact["manifest"]["title"] == "DQN Architecture Report"
    assert artifact["snapshot"]["datasets"]["run_summary"][0]["source_path"] == (
        "lunarlander_dqn/seed1/results.json"
    )
    assert artifact["snapshot"]["datasets"]["reward_curves"] == [
        {
            "run": "seed1",
            "environment": "LunarLander",
            "episode": 1,
            "reward": -10.0,
        },
        {
            "run": "seed1",
            "environment": "LunarLander",
            "episode": 2,
            "reward": 20.0,
        },
    ]


def test_write_artifacts_creates_one_json_per_architecture(tmp_path):
    results_root = tmp_path / "results"
    for group in (
        "lunarlander_dqn",
        "lunarlander_cnn_dqn",
        "lunarlander_cnn_rnn_dqn",
        "cnn_modular_rnn_dqn_search",
    ):
        run_dir = results_root / group / "seed1"
        run_dir.mkdir(parents=True)
        (run_dir / "results.json").write_text(
            json.dumps([{"episode": 1, "reward": 1.0}])
        )

    outputs = write_artifacts(results_root, tmp_path / "reports")

    assert {path.name for path in outputs} == {
        "dqn.artifact.json",
        "cnn_dqn.artifact.json",
        "cnn_rnn_dqn.artifact.json",
        "cnn_modular_rnn_dqn.artifact.json",
    }


def test_build_artifact_bounds_reward_curve_snapshot_to_2000_rows(tmp_path):
    runs = [
        Run(
            path=tmp_path / "dqn_cartpole_hparam_search" / f"trial{run_index}",
            architecture="dqn",
            rows=[{"episode": episode, "reward": float(episode)} for episode in range(100)],
        )
        for run_index in range(30)
    ]

    artifact = build_artifact("dqn", runs, tmp_path)

    assert len(artifact["snapshot"]["datasets"]["reward_curves"]) <= 2000


def test_file_backed_artifact_does_not_claim_sql_backed_metric_cards(tmp_path):
    run = Run(
        path=tmp_path / "lunarlander_dqn" / "seed1",
        architecture="dqn",
        rows=[{"episode": 1, "reward": 1.0}],
    )

    artifact = build_artifact("dqn", [run], tmp_path)

    assert artifact["manifest"]["cards"] == []
    assert "metric-strip" not in {
        block["type"] for block in artifact["manifest"]["blocks"]
    }


def test_file_backed_artifact_keeps_evidence_in_markdown_and_lists_every_log(tmp_path):
    runs = [
        Run(
            path=tmp_path / "lunarlander_dqn" / seed,
            architecture="dqn",
            rows=[{"episode": 1, "reward": reward}],
        )
        for seed, reward in (("seed1", 1.0), ("seed2", 2.0))
    ]

    artifact = build_artifact("dqn", runs, tmp_path)
    block_types = {block["type"] for block in artifact["manifest"]["blocks"]}
    report_text = "\n".join(block["body"] for block in artifact["manifest"]["blocks"])

    assert block_types == {"markdown"}
    assert "lunarlander_dqn/seed1/results.json" in report_text
    assert "lunarlander_dqn/seed2/results.json" in report_text


def test_write_combined_html_contains_all_architectures_and_log_paths(tmp_path):
    results_root = tmp_path / "results"
    groups = {
        "lunarlander_dqn": "DQN",
        "lunarlander_cnn_dqn": "CNN-DQN",
        "lunarlander_cnn_rnn_dqn": "CNN-RNN-DQN",
        "cnn_modular_rnn_dqn_search": "CNN-Modular-RNN-DQN",
    }
    for group in groups:
        run_dir = results_root / group / "seed1"
        run_dir.mkdir(parents=True)
        (run_dir / "results.json").write_text(
            json.dumps([{"episode": 1, "reward": 1.0}])
        )

    output = write_combined_html(results_root, tmp_path / "dqn-report.html")
    html = output.read_text()

    assert output.name == "dqn-report.html"
    for group, title in groups.items():
        assert f">{title}<" in html
        assert f"{group}/seed1/results.json" in html
    assert "https://" not in html
    assert "<script src=" not in html
