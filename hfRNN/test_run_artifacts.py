import logging

import torch

from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger


def test_make_run_dir_creates_expected_structure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = make_run_dir("mnist", root="results")

    assert run_dir.exists()
    assert run_dir.parent.name == "mnist"
    assert run_dir.parent.parent.name == "results"
    assert (run_dir / "activity").is_dir()
    assert len(run_dir.name) == 15  # YYYYMMDD_HHMMSS
    assert run_dir.name[8] == "_"


def test_setup_logger_writes_to_file(tmp_path):
    log_path = tmp_path / "train.log"
    logger = setup_logger("test_run_artifacts_logger", log_path)
    logger.info("hello from test")
    for handler in logger.handlers:
        handler.flush()

    assert "hello from test" in log_path.read_text()


def test_setup_logger_is_idempotent_on_handlers(tmp_path):
    log_path = tmp_path / "train.log"
    setup_logger("test_run_artifacts_logger_idempotent", log_path)
    logger = setup_logger("test_run_artifacts_logger_idempotent", log_path)

    assert len(logger.handlers) == 2  # stream + file, not doubled


def test_save_activity_snapshot_writes_pt_and_png(tmp_path):
    hidden = torch.randn(5, 9)
    save_activity_snapshot(hidden, tmp_path, "epoch_01", module_bounds=[3, 6])

    pt_path = tmp_path / "epoch_01.pt"
    png_path = tmp_path / "epoch_01.png"
    assert pt_path.exists()
    assert png_path.exists()

    loaded = torch.load(pt_path)
    assert torch.equal(loaded, hidden)


def test_save_activity_snapshot_works_without_module_bounds(tmp_path):
    hidden = torch.randn(5, 9)
    save_activity_snapshot(hidden, tmp_path, "epoch_01")

    assert (tmp_path / "epoch_01.pt").exists()
    assert (tmp_path / "epoch_01.png").exists()
