import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def make_run_dir(env_name: str, root: str = "results") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / env_name / timestamp
    (run_dir / "activity").mkdir(parents=True)
    return run_dir


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_activity_snapshot(
    hidden_states: torch.Tensor,
    activity_dir: Path,
    step_label: str,
    module_bounds: list[int] | None = None,
) -> None:
    """`hidden_states` is one sample's trajectory, shape (seq_len, hidden_units)."""
    activity_dir = Path(activity_dir)
    activity_dir.mkdir(parents=True, exist_ok=True)
    hidden_states = hidden_states.detach().cpu()

    torch.save(hidden_states, activity_dir / f"{step_label}.pt")

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(hidden_states.T.numpy(), aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("timestep")
    ax.set_ylabel("unit")
    ax.set_title(step_label)
    fig.colorbar(im, ax=ax)
    if module_bounds:
        for bound in module_bounds:
            ax.axhline(bound - 0.5, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(activity_dir / f"{step_label}.png")
    plt.close(fig)
