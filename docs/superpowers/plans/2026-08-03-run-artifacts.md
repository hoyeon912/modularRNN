# Run Artifacts (figure, log, RNN unit-activity saving) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every training script in `RNN/`, `biRNN/`, `modRNN/`, `hfRNN/` (both `test_mnist.py` and `test_cartpole.py`, 8 scripts total) saves a persisted training-curve figure, a text log file, and periodic snapshots of RNN hidden-unit activity, all under a per-run directory `results/{env_name}/{YYYYMMDD_HHMMSS}/`.

**Architecture:** This repo already duplicates shared code verbatim across its four model directories (`live_plot.py` is byte-identical in all four today) rather than using a shared package — each directory is self-contained and independently runnable. This plan follows that convention: a new `run_artifacts.py` module is built once (in `RNN/`, with TDD) and copied verbatim into the other three directories, exactly like `live_plot.py` already is.

**Tech Stack:** Python, PyTorch, matplotlib, pytest. No new dependencies — `torch` and `matplotlib` are already used by `live_plot.py`.

## Global Constraints

- Shared logic (`run_artifacts.py`, the `LiveTrainingPlot.save()` addition) is copied **verbatim** into each of `RNN/`, `biRNN/`, `modRNN/`, `hfRNN/` — no shared package, no `sys.path` tricks. This matches the existing duplication convention in the repo.
- `return_hidden` defaults to `False` on every model's `forward()`. No existing call site or existing test may change behavior when the argument is omitted.
- Run directories are `results/{env_name}/{YYYYMMDD_HHMMSS}/`, created relative to the current working directory (matches how `mnist_results.json` is written to cwd today).
- Run tests with the project's venv, not the system/pyenv Python: `/Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest <file> -q`. The bare `pytest`/`python` on `PATH` resolve to a pyenv shim without `torch` installed.
- All new/changed Python uses the same style already in these files: 4-space indent, type hints on public functions, no docstrings unless documenting a non-obvious WHY (matching the existing `ModularRNNCell`/mask-init comments).

---

### Task 1: Build `run_artifacts.py` + `LiveTrainingPlot.save()` in `RNN/` (TDD)

**Files:**
- Create: `RNN/run_artifacts.py`
- Create: `RNN/test_run_artifacts.py`
- Modify: `RNN/live_plot.py`
- Modify: `RNN/test_live_plot.py`

**Interfaces:**
- Produces: `run_artifacts.make_run_dir(env_name: str, root: str = "results") -> pathlib.Path`
- Produces: `run_artifacts.setup_logger(name: str, log_path: pathlib.Path) -> logging.Logger`
- Produces: `run_artifacts.save_activity_snapshot(hidden_states: torch.Tensor, activity_dir: pathlib.Path, step_label: str, module_bounds: list[int] | None = None) -> None`
- Produces: `LiveTrainingPlot.save(self, path) -> None`

- [ ] **Step 1: Write the failing tests for `run_artifacts.py`**

Create `RNN/test_run_artifacts.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/hoyeon/Codes/modularRNN/RNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_run_artifacts.py -v`
Expected: FAIL / collection error with `ModuleNotFoundError: No module named 'run_artifacts'`

- [ ] **Step 3: Implement `run_artifacts.py`**

Create `RNN/run_artifacts.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/hoyeon/Codes/modularRNN/RNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_run_artifacts.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Write the failing tests for `LiveTrainingPlot.save()`**

Append to `RNN/test_live_plot.py`:

```python
def test_save_writes_figure_to_path(tmp_path):
    plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)
    out_path = tmp_path / "curve.png"
    plot.save(out_path)
    assert out_path.exists()


def test_save_is_noop_when_disabled(tmp_path):
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    out_path = tmp_path / "curve.png"
    plot.save(out_path)  # must not raise
    assert not out_path.exists()
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `cd /Users/hoyeon/Codes/modularRNN/RNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_live_plot.py -v`
Expected: FAIL with `AttributeError: 'LiveTrainingPlot' object has no attribute 'save'`

- [ ] **Step 7: Implement `LiveTrainingPlot.save()`**

In `RNN/live_plot.py`, add this method to the `LiveTrainingPlot` class, immediately after `update()` (i.e. as the last method in the class):

```python
    def save(self, path) -> None:
        if not self.enabled:
            return
        self.fig.savefig(path)
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `cd /Users/hoyeon/Codes/modularRNN/RNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_live_plot.py test_run_artifacts.py -v`
Expected: PASS (all tests)

- [ ] **Step 9: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add RNN/run_artifacts.py RNN/test_run_artifacts.py RNN/live_plot.py RNN/test_live_plot.py
git commit -m "$(cat <<'EOF'
Add run_artifacts helper module and LiveTrainingPlot.save() to RNN/

Provides make_run_dir, setup_logger, and save_activity_snapshot, the
shared building blocks for per-run figure/log/activity saving.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Copy shared infra to `biRNN/`, `modRNN/`, `hfRNN/`

**Files:**
- Create: `biRNN/run_artifacts.py`, `biRNN/test_run_artifacts.py`
- Create: `modRNN/run_artifacts.py`, `modRNN/test_run_artifacts.py`
- Create: `hfRNN/run_artifacts.py`, `hfRNN/test_run_artifacts.py`
- Modify: `biRNN/live_plot.py`, `biRNN/test_live_plot.py`
- Modify: `modRNN/live_plot.py`, `modRNN/test_live_plot.py`
- Modify: `hfRNN/live_plot.py` (no `hfRNN/test_live_plot.py` exists today — don't create one, out of scope)

**Interfaces:**
- Consumes: the exact file contents of `RNN/run_artifacts.py`, `RNN/test_run_artifacts.py`, and the `LiveTrainingPlot.save()` method + its two tests from Task 1.
- Produces: same as Task 1, replicated in the other three directories.

This task is a verbatim copy — the content is identical across all four directories (matching how `live_plot.py` already exists identically in all four). No new design decisions.

- [ ] **Step 1: Copy `run_artifacts.py` and its test to the other three directories**

```bash
cd /Users/hoyeon/Codes/modularRNN
for d in biRNN modRNN hfRNN; do
  cp RNN/run_artifacts.py "$d/run_artifacts.py"
  cp RNN/test_run_artifacts.py "$d/test_run_artifacts.py"
done
```

- [ ] **Step 2: Add `LiveTrainingPlot.save()` to the other three `live_plot.py` files**

For `biRNN/live_plot.py`, `modRNN/live_plot.py`, and `hfRNN/live_plot.py`, add the same method added in Task 1 Step 7, immediately after `update()`:

```python
    def save(self, path) -> None:
        if not self.enabled:
            return
        self.fig.savefig(path)
```

- [ ] **Step 3: Add the two `save()` tests to `biRNN/test_live_plot.py` and `modRNN/test_live_plot.py`**

Append the same two tests from Task 1 Step 5 to both files:

```python
def test_save_writes_figure_to_path(tmp_path):
    plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)
    out_path = tmp_path / "curve.png"
    plot.save(out_path)
    assert out_path.exists()


def test_save_is_noop_when_disabled(tmp_path):
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    out_path = tmp_path / "curve.png"
    plot.save(out_path)  # must not raise
    assert not out_path.exists()
```

(`hfRNN` has no `test_live_plot.py` to append to — skip it there.)

- [ ] **Step 4: Run the new/changed tests in each directory**

```bash
cd /Users/hoyeon/Codes/modularRNN/biRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_run_artifacts.py test_live_plot.py -v
cd /Users/hoyeon/Codes/modularRNN/modRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_run_artifacts.py test_live_plot.py -v
cd /Users/hoyeon/Codes/modularRNN/hfRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_run_artifacts.py -v
```
Expected: PASS in all three (biRNN/modRNN: 7 tests each; hfRNN: 5 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add biRNN/run_artifacts.py biRNN/test_run_artifacts.py biRNN/live_plot.py biRNN/test_live_plot.py \
        modRNN/run_artifacts.py modRNN/test_run_artifacts.py modRNN/live_plot.py modRNN/test_live_plot.py \
        hfRNN/run_artifacts.py hfRNN/test_run_artifacts.py hfRNN/live_plot.py
git commit -m "$(cat <<'EOF'
Copy run_artifacts module and LiveTrainingPlot.save() to biRNN/modRNN/hfRNN

Verbatim copy from RNN/, matching this repo's existing convention of
duplicating shared training-script infrastructure across model dirs.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `RNN/model.py` — expose hidden-state trajectory (`SimpleRNN`)

**Files:**
- Modify: `RNN/model.py`
- Test: `RNN/test_model.py`

**Interfaces:**
- Produces: `SimpleRNN.forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]`. When `return_hidden=True`, second element has shape `(batch, seq_len, hidden_size*2)`.

- [ ] **Step 1: Write the failing tests**

Append to `RNN/test_model.py` (check the file's existing imports first — it should already import `SimpleRNN` and `get_device`; add `torch` if not already imported):

```python
def test_forward_return_hidden_shape():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 2)
    assert hidden.shape == (3, 5, 16)


def test_forward_return_hidden_shape_all_mode():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="all")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 5, 2)
    assert hidden.shape == (3, 5, 16)


def test_forward_without_return_hidden_returns_tensor_only():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/hoyeon/Codes/modularRNN/RNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_model.py -k return_hidden -v`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'return_hidden'`

- [ ] **Step 3: Implement `return_hidden` in `SimpleRNN.forward`**

In `RNN/model.py`, replace the `forward` method:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        outputs, hidden = self.rnn(x)
        # outputs: (batch, seq_len, hidden_size*2)
        # hidden: (2, batch, hidden_size) -- num_layers=1, bidirectional=True -> [forward, backward]

        if self.output_mode == "all":
            return self.output_proj(outputs)

        forward_last = hidden[0]
        backward_last = hidden[1]
        combined = torch.cat([forward_last, backward_last], dim=1)
        return self.output_proj(combined)
```

with:

```python
    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        outputs, hidden = self.rnn(x)
        # outputs: (batch, seq_len, hidden_size*2)
        # hidden: (2, batch, hidden_size) -- num_layers=1, bidirectional=True -> [forward, backward]

        if self.output_mode == "all":
            result = self.output_proj(outputs)
        else:
            forward_last = hidden[0]
            backward_last = hidden[1]
            combined = torch.cat([forward_last, backward_last], dim=1)
            result = self.output_proj(combined)

        if return_hidden:
            return result, outputs
        return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/hoyeon/Codes/modularRNN/RNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_model.py -v`
Expected: PASS (all tests, including the 3 new ones and every pre-existing test unchanged)

- [ ] **Step 5: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add RNN/model.py RNN/test_model.py
git commit -m "$(cat <<'EOF'
Add return_hidden option to SimpleRNN.forward

Exposes the full per-timestep hidden trajectory so training scripts can
snapshot RNN unit activity. Default False preserves existing behavior.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `biRNN/model.py` — expose hidden-state trajectory (`BidirectionalRNN`)

**Files:**
- Modify: `biRNN/model.py`
- Test: `biRNN/test_model.py`

**Interfaces:**
- Produces: `BidirectionalRNN.forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]`. When `return_hidden=True`, second element has shape `(batch, seq_len, hidden_size*2)`.

- [ ] **Step 1: Write the failing tests**

Append to `biRNN/test_model.py` (add `import torch` if not already present; the file should already import `BidirectionalRNN`, `get_device`):

```python
def test_forward_return_hidden_shape():
    model = BidirectionalRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 2)
    assert hidden.shape == (3, 5, 16)


def test_forward_return_hidden_shape_all_mode():
    model = BidirectionalRNN(input_size=4, hidden_size=8, output_size=2, output_mode="all")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 5, 2)
    assert hidden.shape == (3, 5, 16)


def test_forward_without_return_hidden_returns_tensor_only():
    model = BidirectionalRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/hoyeon/Codes/modularRNN/biRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_model.py -k return_hidden -v`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'return_hidden'`

- [ ] **Step 3: Implement `return_hidden` in `BidirectionalRNN.forward`**

In `biRNN/model.py`, replace the `forward` method:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        batch_size, seq_len, _ = x.shape

        h_fwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        fwd_states = []
        for t in range(seq_len):
            h_fwd = self.fwd_cell(x[:, t, :], h_fwd)
            fwd_states.append(h_fwd)

        h_bwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        bwd_states = [None] * seq_len
        for t in reversed(range(seq_len)):
            h_bwd = self.bwd_cell(x[:, t, :], h_bwd)
            bwd_states[t] = h_bwd

        if self.output_mode == "all":
            combined = torch.stack(
                [torch.cat([fwd_states[t], bwd_states[t]], dim=1) for t in range(seq_len)],
                dim=1,
            )
            return self.output_proj(combined)

        combined = torch.cat([fwd_states[-1], bwd_states[0]], dim=1)
        return self.output_proj(combined)
```

with:

```python
    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        batch_size, seq_len, _ = x.shape

        h_fwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        fwd_states = []
        for t in range(seq_len):
            h_fwd = self.fwd_cell(x[:, t, :], h_fwd)
            fwd_states.append(h_fwd)

        h_bwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        bwd_states = [None] * seq_len
        for t in reversed(range(seq_len)):
            h_bwd = self.bwd_cell(x[:, t, :], h_bwd)
            bwd_states[t] = h_bwd

        outputs = torch.stack(
            [torch.cat([fwd_states[t], bwd_states[t]], dim=1) for t in range(seq_len)],
            dim=1,
        )

        if self.output_mode == "all":
            result = self.output_proj(outputs)
        else:
            combined = torch.cat([fwd_states[-1], bwd_states[0]], dim=1)
            result = self.output_proj(combined)

        if return_hidden:
            return result, outputs
        return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/hoyeon/Codes/modularRNN/biRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_model.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add biRNN/model.py biRNN/test_model.py
git commit -m "$(cat <<'EOF'
Add return_hidden option to BidirectionalRNN.forward

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `modRNN/model.py` — expose hidden-state trajectory (`ModularRNN`)

**Files:**
- Modify: `modRNN/model.py`
- Test: `modRNN/test_model.py`

**Interfaces:**
- Produces: `ModularRNN.forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]`. When `return_hidden=True`, second element has shape `(batch, seq_len, hidden_size)`.

- [ ] **Step 1: Write the failing tests**

Append to `modRNN/test_model.py` (it already imports `torch`, `ModularRNN`, `ModularRNNCell`, `get_device`):

```python
def test_forward_return_hidden_shape():
    model = ModularRNN(input_size=4, hidden_size=9, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 2)
    assert hidden.shape == (3, 5, 9)


def test_forward_return_hidden_shape_all_mode():
    model = ModularRNN(input_size=4, hidden_size=9, output_size=2, output_mode="all")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 5, 2)
    assert hidden.shape == (3, 5, 9)


def test_forward_without_return_hidden_returns_tensor_only():
    model = ModularRNN(input_size=4, hidden_size=9, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/hoyeon/Codes/modularRNN/modRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_model.py -k return_hidden -v`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'return_hidden'`

- [ ] **Step 3: Implement `return_hidden` in `ModularRNN.forward`**

In `modRNN/model.py`, replace the `forward` method:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        ih, hh = self.cell.masked_weights()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        states = []
        for t in range(seq_len):
            h = self.cell.step(x[:, t, :], h, ih, hh)
            states.append(h)

        masked_weight = self.output_proj.weight * self.output_mask

        if self.output_mode == "all":
            combined = torch.stack(states, dim=1)
            return F.linear(combined, masked_weight, self.output_proj.bias)

        return F.linear(states[-1], masked_weight, self.output_proj.bias)
```

with:

```python
    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        ih, hh = self.cell.masked_weights()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        states = []
        for t in range(seq_len):
            h = self.cell.step(x[:, t, :], h, ih, hh)
            states.append(h)

        outputs = torch.stack(states, dim=1)
        masked_weight = self.output_proj.weight * self.output_mask

        if self.output_mode == "all":
            result = F.linear(outputs, masked_weight, self.output_proj.bias)
        else:
            result = F.linear(states[-1], masked_weight, self.output_proj.bias)

        if return_hidden:
            return result, outputs
        return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/hoyeon/Codes/modularRNN/modRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_model.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add modRNN/model.py modRNN/test_model.py
git commit -m "$(cat <<'EOF'
Add return_hidden option to ModularRNN.forward

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `hfRNN/model.py` — copy the same change (identical class)

**Files:**
- Modify: `hfRNN/model.py`
- Test: `hfRNN/test_model.py`

**Interfaces:**
- Consumes: the edited `modRNN/model.py` from Task 5 (verbatim-identical file today; stays verbatim-identical after this task).
- Produces: `ModularRNN.forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]` in `hfRNN`, same as Task 5.

`hfRNN/model.py` is byte-identical to `modRNN/model.py` (verified before writing this plan). Copy it verbatim rather than re-deriving the change.

- [ ] **Step 1: Copy the edited model.py from modRNN**

```bash
cp /Users/hoyeon/Codes/modularRNN/modRNN/model.py /Users/hoyeon/Codes/modularRNN/hfRNN/model.py
```

- [ ] **Step 2: Add the same three tests to `hfRNN/test_model.py`**

Append to `hfRNN/test_model.py` (it already imports `torch`, `ModularRNN`, `ModularRNNCell`, `get_device` — verify this matches `modRNN/test_model.py`'s import line before appending):

```python
def test_forward_return_hidden_shape():
    model = ModularRNN(input_size=4, hidden_size=9, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 2)
    assert hidden.shape == (3, 5, 9)


def test_forward_return_hidden_shape_all_mode():
    model = ModularRNN(input_size=4, hidden_size=9, output_size=2, output_mode="all")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 5, 2)
    assert hidden.shape == (3, 5, 9)


def test_forward_without_return_hidden_returns_tensor_only():
    model = ModularRNN(input_size=4, hidden_size=9, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 2)
```

- [ ] **Step 3: Run the tests**

Run: `cd /Users/hoyeon/Codes/modularRNN/hfRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_model.py -v`
Expected: PASS (all tests)

- [ ] **Step 4: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add hfRNN/model.py hfRNN/test_model.py
git commit -m "$(cat <<'EOF'
Add return_hidden option to ModularRNN.forward in hfRNN

Verbatim copy of the modRNN/model.py change (identical class).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Update `.gitignore` for the new `results/` layout

**Files:**
- Modify: `.gitignore`

**Interfaces:**
- None (config-only change).

The current `.gitignore` has directory-specific patterns for the old flat filenames (`RNN/*_results.json`, `RNN/*_model.pt`, `biRNN/*_results.json`, `biRNN/*_model.pt`, `modRNN/*_results.json`, `modRNN/*_model.pt` — `hfRNN` currently has none of these patterns at all). Once training scripts write into `results/{env}/{timestamp}/` instead (Tasks 8-15), these patterns stop matching anything, and the new `results/` trees (containing model checkpoints, which is the exact kind of artifact this section was already ignoring) would start showing up in `git status`/`git add`. Update the patterns to match the new layout, and add `hfRNN/` to close the pre-existing gap for the directory this plan also touches.

- [ ] **Step 1: Replace the stale patterns**

In `.gitignore`, replace:

```
# Saved training results/model checkpoints
RNN/*_results.json
RNN/*_model.pt
biRNN/*_results.json
biRNN/*_model.pt
modRNN/*_results.json
modRNN/*_model.pt
```

with:

```
# Saved training results/model checkpoints/logs/figures/activity
RNN/results/
biRNN/results/
modRNN/results/
hfRNN/results/
```

- [ ] **Step 2: Verify with git status**

Run: `cd /Users/hoyeon/Codes/modularRNN && git status`
Expected: only `.gitignore` shows as modified; no `results/` directories are tracked (none exist yet at this point in the plan).

- [ ] **Step 3: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add .gitignore
git commit -m "$(cat <<'EOF'
Update .gitignore for the new results/{env}/{timestamp}/ run layout

The old *_results.json/*_model.pt patterns stop matching once training
scripts write into results/ subdirectories instead of flat cwd files.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `RNN/test_mnist.py` — wire up run artifacts

**Files:**
- Modify: `RNN/test_mnist.py`

**Interfaces:**
- Consumes: `run_artifacts.make_run_dir`, `run_artifacts.setup_logger`, `run_artifacts.save_activity_snapshot` (Task 1); `LiveTrainingPlot.save` (Task 1); `SimpleRNN.forward(..., return_hidden=True)` (Task 3).
- Produces: `train(model, train_loader, test_loader, device, epochs, run_dir, lr=1e-3, live_plot=None, module_bounds=None) -> float` (signature change: `results_path`/`model_path` replaced by `run_dir`, `module_bounds` added).

This training script is a slow, manual smoke test (downloads MNIST, trains for several epochs) — there's no fast automated test to add here per the project's existing convention (`test_mnist.py` itself is the test). Verification is running it end-to-end and inspecting the output directory.

- [ ] **Step 1: Replace the file contents**

Replace all of `RNN/test_mnist.py` with:

```python
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from live_plot import LiveTrainingPlot
from model import SimpleRNN, get_device
from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    # images: (batch, 1, 28, 28) -> (batch, 28, 28), each image read as 28 rows of 28 pixels
    return images.squeeze(1)


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} epoch(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    run_dir: Path,
    lr: float = 1e-3,
    live_plot=None,
    module_bounds: list[int] | None = None,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    history = []
    activity_dir = run_dir / "activity"
    sample_images, _ = next(iter(test_loader))
    activity_sample = to_sequence(sample_images[:1]).to(device)
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = to_sequence(images).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            accuracy = evaluate(model, test_loader, device)
            logger.info(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f}")
            history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})
            if live_plot is not None:
                live_plot.update(epoch + 1, avg_loss, accuracy)

            model.eval()
            with torch.no_grad():
                _, hidden = model(activity_sample, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"epoch_{epoch + 1:02d}", module_bounds)
    finally:
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return accuracy


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = to_sequence(images).to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = get_device()

    run_dir = make_run_dir("mnist")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 64
    logger.info(f"model: SimpleRNN(input_size=28, hidden_size={hidden_size}, output_size=10, output_mode='last')")

    train_loader, test_loader = load_data()
    model = SimpleRNN(input_size=28, hidden_size=hidden_size, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="RNN/test_mnist.py")
    accuracy = train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=5,
        run_dir=run_dir,
        live_plot=live_plot,
        module_bounds=[hidden_size],
    )
    logger.info(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

Note the `module_bounds=[hidden_size]` passed from `main()`: `SimpleRNN`'s hidden trajectory is `(batch, seq_len, hidden_size*2)` — forward states in rows `[0, hidden_size)`, backward states in `[hidden_size, hidden_size*2)` — so a single separator line at `hidden_size` marks that split on the activity heatmap.

- [ ] **Step 2: Run it end-to-end**

Run: `cd /Users/hoyeon/Codes/modularRNN/RNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python test_mnist.py`
Expected: training runs to completion (accuracy > 0.90), and afterwards:
- `RNN/results/mnist/<timestamp>/results.json` and `model.pt` exist (same content as before, new location)
- `RNN/results/mnist/<timestamp>/train.log` exists and contains the same lines that were printed to console
- `RNN/results/mnist/<timestamp>/curve.png` exists and opens as a 2-panel loss/accuracy figure
- `RNN/results/mnist/<timestamp>/activity/epoch_01.pt` through `epoch_05.pt` (and matching `.png`) exist; each `.png` shows a heatmap with one horizontal separator line at row 64

- [ ] **Step 3: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add RNN/test_mnist.py
git commit -m "$(cat <<'EOF'
Save log, curve figure, and per-epoch unit activity in RNN/test_mnist.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `biRNN/test_mnist.py` — wire up run artifacts

**Files:**
- Modify: `biRNN/test_mnist.py`

**Interfaces:**
- Same as Task 8, using `BidirectionalRNN` instead of `SimpleRNN`.

- [ ] **Step 1: Replace the file contents**

Replace all of `biRNN/test_mnist.py` with:

```python
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from live_plot import LiveTrainingPlot
from model import BidirectionalRNN, get_device
from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    return images.squeeze(1)


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} epoch(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    run_dir: Path,
    lr: float = 1e-3,
    live_plot=None,
    module_bounds: list[int] | None = None,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    history = []
    activity_dir = run_dir / "activity"
    sample_images, _ = next(iter(test_loader))
    activity_sample = to_sequence(sample_images[:1]).to(device)
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = to_sequence(images).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            accuracy = evaluate(model, test_loader, device)
            logger.info(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f}")
            history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})
            if live_plot is not None:
                live_plot.update(epoch + 1, avg_loss, accuracy)

            model.eval()
            with torch.no_grad():
                _, hidden = model(activity_sample, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"epoch_{epoch + 1:02d}", module_bounds)
    finally:
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return accuracy


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = to_sequence(images).to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = get_device()

    run_dir = make_run_dir("mnist")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 64
    logger.info(
        f"model: BidirectionalRNN(input_size=28, hidden_size={hidden_size}, output_size=10, output_mode='last')"
    )

    train_loader, test_loader = load_data()
    model = BidirectionalRNN(input_size=28, hidden_size=hidden_size, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="biRNN/test_mnist.py")
    accuracy = train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=5,
        run_dir=run_dir,
        live_plot=live_plot,
        module_bounds=[hidden_size],
    )
    logger.info(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it end-to-end**

Run: `cd /Users/hoyeon/Codes/modularRNN/biRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python test_mnist.py`
Expected: same as Task 8 Step 2, under `biRNN/results/mnist/<timestamp>/`.

- [ ] **Step 3: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add biRNN/test_mnist.py
git commit -m "$(cat <<'EOF'
Save log, curve figure, and per-epoch unit activity in biRNN/test_mnist.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `modRNN/test_mnist.py` — wire up run artifacts

**Files:**
- Modify: `modRNN/test_mnist.py`

**Interfaces:**
- Same as Task 8, using `ModularRNN`. `module_bounds` is `[hidden_size // 3, 2 * (hidden_size // 3)]` (input/intermediate/output module boundaries), not the fwd/bwd split used in `RNN`/`biRNN`.

- [ ] **Step 1: Replace the file contents**

Replace all of `modRNN/test_mnist.py` with:

```python
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from live_plot import LiveTrainingPlot
from model import ModularRNN, get_device
from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    return images.squeeze(1)


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} epoch(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    run_dir: Path,
    lr: float = 1e-3,
    live_plot=None,
    module_bounds: list[int] | None = None,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    history = []
    activity_dir = run_dir / "activity"
    sample_images, _ = next(iter(test_loader))
    activity_sample = to_sequence(sample_images[:1]).to(device)
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = to_sequence(images).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            accuracy = evaluate(model, test_loader, device)
            logger.info(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f}")
            history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})
            if live_plot is not None:
                live_plot.update(epoch + 1, avg_loss, accuracy)

            model.eval()
            with torch.no_grad():
                _, hidden = model(activity_sample, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"epoch_{epoch + 1:02d}", module_bounds)
    finally:
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return accuracy


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = to_sequence(images).to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = get_device()

    run_dir = make_run_dir("mnist")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 63
    logger.info(f"model: ModularRNN(input_size=28, hidden_size={hidden_size}, output_size=10, output_mode='last')")

    train_loader, test_loader = load_data()
    model = ModularRNN(input_size=28, hidden_size=hidden_size, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="modRNN/test_mnist.py")
    third = hidden_size // 3
    accuracy = train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=5,
        run_dir=run_dir,
        live_plot=live_plot,
        module_bounds=[third, 2 * third],
    )
    logger.info(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it end-to-end**

Run: `cd /Users/hoyeon/Codes/modularRNN/modRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python test_mnist.py`
Expected: same as Task 8 Step 2, under `modRNN/results/mnist/<timestamp>/`, with activity heatmaps showing two separator lines (at rows 21 and 42, for `hidden_size=63`).

- [ ] **Step 3: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add modRNN/test_mnist.py
git commit -m "$(cat <<'EOF'
Save log, curve figure, and per-epoch unit activity in modRNN/test_mnist.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: `hfRNN/test_mnist.py` — wire up run artifacts (HFOptimizer variant)

**Files:**
- Modify: `hfRNN/test_mnist.py`

**Interfaces:**
- Same as Task 10, but `train()` has no `lr` parameter (uses `HFOptimizer` instead of `torch.optim.Adam`) and logs `damping` in its progress line, matching the existing script.

- [ ] **Step 1: Replace the file contents**

Replace all of `hfRNN/test_mnist.py` with:

```python
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hf_optimizer import HFOptimizer
from live_plot import LiveTrainingPlot
from model import ModularRNN, get_device
from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    return images.squeeze(1)


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} epoch(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    run_dir: Path,
    live_plot=None,
    module_bounds: list[int] | None = None,
) -> float:
    optimizer = HFOptimizer(model, curvature="categorical")
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    history = []
    activity_dir = run_dir / "activity"
    sample_images, _ = next(iter(test_loader))
    activity_sample = to_sequence(sample_images[:1]).to(device)
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = to_sequence(images).to(device)
                labels = labels.to(device)

                def objective_fn(m, images=images, labels=labels):
                    z = m(images)
                    return criterion(z, labels), z

                diagnostics = optimizer.step(objective_fn)
                total_loss += diagnostics["loss_after"]
            avg_loss = total_loss / len(train_loader)
            accuracy = evaluate(model, test_loader, device)
            logger.info(
                f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f} damping {optimizer.damping:.4g}"
            )
            history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})
            if live_plot is not None:
                live_plot.update(epoch + 1, avg_loss, accuracy)

            model.eval()
            with torch.no_grad():
                _, hidden = model(activity_sample, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"epoch_{epoch + 1:02d}", module_bounds)
    finally:
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return accuracy


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = to_sequence(images).to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = get_device()

    run_dir = make_run_dir("mnist")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 63
    logger.info(f"model: ModularRNN(input_size=28, hidden_size={hidden_size}, output_size=10, output_mode='last')")

    train_loader, test_loader = load_data()
    model = ModularRNN(input_size=28, hidden_size=hidden_size, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="hfRNN/test_mnist.py")
    third = hidden_size // 3
    accuracy = train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=5,
        run_dir=run_dir,
        live_plot=live_plot,
        module_bounds=[third, 2 * third],
    )
    logger.info(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it end-to-end**

Run: `cd /Users/hoyeon/Codes/modularRNN/hfRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python test_mnist.py`
Expected: same as Task 10 Step 2, under `hfRNN/results/mnist/<timestamp>/`, with `train.log` lines including the `damping` value.

- [ ] **Step 3: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add hfRNN/test_mnist.py
git commit -m "$(cat <<'EOF'
Save log, curve figure, and per-epoch unit activity in hfRNN/test_mnist.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: `RNN/test_cartpole.py` — wire up run artifacts

**Files:**
- Modify: `RNN/test_cartpole.py`

**Interfaces:**
- Consumes: same as Task 8, plus needs `rollout_episode(model, env, device, max_steps=500, return_states=False) -> float | tuple[float, list]` (new optional return, added in this task).
- Produces: `train(model, device, num_updates, run_dir, episodes_per_update=8, live_plot=None, module_bounds=None) -> tuple[float, list]` (signature change: `results_path`/`model_path` replaced by `run_dir`, `module_bounds` added).

- [ ] **Step 1: Replace the file contents**

Replace all of `RNN/test_cartpole.py` with:

```python
import json
import logging
from pathlib import Path

import gymnasium as gym
import torch

from live_plot import LiveTrainingPlot
from model import SimpleRNN, get_device
from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)


def rollout_episode(model, env, device, max_steps: int = 500, return_states: bool = False):
    model.eval()
    state, _ = env.reset()
    states = [state]
    total_reward = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            action = logits[0, -1].argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            states.append(state)
            if terminated or truncated:
                break
    if return_states:
        return total_reward, states
    return total_reward


def evaluate_reward(model, device, num_episodes: int = 3, max_steps: int = 500) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for _ in range(num_episodes):
        total += rollout_episode(model, env, device, max_steps=max_steps)
    env.close()
    return total / num_episodes


def collect_episode_stochastic(model, env, device, max_steps: int = 500):
    model.train()
    state, _ = env.reset()
    states = [state]
    log_probs = []
    entropies = []
    rewards = []
    for _ in range(max_steps):
        x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        dist = torch.distributions.Categorical(logits=logits[0, -1])
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        states.append(state)
        if terminated or truncated:
            break
    return log_probs, entropies, rewards


def compute_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def reinforce_update(model, optimizer, episode_batch, gamma: float = 0.99, entropy_coef: float = 0.01) -> float:
    all_log_probs = []
    all_entropies = []
    all_returns = []
    for log_probs, entropies, rewards in episode_batch:
        all_log_probs.extend(log_probs)
        all_entropies.extend(entropies)
        all_returns.extend(compute_returns(rewards, gamma))

    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=all_log_probs[0].device)
    baseline = returns_tensor.mean()
    advantages = returns_tensor - baseline

    log_probs_tensor = torch.stack(all_log_probs)
    entropy_tensor = torch.stack(all_entropies)
    policy_loss = -(log_probs_tensor * advantages).mean() - entropy_coef * entropy_tensor.mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item()


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} update(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    device,
    num_updates: int,
    run_dir: Path,
    episodes_per_update: int = 8,
    live_plot=None,
    module_bounds: list[int] | None = None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = gym.make("CartPole-v1")
    activity_dir = run_dir / "activity"
    avg_reward = 0.0
    history = []
    try:
        for update in range(num_updates):
            batch = [collect_episode_stochastic(model, env, device) for _ in range(episodes_per_update)]
            loss = reinforce_update(model, optimizer, batch)
            avg_reward = evaluate_reward(model, device)
            logger.info(f"update {update + 1}/{num_updates} loss {loss:.4f} reward {avg_reward:.1f}")
            history.append({"update": update + 1, "loss": loss, "reward": avg_reward})
            if live_plot is not None:
                live_plot.update(update + 1, loss, avg_reward)

            _, states = rollout_episode(model, env, device, return_states=True)
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, hidden = model(x, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"update_{update + 1:02d}", module_bounds)

            if device.type == "mps" and (update + 1) % 10 == 0:
                torch.mps.empty_cache()
            if avg_reward >= 500:
                logger.info(f"reached max reward (500) at update {update + 1}, stopping early")
                break
    finally:
        env.close()
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return avg_reward, history


def main():
    device = get_device()

    run_dir = make_run_dir("cartpole")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 32
    logger.info(f"model: SimpleRNN(input_size=4, hidden_size={hidden_size}, output_size=2, output_mode='all')")

    model = SimpleRNN(input_size=4, hidden_size=hidden_size, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="RNN/test_cartpole.py", metrics=("loss", "reward"))
    avg_reward, _ = train(
        model,
        device,
        num_updates=5,
        run_dir=run_dir,
        live_plot=live_plot,
        module_bounds=[hidden_size],
    )
    logger.info(f"average reward: {avg_reward:.1f}")
    assert avg_reward > 150, f"expected average reward > 150, got {avg_reward:.1f}"

    try:
        render_env = gym.make("CartPole-v1", render_mode="human")
        reward = rollout_episode(model, render_env, device)
        render_env.close()
        logger.info(f"rendered episode reward: {reward:.0f}")
    except Exception as e:
        logger.info(f"render skipped (no display available): {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it end-to-end**

Run: `cd /Users/hoyeon/Codes/modularRNN/RNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python test_cartpole.py`
Expected: training runs to completion (avg reward > 150), and afterwards `RNN/results/cartpole/<timestamp>/` has `results.json`, `model.pt`, `train.log`, `curve.png`, and `activity/update_01.{pt,png}` through `update_05.{pt,png}` (or fewer, if it stopped early at reward 500).

- [ ] **Step 3: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add RNN/test_cartpole.py
git commit -m "$(cat <<'EOF'
Save log, curve figure, and per-update unit activity in RNN/test_cartpole.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: `biRNN/test_cartpole.py` — wire up run artifacts

**Files:**
- Modify: `biRNN/test_cartpole.py`

**Interfaces:** Same as Task 12, using `BidirectionalRNN`, `hidden_size=32`, `module_bounds=[32]`.

- [ ] **Step 1: Replace the file contents**

Replace all of `biRNN/test_cartpole.py` with:

```python
import json
import logging
from pathlib import Path

import gymnasium as gym
import torch

from live_plot import LiveTrainingPlot
from model import BidirectionalRNN, get_device
from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)


def rollout_episode(model, env, device, max_steps: int = 500, return_states: bool = False):
    model.eval()
    state, _ = env.reset()
    states = [state]
    total_reward = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            action = logits[0, -1].argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            states.append(state)
            if terminated or truncated:
                break
    if return_states:
        return total_reward, states
    return total_reward


def evaluate_reward(model, device, num_episodes: int = 3, max_steps: int = 500) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for _ in range(num_episodes):
        total += rollout_episode(model, env, device, max_steps=max_steps)
    env.close()
    return total / num_episodes


def collect_episode_stochastic(model, env, device, max_steps: int = 500):
    model.train()
    state, _ = env.reset()
    states = [state]
    log_probs = []
    entropies = []
    rewards = []
    for _ in range(max_steps):
        x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        dist = torch.distributions.Categorical(logits=logits[0, -1])
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        states.append(state)
        if terminated or truncated:
            break
    return log_probs, entropies, rewards


def compute_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def reinforce_update(model, optimizer, episode_batch, gamma: float = 0.99, entropy_coef: float = 0.01) -> float:
    all_log_probs = []
    all_entropies = []
    all_returns = []
    for log_probs, entropies, rewards in episode_batch:
        all_log_probs.extend(log_probs)
        all_entropies.extend(entropies)
        all_returns.extend(compute_returns(rewards, gamma))

    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=all_log_probs[0].device)
    baseline = returns_tensor.mean()
    advantages = returns_tensor - baseline

    log_probs_tensor = torch.stack(all_log_probs)
    entropy_tensor = torch.stack(all_entropies)
    policy_loss = -(log_probs_tensor * advantages).mean() - entropy_coef * entropy_tensor.mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item()


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} update(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    device,
    num_updates: int,
    run_dir: Path,
    episodes_per_update: int = 8,
    live_plot=None,
    module_bounds: list[int] | None = None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = gym.make("CartPole-v1")
    activity_dir = run_dir / "activity"
    avg_reward = 0.0
    history = []
    try:
        for update in range(num_updates):
            batch = [collect_episode_stochastic(model, env, device) for _ in range(episodes_per_update)]
            loss = reinforce_update(model, optimizer, batch)
            avg_reward = evaluate_reward(model, device)
            logger.info(f"update {update + 1}/{num_updates} loss {loss:.4f} reward {avg_reward:.1f}")
            history.append({"update": update + 1, "loss": loss, "reward": avg_reward})
            if live_plot is not None:
                live_plot.update(update + 1, loss, avg_reward)

            _, states = rollout_episode(model, env, device, return_states=True)
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, hidden = model(x, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"update_{update + 1:02d}", module_bounds)

            if device.type == "mps" and (update + 1) % 10 == 0:
                torch.mps.empty_cache()
            if avg_reward >= 500:
                logger.info(f"reached max reward (500) at update {update + 1}, stopping early")
                break
    finally:
        env.close()
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return avg_reward, history


def main():
    device = get_device()

    run_dir = make_run_dir("cartpole")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 32
    logger.info(
        f"model: BidirectionalRNN(input_size=4, hidden_size={hidden_size}, output_size=2, output_mode='all')"
    )

    model = BidirectionalRNN(input_size=4, hidden_size=hidden_size, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="biRNN/test_cartpole.py", metrics=("loss", "reward"))
    avg_reward, _ = train(
        model,
        device,
        num_updates=5,
        run_dir=run_dir,
        live_plot=live_plot,
        module_bounds=[hidden_size],
    )
    logger.info(f"average reward: {avg_reward:.1f}")
    assert avg_reward > 150, f"expected average reward > 150, got {avg_reward:.1f}"

    try:
        render_env = gym.make("CartPole-v1", render_mode="human")
        reward = rollout_episode(model, render_env, device)
        render_env.close()
        logger.info(f"rendered episode reward: {reward:.0f}")
    except Exception as e:
        logger.info(f"render skipped (no display available): {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it end-to-end**

Run: `cd /Users/hoyeon/Codes/modularRNN/biRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python test_cartpole.py`
Expected: same as Task 12 Step 2, under `biRNN/results/cartpole/<timestamp>/`.

- [ ] **Step 3: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add biRNN/test_cartpole.py
git commit -m "$(cat <<'EOF'
Save log, curve figure, and per-update unit activity in biRNN/test_cartpole.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: `modRNN/test_cartpole.py` — wire up run artifacts

**Files:**
- Modify: `modRNN/test_cartpole.py`

**Interfaces:** Same as Task 12, using `ModularRNN`, `hidden_size=300`, `module_bounds=[100, 200]`.

- [ ] **Step 1: Replace the file contents**

Replace all of `modRNN/test_cartpole.py` with:

```python
import json
import logging
from pathlib import Path

import gymnasium as gym
import torch

from live_plot import LiveTrainingPlot
from model import ModularRNN, get_device
from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)


def rollout_episode(model, env, device, max_steps: int = 500, return_states: bool = False):
    model.eval()
    state, _ = env.reset()
    states = [state]
    total_reward = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            action = logits[0, -1].argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            states.append(state)
            if terminated or truncated:
                break
    if return_states:
        return total_reward, states
    return total_reward


def evaluate_reward(model, device, num_episodes: int = 3, max_steps: int = 500) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for _ in range(num_episodes):
        total += rollout_episode(model, env, device, max_steps=max_steps)
    env.close()
    return total / num_episodes


def collect_episode_stochastic(model, env, device, max_steps: int = 500):
    model.train()
    state, _ = env.reset()
    states = [state]
    log_probs = []
    entropies = []
    rewards = []
    for _ in range(max_steps):
        x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        dist = torch.distributions.Categorical(logits=logits[0, -1])
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        states.append(state)
        if terminated or truncated:
            break
    return log_probs, entropies, rewards


def compute_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def reinforce_update(model, optimizer, episode_batch, gamma: float = 0.99, entropy_coef: float = 0.01) -> float:
    all_log_probs = []
    all_entropies = []
    all_returns = []
    for log_probs, entropies, rewards in episode_batch:
        all_log_probs.extend(log_probs)
        all_entropies.extend(entropies)
        all_returns.extend(compute_returns(rewards, gamma))

    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=all_log_probs[0].device)
    baseline = returns_tensor.mean()
    advantages = returns_tensor - baseline

    log_probs_tensor = torch.stack(all_log_probs)
    entropy_tensor = torch.stack(all_entropies)
    policy_loss = -(log_probs_tensor * advantages).mean() - entropy_coef * entropy_tensor.mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item()


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} update(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    device,
    num_updates: int,
    run_dir: Path,
    episodes_per_update: int = 8,
    live_plot=None,
    module_bounds: list[int] | None = None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = gym.make("CartPole-v1")
    activity_dir = run_dir / "activity"
    avg_reward = 0.0
    history = []
    try:
        for update in range(num_updates):
            batch = [collect_episode_stochastic(model, env, device) for _ in range(episodes_per_update)]
            loss = reinforce_update(model, optimizer, batch)
            avg_reward = evaluate_reward(model, device)
            logger.info(f"update {update + 1}/{num_updates} loss {loss:.4f} reward {avg_reward:.1f}")
            history.append({"update": update + 1, "loss": loss, "reward": avg_reward})
            if live_plot is not None:
                live_plot.update(update + 1, loss, avg_reward)

            _, states = rollout_episode(model, env, device, return_states=True)
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, hidden = model(x, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"update_{update + 1:02d}", module_bounds)

            if device.type == "mps" and (update + 1) % 10 == 0:
                torch.mps.empty_cache()
            if avg_reward >= 500:
                logger.info(f"reached max reward (500) at update {update + 1}, stopping early")
                break
    finally:
        env.close()
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return avg_reward, history


def main():
    device = get_device()

    run_dir = make_run_dir("cartpole")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 300
    logger.info(f"model: ModularRNN(input_size=4, hidden_size={hidden_size}, output_size=2, output_mode='all')")

    model = ModularRNN(input_size=4, hidden_size=hidden_size, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="modRNN/test_cartpole.py", metrics=("loss", "reward"))
    third = hidden_size // 3
    avg_reward, _ = train(
        model,
        device,
        num_updates=5,
        run_dir=run_dir,
        episodes_per_update=4,
        live_plot=live_plot,
        module_bounds=[third, 2 * third],
    )
    logger.info(f"average reward: {avg_reward:.1f}")
    assert avg_reward > 150, f"expected average reward > 150, got {avg_reward:.1f}"

    try:
        render_env = gym.make("CartPole-v1", render_mode="human")
        reward = rollout_episode(model, render_env, device)
        render_env.close()
        logger.info(f"rendered episode reward: {reward:.0f}")
    except Exception as e:
        logger.info(f"render skipped (no display available): {e}")


if __name__ == "__main__":
    main()
```

Note: `episodes_per_update=4` is passed in `main()` here (matching the pre-existing `modRNN/test_cartpole.py`), unlike `RNN`/`biRNN` which use the `train()` default of 8.

- [ ] **Step 2: Run it end-to-end**

Run: `cd /Users/hoyeon/Codes/modularRNN/modRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python test_cartpole.py`
Expected: same as Task 12 Step 2, under `modRNN/results/cartpole/<timestamp>/`, with activity heatmaps showing separator lines at rows 100 and 200.

- [ ] **Step 3: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add modRNN/test_cartpole.py
git commit -m "$(cat <<'EOF'
Save log, curve figure, and per-update unit activity in modRNN/test_cartpole.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: `hfRNN/test_cartpole.py` — wire up run artifacts (HFOptimizer variant)

**Files:**
- Modify: `hfRNN/test_cartpole.py`

**Interfaces:**
- Consumes: `rollout_episode(model, env, device, max_steps=500, return_states=False)` (extended in this task, same as Task 12); `model.forward(..., return_hidden=True)` (Task 6).
- Produces: `train(model, device, num_updates, run_dir, episodes_per_update=8, live_plot=None, module_bounds=None) -> tuple[float, list]`.

This file's `train()` doesn't use `reinforce_update`/`torch.optim.Adam` — it batches episodes via `collect_episode_stochastic` + `build_reinforce_objective` and calls `HFOptimizer.step()` once per update. That structure is unchanged here; only run-artifact wiring is added.

- [ ] **Step 1: Replace the file contents**

Replace all of `hfRNN/test_cartpole.py` with:

```python
# hfRNN/test_cartpole.py
import json
import logging
from pathlib import Path

import gymnasium as gym
import torch

from hf_optimizer import HFOptimizer
from live_plot import LiveTrainingPlot
from model import ModularRNN, get_device
from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)


def rollout_episode(model, env, device, max_steps: int = 500, return_states: bool = False):
    model.eval()
    state, _ = env.reset()
    states = [state]
    total_reward = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            action = logits[0, -1].argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            states.append(state)
            if terminated or truncated:
                break
    if return_states:
        return total_reward, states
    return total_reward


def evaluate_reward(model, device, num_episodes: int = 3, max_steps: int = 500) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for _ in range(num_episodes):
        total += rollout_episode(model, env, device, max_steps=max_steps)
    env.close()
    return total / num_episodes


def compute_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def collect_episode_stochastic(model, env, device, max_steps: int = 500):
    model.eval()
    state, _ = env.reset()
    states = [state]
    actions = []
    rewards = []
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            dist = torch.distributions.Categorical(logits=logits[0, -1])
            action = dist.sample()
            state, reward, terminated, truncated, _ = env.step(action.item())
            actions.append(action.item())
            rewards.append(reward)
            states.append(state)
            if terminated or truncated:
                break
    return states[:-1], actions, rewards


def build_reinforce_objective(states, actions, returns, device, entropy_coef: float = 0.01):
    """`states`/`actions`/`returns` are flat lists pooled across an episode batch (one
    entry per visited timestep). Returns an objective_fn(model) -> (loss, z) closure for
    HFOptimizer.step() — recomputes the forward pass under the model's *current* params
    each call, so it's valid across every CG iteration within one HF step."""
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1, obs_dim)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    baseline = returns_tensor.mean()
    advantages = returns_tensor - baseline

    def objective_fn(model):
        z = model(states_tensor)[:, -1, :]  # (N, num_actions), output_mode="all" sliced to last step of each 1-step sequence
        dist = torch.distributions.Categorical(logits=z)
        log_probs = dist.log_prob(actions_tensor)
        loss = -(log_probs * advantages).mean() - entropy_coef * dist.entropy().mean()
        return loss, z

    return objective_fn


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} update(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    device,
    num_updates: int,
    run_dir: Path,
    episodes_per_update: int = 8,
    live_plot=None,
    module_bounds: list[int] | None = None,
):
    optimizer = HFOptimizer(model, curvature="categorical")
    env = gym.make("CartPole-v1")
    activity_dir = run_dir / "activity"
    avg_reward = 0.0
    history = []
    try:
        for update in range(num_updates):
            all_states, all_actions, all_returns = [], [], []
            for _ in range(episodes_per_update):
                states, actions, rewards = collect_episode_stochastic(model, env, device)
                all_states.extend(states)
                all_actions.extend(actions)
                all_returns.extend(compute_returns(rewards))

            objective_fn = build_reinforce_objective(all_states, all_actions, all_returns, device)
            diagnostics = optimizer.step(objective_fn)
            avg_reward = evaluate_reward(model, device)
            logger.info(
                f"update {update + 1}/{num_updates} loss {diagnostics['loss_after']:.4f} "
                f"reward {avg_reward:.1f} damping {optimizer.damping:.4g}"
            )
            history.append({"update": update + 1, "loss": diagnostics["loss_after"], "reward": avg_reward})
            if live_plot is not None:
                live_plot.update(update + 1, diagnostics["loss_after"], avg_reward)

            _, states = rollout_episode(model, env, device, return_states=True)
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, hidden = model(x, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"update_{update + 1:02d}", module_bounds)

            if device.type == "mps" and (update + 1) % 10 == 0:
                torch.mps.empty_cache()
            if avg_reward >= 500:
                logger.info(f"reached max reward (500) at update {update + 1}, stopping early")
                break
    finally:
        env.close()
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return avg_reward, history


def main():
    device = get_device()

    run_dir = make_run_dir("cartpole")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 300
    logger.info(f"model: ModularRNN(input_size=4, hidden_size={hidden_size}, output_size=2, output_mode='all')")

    model = ModularRNN(input_size=4, hidden_size=hidden_size, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="hfRNN/test_cartpole.py", metrics=("loss", "reward"))
    third = hidden_size // 3
    avg_reward, _ = train(
        model,
        device,
        num_updates=5,
        run_dir=run_dir,
        episodes_per_update=4,
        live_plot=live_plot,
        module_bounds=[third, 2 * third],
    )
    logger.info(f"average reward: {avg_reward:.1f}")
    assert avg_reward > 150, f"expected average reward > 150, got {avg_reward:.1f}"

    try:
        render_env = gym.make("CartPole-v1", render_mode="human")
        reward = rollout_episode(model, render_env, device)
        render_env.close()
        logger.info(f"rendered episode reward: {reward:.0f}")
    except Exception as e:
        logger.info(f"render skipped (no display available): {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it end-to-end**

Run: `cd /Users/hoyeon/Codes/modularRNN/hfRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python test_cartpole.py`
Expected: same as Task 14 Step 2, under `hfRNN/results/cartpole/<timestamp>/`.

- [ ] **Step 3: Run the full test suite for this directory as a final sanity check**

Run: `cd /Users/hoyeon/Codes/modularRNN/hfRNN && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest test_model.py test_hf_optimizer.py test_run_artifacts.py -v`
Expected: PASS (all tests — `test_hf_optimizer.py` is untouched by this plan and should still pass unmodified)

- [ ] **Step 4: Commit**

```bash
cd /Users/hoyeon/Codes/modularRNN
git add hfRNN/test_cartpole.py
git commit -m "$(cat <<'EOF'
Save log, curve figure, and per-update unit activity in hfRNN/test_cartpole.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (after all tasks)

Run the full fast test suite across all four directories:

```bash
cd /Users/hoyeon/Codes/modularRNN
for d in RNN biRNN modRNN hfRNN; do
  echo "=== $d ==="
  (cd "$d" && /Users/hoyeon/Codes/modularRNN/.venv/bin/python -m pytest -q)
done
```

Expected: all pass, in all four directories. (The `test_mnist.py`/`test_cartpole.py` end-to-end runs from Tasks 8-15 are the slow manual smoke tests per the project's existing convention and don't need to be re-run here unless something changed since.)
