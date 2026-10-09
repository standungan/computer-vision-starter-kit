import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvkit.cli import list_tasks, task_config_path  # noqa: E402
from cvkit.config import load_config  # noqa: E402


@pytest.fixture
def smoke_cfg(tmp_path):
    """A factory: quick config for a task, pointed at tmp dirs and shrunk to run in seconds."""
    def make(task: str) -> dict:
        cfg = load_config(task_config_path(task, quick=True))
        cfg["data"]["root"] = str(tmp_path / "data")
        cfg["data"]["num_workers"] = 0
        cfg["output"]["dir"] = str(tmp_path / "out")
        cfg["train"]["epochs"] = 1
        cfg["train"]["limit_batches"] = 2
        cfg["model"]["pretrained"] = False
        cfg["demo"] = {**cfg.get("demo", {}), "n": 24, "size": cfg.get("demo", {}).get("size", 96)}
        return cfg
    return make


ALL_TASKS = list_tasks()
