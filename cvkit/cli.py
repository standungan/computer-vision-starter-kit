"""One command line for every task.

    python -m cvkit <task> <command> [--config FILE | --quick] [--set key=value ...]

    commands
      quickstart   demo data (if missing) -> train -> evaluate -> sample predictions
      demo-data    generate the synthetic demo dataset at data.root
      train        train and save outputs/.../checkpoints/best.pt
      evaluate     metrics + plots on the held-out test split
      predict      run the trained model on --input (file or folder)
      app          launch the Gradio demo

Run `python -m cvkit` with no arguments to list the tasks.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .config import load_config
from .synthetic import make_demo_data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:             # so `tasks.<name>` imports work however cvkit was launched
    sys.path.insert(0, str(REPO_ROOT))
COMMANDS = ["quickstart", "demo-data", "train", "evaluate", "predict", "app"]


def list_tasks() -> list[str]:
    tasks_dir = REPO_ROOT / "tasks"
    return sorted(p.name for p in tasks_dir.iterdir()
                  if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "configs").exists())


def task_config_path(task: str, quick: bool) -> Path:
    return REPO_ROOT / "tasks" / task / "configs" / ("quick.yaml" if quick else "default.yaml")


def task_module(task: str, name: str):
    return importlib.import_module(f"tasks.{task}.{name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cvkit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", help="one of: " + ", ".join(list_tasks()))
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--config", help="YAML config (default: the task's configs/default.yaml)")
    p.add_argument("--quick", action="store_true",
                   help="use the task's configs/quick.yaml (CPU-friendly, ~minutes)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set train.epochs=3 data.root=data/mine")
    p.add_argument("--input", help="predict: an image file or a folder of images")
    p.add_argument("--share", action="store_true", help="app: create a temporary public link")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv in (["-h"], ["--help"]):
        print(__doc__)
        print("tasks:", ", ".join(list_tasks()))
        return 0
    args = build_parser().parse_args(argv)
    if args.task not in list_tasks():
        print(f"unknown task '{args.task}'. Available: {', '.join(list_tasks())}", file=sys.stderr)
        return 2

    config_path = Path(args.config) if args.config else task_config_path(args.task, args.quick)
    cfg = load_config(config_path, args.set)
    print(f"[cvkit] task={args.task} command={args.command} config={config_path}")

    cmd = args.command
    if cmd == "demo-data":
        root = make_demo_data(args.task, cfg)
        print(f"demo data written to {root}")
    elif cmd == "train":
        task_module(args.task, "train").train(cfg)
    elif cmd == "evaluate":
        task_module(args.task, "evaluate").evaluate(cfg)
    elif cmd == "predict":
        task_module(args.task, "predict").run(cfg, args.input)
    elif cmd == "app":
        task_module(args.task, "app").build_app(cfg).launch(share=args.share)
    elif cmd == "quickstart":
        quickstart(args.task, cfg)
    return 0


def quickstart(task: str, cfg: dict) -> None:
    """The level-0 experience: everything, end to end, one command."""
    root = Path(cfg["data"]["root"])
    if not root.exists():
        print(f"\n== 1/4 creating demo data at {root}")
        make_demo_data(task, cfg)
    else:
        print(f"\n== 1/4 using existing data at {root}")
    print("\n== 2/4 training")
    task_module(task, "train").train(cfg)
    print("\n== 3/4 evaluating on the test split")
    task_module(task, "evaluate").evaluate(cfg)
    print("\n== 4/4 predicting on a few test images")
    task_module(task, "predict").run(cfg, None)
    print(f"\nDone. Look in {cfg['output']['dir']}/reports/ for the plots, "
          f"and run `python -m cvkit {task} app` for the demo app.")


if __name__ == "__main__":
    raise SystemExit(main())
