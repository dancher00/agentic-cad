from __future__ import annotations

import importlib.util
from pathlib import Path


def test_photo_benchmark_supplies_only_images_and_text() -> None:
    spec = importlib.util.spec_from_file_location(
        "photo_benchmark_runner", Path("scripts/run_photo_cad_benchmark.py")
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    args = runner.command(Path("captures/book/frames"), Path("work/book"), "book", "mvs", True)
    assert "photo-cad" in args
    assert args[args.index("--object") + 1] == "book"
    assert "--offline" in args
    assert not {"--cameras", "--masks", "--reference"}.intersection(args)
    assert runner.CASES == ("book", "bottle", "camera", "cup", "laptop")
