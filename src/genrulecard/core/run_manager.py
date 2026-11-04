from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import RunPaths


def create_run_paths(output_root: Path | str, dataset_name: str, seed: int) -> RunPaths:
    timestamp = datetime.now().strftime("%Y%m%d__%H_%M_%S")
    root = Path(output_root) / f"{dataset_name}_{timestamp}_seed{seed}"
    paths = RunPaths(
        root=root,
        rulecard=root / "01_rulecard_fit",
        vae=root / "02_vae",
        genetic=root / "03_genetic",
        figures=root / "04_figures",
        results=root / "05_results",
    )
    for path in (paths.root, paths.rulecard, paths.vae, paths.genetic, paths.figures, paths.results):
        path.mkdir(parents=True, exist_ok=False if path == paths.root else True)
    return paths


def save_json(data: dict[str, Any], destination: Path) -> None:
    destination.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
