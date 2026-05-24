from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _path_from_env(env_name: str, default_relative_path: str) -> Path:
    raw_path = os.getenv(env_name)
    path = Path(raw_path) if raw_path else ROOT_DIR / default_relative_path
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = ROOT_DIR
    fraud_data: Path = _path_from_env("FRAUD_DATA_PATH", "detection_fraude.csv")
    customer_data: Path = _path_from_env("CUSTOMER_DATA_PATH", "data_cluster.csv")
    model_dir: Path = _path_from_env("MODEL_DIR", "models")
    report_dir: Path = _path_from_env("REPORT_DIR", "reports")
    processed_dir: Path = ROOT_DIR / "data" / "processed"

    def ensure_outputs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


PATHS = ProjectPaths()
RANDOM_STATE = 42
