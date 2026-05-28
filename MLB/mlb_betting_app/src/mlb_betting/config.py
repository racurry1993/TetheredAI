from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    model_dir: Path
    odds_db_path: Path
    odds_api_key: Optional[str]
    odds_sport_key: str
    odds_regions: str
    odds_markets: str
    odds_format: str


def find_project_root() -> Path:
    """
    Find the project root from either:
    - current working directory
    - this config.py file location
    """
    candidates = []

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    file_path = Path(__file__).resolve()
    candidates.extend([file_path, *file_path.parents])

    for path in candidates:
        if (path / "requirements.txt").exists() and (path / "src").exists():
            return path

    return Path.cwd().resolve()


def load_local_env(env_path: Path) -> None:
    """
    Minimal .env loader using only the Python standard library.

    Supports:
        KEY=value

    Ignores:
        blank lines
        comments beginning with #
    """
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def get_settings() -> Settings:
    root = find_project_root()

    env_path = root / ".env"
    load_local_env(env_path)

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    if not model_dir.is_absolute():
        model_dir = root / model_dir

    odds_db_path = Path(os.getenv("ODDS_DB_PATH", "data/odds.db"))
    if not odds_db_path.is_absolute():
        odds_db_path = root / odds_db_path

    return Settings(
        project_root=root,
        data_dir=data_dir,
        model_dir=model_dir,
        odds_db_path=odds_db_path,
        odds_api_key=os.getenv("ODDS_API_KEY"),
        odds_sport_key=os.getenv("ODDS_SPORT_KEY", "baseball_mlb"),
        odds_regions=os.getenv("ODDS_REGIONS", "us"),
        odds_markets=os.getenv("ODDS_MARKETS", "h2h,spreads,totals"),
        odds_format=os.getenv("ODDS_FORMAT", "american"),
    )