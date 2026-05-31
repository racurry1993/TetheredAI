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


def find_project_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "src").exists() and ((path / "requirements.txt").exists() or (path / "pyproject.toml").exists()):
            return path
    return start


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve(root: Path, value: str, default: Path) -> Path:
    path = Path(os.getenv(value, str(default)))
    return path if path.is_absolute() else root / path


def get_settings() -> Settings:
    root = find_project_root()
    _load_env_file(root / ".env")
    data_dir = _resolve(root, "DATA_DIR", root / "data")
    model_dir = _resolve(root, "MODEL_DIR", root / "models")
    odds_db_path = _resolve(root, "ODDS_DB_PATH", data_dir / "odds.db")
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
