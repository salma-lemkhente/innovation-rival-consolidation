import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load() -> dict:
    path = REPO / "config.toml"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: copy config.example.toml to config.toml and edit [roots]")
    with open(path, "rb") as f:
        return tomllib.load(f)


CFG = _load()
YEAR_MIN = CFG["sample"]["year_min"]
YEAR_MAX = CFG["sample"]["year_max"]


def _root(name: str) -> Path:
    root = Path(CFG["roots"][name])
    return root if root.is_absolute() else REPO / root


def raw(name: str) -> Path:
    if name not in CFG["raw"]:
        raise KeyError(f"'{name}' is not declared under [raw] in config.toml")
    path = _root("raw") / CFG["raw"][name]
    if not path.exists():
        raise FileNotFoundError(f"raw input '{name}' not found at {path}")
    return path


def out(stage: str, filename: str) -> Path:
    folder = _root("data") / stage
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def here(script_file: str, filename: str) -> Path:
    return Path(script_file).resolve().parent / filename
