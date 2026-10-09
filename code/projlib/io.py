from pathlib import Path

import pandas as pd


def write(df: pd.DataFrame, path: Path, key: list[str] | None = None) -> None:
    if key is not None:
        dup = int(df.duplicated(key).sum())
        if dup:
            raise ValueError(f"{path.name}: {dup:,} duplicate rows on {key}")
    df.to_parquet(path, index=False)
    print(f"  wrote {path.name}: {len(df):,} rows, {df.shape[1]} columns")
