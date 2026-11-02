"""Download the UCI Dry Bean dataset and save a local CSV copy."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "dry_bean_full.csv"


def download_with_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    dry_bean = fetch_ucirepo(id=602)
    X = dry_bean.data.features
    y = dry_bean.data.targets
    df = pd.concat([X, y], axis=1)
    if "Class" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Class"})
    return df


def download_with_csv_mirrors() -> pd.DataFrame:
    mirrors = [
        "https://www.stat.rice.edu/~scottdw/stat541/PROJECTS/uci-data/DRY-BEAN/Dry_Bean_Dataset.csv",
        "https://www.stat.cmu.edu/~larry/All_of_Regression/LarryListProg/Code/09_CLASSIFICATION/BEANS/dry-beans.csv",
    ]
    last_error = None
    for url in mirrors:
        try:
            df = pd.read_csv(url)
            if "Class" not in df.columns:
                df = df.rename(columns={df.columns[-1]: "Class"})
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Unable to download dataset from mirrors. Last error: {last_error}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = download_with_ucimlrepo()
        source = "ucimlrepo (UCI id=602)"
    except Exception as exc:  # noqa: BLE001
        print(f"ucimlrepo download failed ({exc}); trying CSV mirrors...")
        df = download_with_csv_mirrors()
        source = "public CSV mirror"

    df["Class"] = df["Class"].astype(str).str.strip()
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")
    print(f"Source: {source}")
    print(f"Shape: {df.shape}")
    print(f"Features: {[c for c in df.columns if c != 'Class']}")
    print("Class distribution:")
    print(df["Class"].value_counts().to_string())


if __name__ == "__main__":
    main()
