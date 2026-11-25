"""Download the Hotel Booking Demand dataset into data/hotels.csv and sanity-check it."""

import shutil
import ssl
from pathlib import Path
from urllib.request import urlopen

import certifi
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-02-11/hotels.csv"
)
EXPECTED_SHAPE = (119_390, 32)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = DATA_DIR / "hotels.csv"


def download(url: str, target: Path) -> None:
    # certifi: python.org builds on macOS have no root certificates
    context = ssl.create_default_context(cafile=certifi.where())
    partial = target.with_suffix(".csv.part")
    try:
        with urlopen(url, context=context, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not TARGET.exists():
        print(f"Downloading {URL}")
        download(URL, TARGET)

    df = pd.read_csv(TARGET, na_values=["NULL"])
    if df.shape != EXPECTED_SHAPE:
        raise SystemExit(
            f"Unexpected shape {df.shape}, expected {EXPECTED_SHAPE}. Delete {TARGET} and retry."
        )

    rate = df["is_canceled"].mean()
    print(f"OK: {TARGET} — {df.shape[0]:,} rows × {df.shape[1]} columns, {rate:.1%} cancelled")


if __name__ == "__main__":
    main()
