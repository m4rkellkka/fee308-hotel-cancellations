"""Shared data loading, cleaning and the fixed train/test split used by every notebook."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "hotels.csv"

TARGET = "is_canceled"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# recorded after the outcome is known
LEAKAGE_COLS = ["reservation_status", "reservation_status_date", "assigned_room_type"]
# company: 94% missing anonymised id; year/week/day: do not carry over to new bookings
UNUSED_COLS = [
    "company",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NULL"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove data errors: bookings with no guests and the one negative daily rate."""
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    return df[(guests > 0) & (df["adr"] >= 0)].reset_index(drop=True)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_train_test(clean(load_raw()))
