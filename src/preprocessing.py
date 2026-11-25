"""Shared data loading, cleaning, train/test split and the preprocessing pipeline used by every notebook."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
# may be filled in after the booking date, checked with a run without them
SUSPICIOUS_COLS = [
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "country",
]

CATEGORICAL_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]
# values with less than 1% of the training rows go into one "infrequent" column
RARE_MIN_FREQ = 0.01


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


def get_X_y(df: pd.DataFrame, drop_suspicious: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    drop = [TARGET] + LEAKAGE_COLS + UNUSED_COLS
    if drop_suspicious:
        drop += SUSPICIOUS_COLS
    X = df.drop(columns=drop)
    # agent is an anonymised id stored as float, turn it into a label
    X["agent"] = df["agent"].map(lambda a: np.nan if pd.isna(a) else str(int(a))).astype(object)
    if "country" in X:
        X["country"] = X["country"].astype(object)
    return X, df[TARGET]


def _rare_onehot(fill_value: str) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=fill_value)),
        ("onehot", OneHotEncoder(
            min_frequency=RARE_MIN_FREQ,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )),
    ])


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Column lists are taken from X, so the same code works with and without SUSPICIOUS_COLS."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in X]
    num_cols = [c for c in X.columns if c not in cat_cols + ["country", "agent"]]

    transformers = [
        ("agent", _rare_onehot("none"), ["agent"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
    ]
    if "country" in X:
        transformers.insert(0, ("country", _rare_onehot("unknown"), ["country"]))
    return ColumnTransformer(transformers)


def make_pipeline(model, X: pd.DataFrame) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(X)), ("model", model)])
