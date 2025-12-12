from pathlib import Path
import pandas as pd

from .config import DATA_PROCESSED
from .data_loading import load_all_simulations, simulations_to_dataframe
from .extraction import run_extraction
from .categorization import cluster_categories, add_gpt_categories, build_category_classifier


def build_full_dataset() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    sims = load_all_simulations()
    df = simulations_to_dataframe(sims)

    df = run_extraction(df)

    df, kmeans, cluster_to_label = cluster_categories(df, text_col="reason_gpt")

    allowed_categories = sorted(df["category_txf"].unique().tolist())
    df = add_gpt_categories(df, allowed_categories)

    df.to_parquet(DATA_PROCESSED / "simulations_labeled.parquet", index=False)


def train_category_classifier() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PROCESSED / "simulations_labeled.parquet")
    knn = build_category_classifier(df, text_col="reason_gpt")

    import joblib

    joblib.dump(knn, DATA_PROCESSED / "category_knn.joblib")


if __name__ == "__main__":
    build_full_dataset()
    train_category_classifier()
