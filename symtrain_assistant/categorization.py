from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from symtrain_assistant.models import embed_texts  # no call_gpt_json here
from .config import N_CLUSTERS


def cluster_categories(df: pd.DataFrame, text_col: str = "reason_gpt") -> pd.DataFrame:
    """
    Transformer-based categorization with KMeans on reasons.
    """
    df = df.copy()
    texts = df[text_col].fillna("").tolist()
    embeddings = embed_texts(texts)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    df["cluster_id"] = clusters

    # Placeholder mapping; after you inspect, rename to meaningful labels.
    cluster_to_label = {i: f"Category_{i}" for i in range(N_CLUSTERS)}
    df["category_txf"] = df["cluster_id"].map(cluster_to_label)

    return df, kmeans, cluster_to_label


def add_gpt_categories(
    df: pd.DataFrame,
    allowed_categories: List[str] | None = None,
) -> pd.DataFrame:
    """
    Offline we DON'T call GPT at all.
    Just copy the transformer-based category into category_gpt so
    downstream code works.
    """
    df = df.copy()
    df["category_gpt"] = df["category_txf"]
    return df


def build_category_classifier(df: pd.DataFrame, text_col: str = "reason_gpt") -> KNeighborsClassifier:
    """
    Simple KNN classifier on embeddings to quickly predict category from new text.
    """
    mask = df["category_gpt"].notna()
    texts = df.loc[mask, text_col].tolist()
    labels = df.loc[mask, "category_gpt"].tolist()

    X = embed_texts(texts)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, labels)
    return knn


def predict_category_knn(text: str, knn: KNeighborsClassifier) -> str:
    X = embed_texts([text])
    return knn.predict(X)[0]
