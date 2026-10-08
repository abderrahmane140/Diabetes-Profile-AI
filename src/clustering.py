import numpy as np
import pandas as pd


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def create_kmeans(k, random_state=42):
    """create an unfittd K-Means model."""


    return KMeans(
        n_clusters=k,  # Number of clusters (groups) to create.
        init="k-means++", # Choose initial centers that tend to be spread apart.
        n_init=10, # Try 10 initializations; keep the result with lowest inertia.
        random_state=random_state, # Seed to reproduce the same random choices.
    )


def compare_cluster_counts(
    X_train_scaled,
    K_values=range(2,7)
): 
    """Compare candidate cluster counts using training data"""

    results = []

    for k in K_values:
        model = create_kmeans(k)

        labels = model.fit_predict(X_train_scaled)

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        if len(unique_labels) != k:
            raise ValueError(
                f"Requested {k} cluster but obtained "
                f"{len(unique_labels)}"
            )

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(
                X_train_scaled,
                labels
            ),
            "smallest_cluster": int(counts.min()),
            "largest_cluster" : int(counts.max())
        })

    return pd.DataFrame(results)



def summarize_clusters(features_original_units, labels):
    """Calculate cluster means and sizes in original units."""

    data = features_original_units.copy()
    data["Cluster"] = labels

    means = data.groupby("Cluster").mean()

    means.insert(
        0,
        "patient_count",
        data.groupby("Cluster").size()
    )

    return means