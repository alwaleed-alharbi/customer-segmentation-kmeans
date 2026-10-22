"""
My own simple silhouette score (for learning).

The elbow method only looks at SSE, and SSE always goes down when K goes up.
The silhouette score asks a different question for every point:
"Am I closer to my own cluster than to the nearest other cluster?"

For one point i:
  a(i) = average distance from i to the other points in its OWN cluster
  b(i) = average distance from i to the points of the NEAREST other cluster
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

s(i) is between -1 and 1:
  close to  1 -> the point sits well inside its cluster
  close to  0 -> the point is on the border between two clusters
  below     0 -> the point is probably in the wrong cluster

The silhouette score of the whole clustering is the average of all s(i).
"""
import numpy as np


def silhouette_samples_scratch(X, labels):
    """Return s(i) for every point."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    clusters = np.unique(labels)

    # distance between every pair of points (n x n)
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        n_own = own.sum()
        if n_own == 1:          # a cluster with one point gets s = 0
            continue

        # a(i): mean distance to my own cluster (excluding myself)
        a = dist[i, own].sum() / (n_own - 1)

        # b(i): smallest mean distance to any other cluster
        b = min(dist[i, labels == c].mean() for c in clusters if c != labels[i])

        s[i] = (b - a) / max(a, b)
    return s


def silhouette_score_scratch(X, labels):
    """Average silhouette over all points."""
    return float(silhouette_samples_scratch(X, labels).mean())


if __name__ == "__main__":
    # two clear groups -> score should be close to 1
    points = np.array([[1.0, 1.0], [1.5, 2.0], [1.2, 1.4],
                       [8.0, 8.0], [8.5, 9.0], [9.0, 8.2]])
    good = np.array([0, 0, 0, 1, 1, 1])
    bad = np.array([0, 1, 0, 1, 0, 1])
    print("good grouping:", round(silhouette_score_scratch(points, good), 3))
    print("bad grouping: ", round(silhouette_score_scratch(points, bad), 3))
