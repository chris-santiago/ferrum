"""Phase 9 Linkage transform fixture generator.

Computes scipy reference linkage matrices for 5 curated (method, metric)
pairs + chebyshev metric + median/centroid edge cases.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd


# Deterministic 10x4 numeric input matrix.
np.random.seed(0)
DATA = np.random.normal(size=(10, 4)).round(4)

CASES = [
    ("ward_euclidean",       "ward",     "euclidean"),
    ("complete_euclidean",   "complete", "euclidean"),
    ("average_correlation",  "average",  "correlation"),
    ("single_manhattan",     "single",   "cityblock"),
    ("complete_cosine",      "complete", "cosine"),
    ("complete_chebyshev",   "complete", "chebyshev"),
    ("median_euclidean",     "median",   "euclidean"),
    ("centroid_euclidean",   "centroid", "euclidean"),
]


def case_payload(name, method, metric):
    if method == "ward":
        Z = sch.linkage(DATA, method="ward")
    else:
        condensed = ssd.pdist(DATA, metric=metric)
        Z = sch.linkage(condensed, method=method)
    leaves = sch.leaves_list(Z).tolist()
    return {
        "name": name,
        "method": method,
        "metric": metric,
        "n": int(DATA.shape[0]),
        "linkage": Z.tolist(),    # n-1 rows x 4 cols
        "order": leaves,
    }


def main():
    payload = {
        "_pinned_versions": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "data": DATA.tolist(),
        "cases": [case_payload(*c) for c in CASES],
    }
    out = Path(__file__).resolve().parent / "linkage_refs.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
