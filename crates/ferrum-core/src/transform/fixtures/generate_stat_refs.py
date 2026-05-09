"""
Phase 5 stat-engine fixture generator.

Computes reference values for stat_kde and stat_smooth (LM, LOESS) using
the SAME formulas the Rust impl uses, so cargo tests can assert bit-close
matches without scipy in the Rust dev environment.

Usage (from repo root):
    uv pip install -r crates/ferrum-core/src/transform/fixtures/requirements-fixtures.txt
    uv run python crates/ferrum-core/src/transform/fixtures/generate_stat_refs.py

Pinned versions: see requirements-fixtures.txt. Re-run and commit the JSON
when the spec or the bandwidth/LOESS formulas change.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats  # sanity check only — see kde_sanity_check below


# ---------- KDE ----------

def kde_bandwidth(x, method):
    sigma = np.std(x, ddof=1)
    n = len(x)
    if method == "scott":
        return sigma * n ** (-1.0 / 5.0)
    if method == "silverman":
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        return 0.9 * min(sigma, iqr / 1.34) * n ** (-1.0 / 5.0)
    raise ValueError(method)


def kde_compute_grid(x, bw, grid, cumulative=False):
    # Gaussian kernel.
    n = len(x)
    diff = (grid[:, None] - x[None, :]) / bw
    density = np.exp(-0.5 * diff ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    if cumulative:
        # Trapezoidal cumulative integral on the grid.
        steps = np.diff(grid)
        seg_avg = 0.5 * (density[1:] + density[:-1])
        return np.concatenate([[0.0], np.cumsum(seg_avg * steps)])
    return density


def kde_sanity_check(x, bw, grid):
    # Cross-check that our hand-rolled gaussian sum matches scipy's gaussian_kde
    # when we feed it the same bandwidth (set via covariance_factor override).
    kde = scipy.stats.gaussian_kde(x, bw_method=bw / np.std(x, ddof=1))
    return kde(grid)


def kde_cases():
    cases = []

    # Case 1: small normal sample, scott bandwidth, no cumulative
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=100).tolist()
    bw = kde_bandwidth(np.asarray(x), "scott")
    grid = np.linspace(-3.0, 3.0, 64)
    density = kde_compute_grid(np.asarray(x), bw, grid)
    cases.append({
        "name": "scott_normal_n100",
        "input": x,
        "bandwidth": "scott",
        "n": 64,
        "extent": [-3.0, 3.0],
        "cumulative": False,
        "expected_bandwidth": float(bw),
        "value": grid.tolist(),
        "density": density.tolist(),
    })

    # Case 2: silverman bandwidth, larger sample
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 0.5, size=200).tolist()
    bw = kde_bandwidth(np.asarray(x), "silverman")
    grid = np.linspace(0.0, 4.0, 128)
    density = kde_compute_grid(np.asarray(x), bw, grid)
    cases.append({
        "name": "silverman_normal_n200",
        "input": x,
        "bandwidth": "silverman",
        "n": 128,
        "extent": [0.0, 4.0],
        "cumulative": False,
        "expected_bandwidth": float(bw),
        "value": grid.tolist(),
        "density": density.tolist(),
    })

    # Case 3: fixed bandwidth, cumulative
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    bw = 0.5
    grid = np.linspace(-1.0, 5.0, 32)
    density = kde_compute_grid(np.asarray(x), bw, grid, cumulative=True)
    cases.append({
        "name": "fixed_h05_cumulative",
        "input": x,
        "bandwidth": "fixed",
        "fixed_bandwidth": 0.5,
        "n": 32,
        "extent": [-1.0, 5.0],
        "cumulative": True,
        "expected_bandwidth": 0.5,
        "value": grid.tolist(),
        "density": density.tolist(),
    })
    return cases


# ---------- LOESS ----------

def tricube(u):
    u = np.abs(u)
    w = np.where(u < 1.0, (1.0 - u ** 3) ** 3, 0.0)
    return w


def loess_at_point(x, y, x0, bw_frac, degree):
    n = len(x)
    k = max(int(np.ceil(bw_frac * n)), degree + 1)
    dists = np.abs(x - x0)
    idx = np.argsort(dists)[:k]
    h = dists[idx[-1]]
    if h == 0.0:
        # All k nearest are at the same x; return mean of their y.
        return float(np.mean(y[idx]))
    w = tricube((x[idx] - x0) / h)
    if degree == 1:
        # Solve weighted normal equations 2x2 for [a, b] in y = a + b*x.
        X = np.column_stack([np.ones(k), x[idx]])
    elif degree == 2:
        X = np.column_stack([np.ones(k), x[idx], x[idx] ** 2])
    else:
        raise ValueError(degree)
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y[idx]
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return float("nan")
    if degree == 1:
        return float(beta[0] + beta[1] * x0)
    return float(beta[0] + beta[1] * x0 + beta[2] * x0 ** 2)


def loess_compute_grid(x, y, x_grid, bw_frac, degree):
    return np.array([loess_at_point(np.asarray(x), np.asarray(y), x0, bw_frac, degree) for x0 in x_grid])


def loess_cases():
    cases = []

    # Case 1: degree=1 on a noisy sine
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 6.28, 50)
    y = np.sin(x) + rng.normal(0.0, 0.1, size=50)
    grid = np.linspace(0.0, 6.28, 25)
    fit = loess_compute_grid(x, y, grid, bw_frac=0.5, degree=1)
    cases.append({
        "name": "deg1_sine",
        "x": x.tolist(),
        "y": y.tolist(),
        "bandwidth": 0.5,
        "degree": 1,
        "n": 25,
        "x_grid": grid.tolist(),
        "y_fit": fit.tolist(),
    })

    # Case 2: degree=2 on a noisy quadratic
    x = np.linspace(-2.0, 2.0, 60)
    y = x ** 2 + rng.normal(0.0, 0.05, size=60)
    grid = np.linspace(-2.0, 2.0, 30)
    fit = loess_compute_grid(x, y, grid, bw_frac=0.4, degree=2)
    cases.append({
        "name": "deg2_quadratic",
        "x": x.tolist(),
        "y": y.tolist(),
        "bandwidth": 0.4,
        "degree": 2,
        "n": 30,
        "x_grid": grid.tolist(),
        "y_fit": fit.tolist(),
    })

    return cases


# ---------- Main ----------

def main():
    here = Path(__file__).resolve().parent
    out_path = here / "stat_refs.json"
    payload = {
        "_pinned_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "kde": kde_cases(),
        "loess": loess_cases(),
    }

    # Optional sanity-check log line so the operator can confirm scipy alignment.
    case = payload["kde"][0]
    grid = np.asarray(case["value"])
    sci = kde_sanity_check(np.asarray(case["input"]), case["expected_bandwidth"], grid)
    ours = np.asarray(case["density"])
    max_abs_diff = float(np.max(np.abs(sci - ours)))
    print(f"kde scipy sanity diff: {max_abs_diff:.3e}", file=sys.stderr)

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
