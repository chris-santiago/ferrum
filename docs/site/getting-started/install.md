# Install

Ferrum runs on Python 3.10 or newer. The Rust core ships as a pre-built extension inside the wheel, so installation does not require a Rust toolchain on your machine — `pip install` is the entire setup.

## Standard install

=== "pip"

    <!--pytest.mark.skip-->
    ```bash
    pip install ferrum-viz[all]
    ```

=== "uv"

    <!--pytest.mark.skip-->
    ```bash
    uv add ferrum-viz[all]
    ```

=== "poetry"

    <!--pytest.mark.skip-->
    ```bash
    poetry add ferrum-viz[all]
    ```

This installs Ferrum with all optional dependencies — model diagnostics, SHAP explanations, and interactive Jupyter rendering. For a leaner install (e.g. containers or CI), drop `[all]`: `pip install ferrum-viz`.

!!! info "Pre-1.0 release status"
    Ferrum is currently pre-1.0 (the live version is shown in the page footer). The public surface is stabilizing toward the 1.0 commitment described in [Why Ferrum](why-ferrum.md), but APIs may shift between minor versions until 1.0 is cut. Pin a specific version in production code.

## What you don't need

Ferrum deliberately avoids common visualization-stack pain points. You do not need any of these to use Ferrum:

- **matplotlib** — not a dependency. Ferrum's rendering pipeline is independent.
- **Cairo, X11, or any display server** — rendering is pure Rust. SVG and PNG output work in headless environments.
- **A Jupyter kernel or notebook runtime** — Ferrum renders identically from scripts, CI, SSH sessions, containers, and Kubernetes jobs.
- **A Rust toolchain** — the compiled extension ships in the wheel.

This is intentional. A visualization library that requires fragile system dependencies is harder to use in the places where real work happens, so Ferrum favors a pure-Rust rendering stack, wheel-based installation, and headless execution paths.

## Verify the install

Open a Python REPL and import Ferrum:

```python
import ferrum

print(ferrum.__version__)
```

You should see a version string printed and no import error. If the import succeeds, the compiled Rust core (`ferrum._core`) loaded correctly and Ferrum is ready to use.

## Supported Python versions

Ferrum supports CPython 3.10 and newer. Older Python versions are not tested. Free-threaded Python builds are not yet exercised in CI.

## Optional extras

Ferrum ships four optional dependency groups for features that need external packages:

| Extra | What it adds | Install |
|---|---|---|
| `models` | Diagnostics from a **fitted model** (`fm.roc_chart(model, X, y)`, etc.) via sklearn. Not needed for the raw-array path — see below. | `pip install ferrum-viz[models]` |
| `shap` | SHAP explanations (beeswarm, bar, waterfall) via sklearn + shap | `pip install ferrum-viz[shap]` |
| `jupyter` | Interactive rendering via anywidget (WASM/GPU canvas in notebooks) | `pip install ferrum-viz[jupyter]` |
| `all` | Everything above | `pip install ferrum-viz[all]` |

The base install (`pip install ferrum-viz`) covers the grammar API, all marks, themes, composition, static SVG/PNG rendering, and DataFrame ingestion. The extras add fitted-model diagnostics (sklearn), SHAP explanations, and Jupyter interactivity.

Diagnostic charts built from raw `y_true=` / `y_pred=` arrays need **none** of these extras: the metric computations (ROC, precision-recall, calibration, confusion matrix, threshold sweep) run on Ferrum's built-in Rust kernels. sklearn is required only when you pass a fitted model object.

## Optional dataframe ecosystems

Ferrum accepts data from the full Python dataframe ecosystem through Narwhals. The runtime dependencies installed above are sufficient for Polars and Arrow inputs. If you want to pass pandas or other dataframe types directly to Ferrum, install them alongside:

<!--pytest.mark.skip-->
```bash
pip install pandas
```

You do not need to convert your data before plotting — Narwhals handles the interop layer transparently.

## Where to go next

- Render [your first plot](first-plot.md) to confirm everything works end-to-end.
- Read [Why Ferrum](why-ferrum.md) for the motivation behind the design choices above.
