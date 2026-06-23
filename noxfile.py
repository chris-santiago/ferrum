import nox

nox.options.default_venv_backend = "uv"


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Check and format with ruff (runs in the current environment)."""
    session.run("ruff", "check", "src", "tests", "--fix", external=True)
    session.run("ruff", "format", "src", "tests", external=True)


@nox.session
def test(session: nox.Session) -> None:
    """Run the test suite in an isolated venv."""
    session.run("uv", "sync", "--all-extras", "--all-groups", external=True)
    session.run("uv", "run", "pytest", "-n", "auto", *session.posargs, external=True)


@nox.session
def doctests(session: nox.Session) -> None:
    """Run the mark-desugar docstring examples as doctests.

    Scoped to the desugar modules so their Returns/Examples sections (which
    document ``MarkDesugarResult`` attribute access) stay correct and cannot
    silently drift back to the retired tuple-return protocol.  The public
    ``mark_*`` mixin docstrings under ``_chart_mixins`` carry illustrative
    examples that reference undefined estimators and are deliberately not
    collected here.
    """
    session.run("uv", "sync", "--all-extras", "--all-groups", external=True)
    session.run(
        "uv",
        "run",
        "pytest",
        "--doctest-modules",
        "src/ferrum/_layer.py",
        "src/ferrum/marks/composite.py",
        "src/ferrum/marks/statistical.py",
        "src/ferrum/marks/heavy_stat.py",
        *session.posargs,
        external=True,
    )


@nox.session(python=False)
def cargo_test(session: nox.Session) -> None:
    """Run Rust tests with correct macOS DYLD paths."""
    import subprocess

    result = subprocess.run(
        ["uv", "run", "python", "-c", "import sys; print(sys.base_prefix)"],
        capture_output=True,
        text=True,
    )
    base_prefix = result.stdout.strip()
    lib_dir = f"{base_prefix}/lib"
    python_exe = subprocess.run(
        ["uv", "run", "python", "-c", "import sys; print(sys.executable)"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    env = {
        "DYLD_LIBRARY_PATH": lib_dir,
        "RUSTFLAGS": f"-L {lib_dir}",
        "PYO3_PYTHON": python_exe,
        "PYTHONHOME": base_prefix,
    }
    session.run(
        "cargo",
        "test",
        "-p",
        "ferrum-core",
        *session.posargs,
        external=True,
        env=env,
    )


@nox.session()
def build(session: nox.Session) -> None:
    """Build the ferrum package in an isolated environment."""
    session.run("uv", "build", external=True)


@nox.session
def docs(session: nox.Session) -> None:
    """Build the docs site and fail on warnings."""
    session.run("uv", "run", "zensical", "build", external=True)
