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
    session.install("maturin>=1.7,<2.0")
    session.run("maturin", "develop")
    session.install(".[all]", "pytest>=8", "scipy>=1.10", "skops>=0.9", "resvg-py>=0.3")
    session.run("pytest", *session.posargs)


@nox.session
def build(session: nox.Session) -> None:
    """Build the Rust extension in an isolated venv."""
    session.install("maturin>=1.7,<2.0")
    session.run("maturin", "develop", "--release")
