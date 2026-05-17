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


@nox.session()
def build(session: nox.Session) -> None:
    """Build the ferrum package in an isolated environment."""
    session.run("uv", "build", external=True)
