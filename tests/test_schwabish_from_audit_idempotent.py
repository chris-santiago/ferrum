"""Idempotence + scope contract for the schwabish-fixer agent.

These are documentation/contract tests — actually invoking the agent
requires the full Agent tool runtime, so we assert the agent's
frontmatter and body document the contractual obligations the
orchestrator relies on.
"""

from pathlib import Path


def test_fixer_agent_documents_idempotence():
    text = Path(".claude/agents/schwabish-fixer.md").read_text()
    assert "idempotence" in text.lower() or "idempotent" in text.lower(), (
        "schwabish-fixer must document idempotence"
    )


def test_fixer_restricted_to_gallery():
    text = Path(".claude/agents/schwabish-fixer.md").read_text().lower()
    assert "do not edit `src/ferrum/`" in text or (
        "restricted to" in text and "gallery/" in text
    ), "schwabish-fixer must document scope restriction"


def test_fixer_does_not_commit():
    text = Path(".claude/agents/schwabish-fixer.md").read_text().lower()
    assert "do not commit" in text, (
        "schwabish-fixer must document that committing belongs to the orchestrator"
    )
