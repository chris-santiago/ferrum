"""Readers of a spec's pipeline wire: the key each step publishes, at any depth.

A chart's pipeline is a list of steps; a ``branch`` runs its own list and
publishes its tail under its name and every named output inside it under
``secondary_key(branch, key)``. A layer's ``data_source`` names one of those
keys. These readers map each key back to the step whose output it is, so a
test asks "which step does this layer read" without restating the scoping.
"""

from __future__ import annotations

from ferrum._core import secondary_key


def published_steps(wire: dict) -> dict[str, dict]:
    """Each key *wire*'s pipeline publishes, mapped to the step whose output it is.

    A named step publishes under its name; a branch publishes its tail, the
    output of its last unnamed step other than a pass-through identity (a
    branch with none publishes its input, which no step computed); both are
    scoped by every enclosing branch. Each mapped step is a copy whose
    ``name`` is the key a layer reads.
    """
    out: dict[str, dict] = {}

    def walk(steps: list[dict], scope: str | None) -> None:
        for step in steps:
            name = step.get("name")
            if name is None:
                continue
            key = name if scope is None else secondary_key(scope, name)
            if step["type"] == "branch":
                inner = step["transforms"]
                chained = [t for t in inner if "name" not in t and t["type"] != "identity"]
                if chained:
                    out[key] = {**chained[-1], "name": key}
                walk(inner, key)
            else:
                out[key] = {**step, "name": key}

    walk(wire.get("transforms") or [], None)
    return out


def all_steps(wire: dict) -> list[dict]:
    """Every step of *wire*'s pipeline, each branch's own steps right after it, at any depth."""

    def walk(steps: list[dict]) -> list[dict]:
        out: list[dict] = []
        for step in steps:
            out.append(step)
            if step["type"] == "branch":
                out += walk(step["transforms"])
        return out

    return walk(wire.get("transforms") or [])


def steps_of_type(wire: dict, kind: str) -> list[dict]:
    """The published steps of type *kind*, each named by the key a layer reads it under."""
    return [step for step in published_steps(wire).values() if step["type"] == kind]


def operand_branches(wire: dict) -> list[str]:
    """The names of the top-level branches: one per ``+`` operand of a composition."""
    return [step["name"] for step in wire.get("transforms") or [] if step["type"] == "branch"]
