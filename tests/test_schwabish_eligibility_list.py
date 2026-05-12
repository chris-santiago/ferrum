"""Subjective findings never appear in the eligibility list."""
import re
from pathlib import Path


def test_eligibility_list_excludes_subjective_finding_ids():
    text = Path(".claude/skills/schwabish/apply_eligibility.md").read_text()
    body = text.split("## Non-eligible")[0]  # only the eligible section
    # subjective IDs that must NOT appear in the eligible section
    forbidden = ["T1_active_title", "T3_callout", "T1_subtitle"]
    for fid in forbidden:
        assert fid not in body, (
            f"subjective finding {fid!r} found in eligibility list"
        )


def test_eligibility_list_only_lists_T2_or_T4_in_eligible_section():
    text = Path(".claude/skills/schwabish/apply_eligibility.md").read_text()
    eligible_section = text.split("## Non-eligible")[0]
    finding_ids = re.findall(r"T\d_\w+", eligible_section)
    for fid in finding_ids:
        assert fid.startswith("T2_") or fid.startswith("T4_"), (
            f"non-T2/T4 finding {fid!r} in eligible section"
        )
