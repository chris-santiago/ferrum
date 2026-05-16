"""Smoke test that the schwabish skill and judge agent files exist and parse."""

from pathlib import Path

import yaml


def test_skill_md_has_frontmatter():
    skill_path = Path(".claude/skills/schwabish/SKILL.md")
    text = skill_path.read_text()
    assert text.startswith("---")
    body = text.split("---", 2)
    assert len(body) == 3
    fm = yaml.safe_load(body[1])
    assert fm["name"] == "schwabish"
    assert "description" in fm


def test_judge_agent_has_read_only_tools():
    agent_path = Path(".claude/agents/schwabish-judge.md")
    text = agent_path.read_text()
    body = text.split("---", 2)
    fm = yaml.safe_load(body[1])
    tools = {t.strip() for t in fm["tools"].split(",")}
    assert "Edit" not in tools
    assert "Write" not in tools
    assert "Read" in tools


def test_judge_prompt_embeds_principles_doc():
    prompt = Path(".claude/skills/schwabish/judge_prompt.md").read_text()
    assert "T1" in prompt and "T2" in prompt and "T3" in prompt and "T4" in prompt
    assert "objective" in prompt.lower()


def test_eligibility_list_lists_objective_findings():
    elig = Path(".claude/skills/schwabish/apply_eligibility.md").read_text()
    for finding_id in [
        "T4_auc_label_missing",
        "T4_ap_label_missing",
        "T4_brier_label_missing",
        "T4_residual_metrics_missing",
        "T4_cell_counts_missing",
        "T4_importance_values_missing",
        "T2_direct_labels_eligible",
        "T4_pr_baseline_missing",
        "T4_residual_zero_line_missing",
        "T4_calibration_diagonal_missing",
    ]:
        assert finding_id in elig, f"missing eligibility entry: {finding_id}"
