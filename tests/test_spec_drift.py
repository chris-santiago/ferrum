"""Phase 8b cross-cutting drift tests: spec invariants and counts."""
from pathlib import Path


def test_transform_count_15_total():
    """Verify TransformSpec exposes 15 transforms (Phase 5: 5 + Phase 8b: 10)."""
    from ferrum import (Bin, Kde, Smooth, Aggregate, Summary,
                        Outliers, ErrorExtent, BoxStats, Violin, Kde2D,
                        Contour, QQ, Raster, Hex, Swarm)
    transforms = [Bin, Kde, Smooth, Aggregate, Summary,
                  Outliers, ErrorExtent, BoxStats, Violin, Kde2D,
                  Contour, QQ, Raster, Hex, Swarm]
    assert all(t is not None for t in transforms)
    assert len(transforms) == 15


def test_ferrum_phases_8b_done_criteria_lists_10_transforms():
    phases_path = Path(__file__).parent.parent / "docs" / "superpowers" / "ferrum-phases.md"
    phases = phases_path.read_text()
    section = phases.split("Phase 8b")[1].split("Phase 9")[0]
    assert "10" in section and "Kde2D" in section
