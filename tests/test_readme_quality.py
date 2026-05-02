from pathlib import Path


def test_readme_mentions_core_strategy():
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")

    required_terms = [
        "DINOv3",
        "LoRA",
        "Register-aware",
        "Top-4",
        "Median",
        "448",
        "대상",
    ]

    missing = [term for term in required_terms if term not in text]
    assert not missing, f"README missing terms: {missing}"
