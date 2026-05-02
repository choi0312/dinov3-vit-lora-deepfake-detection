import re
from pathlib import Path


def test_no_huggingface_token_literal():
    root = Path(__file__).resolve().parents[1]
    token_pattern = re.compile(r"hf_[A-Za-z0-9_\-]{20,}")

    hits = []

    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if ".git" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".yaml", ".yml", ".toml"}:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        if token_pattern.search(text):
            hits.append(str(path.relative_to(root)))

    assert not hits, f"Hardcoded HuggingFace token-like strings found: {hits}"
