from pathlib import Path


def test_required_files_exist():
    root = Path(__file__).resolve().parents[1]

    required = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        ".dockerignore",
        "train.py",
        "inference.py",
        "src/models.py",
        "src/dataset.py",
        "src/utils.py",
        "config/config.yaml",
    ]

    missing = [path for path in required if not (root / path).exists()]
    assert not missing, f"Missing files: {missing}"


def test_large_directories_are_excluded():
    root = Path(__file__).resolve().parents[1]

    excluded = [
        "model",
        "train_data",
        "test_data",
        "__MACOSX",
    ]

    existing = [path for path in excluded if (root / path).exists()]
    assert not existing, f"Excluded large/system directories should not be committed: {existing}"
