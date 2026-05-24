from __future__ import annotations

import importlib.util
import sys


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "joblib",
    "fastapi",
    "streamlit",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    missing = []
    for package in REQUIRED_PACKAGES:
        found = importlib.util.find_spec(package) is not None
        print(f"{package}: {'OK' if found else 'MANQUANT'}")
        if not found:
            missing.append(package)
    if missing:
        print("\nInstaller les dependances avec: python -m pip install -r requirements.txt")


if __name__ == "__main__":
    main()
