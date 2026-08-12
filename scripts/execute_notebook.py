"""Execute one notebook with the project interpreter and save its real outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def execute(relative_path: str) -> Path:
    path = ROOT / relative_path
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        # The venv's bundled python3 spec resolves to this exact interpreter.
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"Executed and saved: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", nargs="?", default="notebooks/04_rl_model_comparison.ipynb")
    execute(parser.parse_args().notebook)
