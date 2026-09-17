"""Regression guard: Synthetic Generation v2 must not modify v1 synth scripts.

Synthetic Generation v2 (see docs/superpowers/plans/2026-09-17-synthetic-generation-v2.md)
is a *parallel* pipeline. This test asserts that v1's legacy expense-noise path is still
present, proving v1 was not "upgraded in place" when v2 was added.
"""

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "training" / "scripts"


def test_v1_generate_transactions_still_has_expense_gaussian() -> None:
    src = (SCRIPTS_DIR / "generate_transactions.py").read_text()
    assert "rng.normal(1.0, 0.15)" in src
    assert "max(0.5, min(2.0, variation))" in src


def test_v1_scripts_exist() -> None:
    for name in (
        "generate_personas.py",
        "generate_transactions.py",
        "synthesizer.py",
        "preprocessor.py",
    ):
        assert (SCRIPTS_DIR / name).is_file()
