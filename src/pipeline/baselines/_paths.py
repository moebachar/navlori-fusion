"""Canonical paths to external_methods submodules.

Edit only if the project root layout changes (the submodule directory
names are fixed to ``external_methods/<repo_name>/``).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_METHODS = PROJECT_ROOT / "external_methods"

WLANLOC_SRC = EXTERNAL_METHODS / "wlan_localization" / "src"
RONIN_SRC = EXTERNAL_METHODS / "ronin" / "source"
RONIN_LISTS = EXTERNAL_METHODS / "ronin" / "lists"
TARTANVO_ROOT = EXTERNAL_METHODS / "tartanvo"
DPVO_ROOT = EXTERNAL_METHODS / "dpvo"

__all__ = [
    "PROJECT_ROOT", "EXTERNAL_METHODS",
    "WLANLOC_SRC", "RONIN_SRC", "RONIN_LISTS",
    "TARTANVO_ROOT", "DPVO_ROOT",
]
