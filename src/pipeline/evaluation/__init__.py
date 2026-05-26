"""Encoder evaluation metrics + run-2 main-results table.

- Low-level encoder probes (linear, kNN, alignment/uniformity,
  effective dim, smoothness, trustworthiness) live in
  ``encoder_eval.py``.
- The paper-ready ``MainResultsTable`` lives in
  ``main_results_table.py`` (PLAN_29 consolidation; excludes IPIN
  + MoTTransformer per ``handoff/SCIENTIST_NOTE_notebook-
  exclusions.md``).
"""

from .encoder_eval import (
    alignment_uniformity,
    effective_dimensionality,
    evaluate_encoder,
    extract_embeddings,
    knn_probe,
    linear_probe,
    print_report,
    temporal_smoothness,
    trustworthiness,
)
from .main_results_table import (
    MainResultsTable,
    PAPER_ARCHS,
    PAPER_DATASETS,
    SOTA_COLS,
    TableCell,
)

__all__ = [
    # encoder probes
    "linear_probe",
    "alignment_uniformity",
    "effective_dimensionality",
    "temporal_smoothness",
    "knn_probe",
    "trustworthiness",
    "extract_embeddings",
    "evaluate_encoder",
    "print_report",
    # main results table
    "MainResultsTable",
    "TableCell",
    "PAPER_DATASETS",
    "PAPER_ARCHS",
    "SOTA_COLS",
]
