"""Evaluate a trained model."""

import hydra
from omegaconf import DictConfig

from src.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = setup_logging()
    log.info("NavLoRI Fusion — Evaluation")

    # TODO: Load model from MLflow registry or checkpoint
    # TODO: Run evaluation
    # TODO: Generate visualizations


if __name__ == "__main__":
    main()
