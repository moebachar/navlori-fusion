"""Hydra entry point for training."""

import hydra
from omegaconf import DictConfig

from src.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = setup_logging()
    log.info("NavLoRI Fusion — Training")
    log.info(f"Pipeline stages: {cfg.pipeline.stages}")
    log.info(f"Device: {cfg.device}")

    # TODO: Build pipeline from config
    # TODO: Load data
    # TODO: Training loop with MLflow + TensorBoard


if __name__ == "__main__":
    main()
