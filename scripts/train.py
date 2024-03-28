from pathlib import Path

from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.trainers.ae_trainer import AETrainerModule
from cmonge.trainers.conditional_monge_trainer import ConditionalMongeTrainer
from cmonge.utils import load_config
from loguru import logger


def train_conditional_monge(
    drug: str,
    ood: str,
    config_path: Path,
    logger_path: Path,
):
    logger.info(f"Experiment: Training model on {drug}")
    config = load_config(config_path)

    config.condition.conditions = [
        str(drug) + str(b) for b in ["-10", "-100", "-1000", "-10000"]
    ]
    config.condition.ood = [f"{drug}-{ood}"]
    config.data.drug_condition = drug

    # Train an AE model to reduce data dimension
    config.data.ae = True
    config.data.reduction = None
    datamodule = ConditionalDataModule(config.data, config.condition)
    ae_trainer = AETrainerModule(config.ae)
    ae_trainer.train(datamodule)

    # Train conditional monge model
    config.data.ae = False
    config.data.reduction = "ae"
    datamodule = ConditionalDataModule(config.data, config.condition)
    trainer = ConditionalMongeTrainer(
        jobid=1, logger_path=logger_path, config=config.model, datamodule=datamodule
    )
    trainer.train(datamodule)
    trainer.evaluate(datamodule)


if __name__ == "__main__":
    import typer

    typer.run(train_conditional_monge)
