from pathlib import Path

from loguru import logger

from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.trainers.ae_trainer import AETrainerModule
from cmonge.trainers.conditional_monge_trainer import ConditionalMongeTrainer
from cmonge.utils import load_config

logger_path = "logs/demo_logs.yml"
config_path = "configs/demo_config.yml"

config = load_config(config_path)
logger.info(f"Experiment: Training model on {config.condition.conditions}")


# Train an AE model to reduce data dimension
config.data.ae = True
config.data.reduction = None
datamodule = ConditionalDataModule(config.data, config.condition)
ae_trainer = AETrainerModule(config.ae)
ae_trainer.train(datamodule)

# Train conditional monge model
config.data.ae = False
config.ae.model.act_fn = "gelu"
config.data.reduction = "ae"
datamodule = ConditionalDataModule(config.data, config.condition, ae_config=config.ae)
trainer = ConditionalMongeTrainer(
    jobid=1, logger_path=logger_path, config=config.model, datamodule=datamodule
)
trainer.train(datamodule)
trainer.evaluate(datamodule)
