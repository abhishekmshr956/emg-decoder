from pathlib import Path
from typing import Tuple, Type

from pytorch_lightning import Trainer, seed_everything, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar


def train(system_class: Type[LightningModule],
          config: dict,
          experiment_dir: str = "experiments",
          accelerator: str = None,
          devices: list = None,
          ckpt_path: str = None,
          run: bool = True) -> Tuple[Type[LightningModule], Trainer]:
    """
    Training function for a given system class and config.
    :param system_class: LightningModule class to train
    :param config: Config dict
    :param experiment_dir: Directory to save models and logs
    :param accelerator: Type of accelerator to use (e.g. "cpu", "gpu", or "mps")
    :param devices: Indices of devices to use (e.g. [0, 1, 2, 3])
    :param ckpt_path: Path to checkpoint to load
    :param run: Whether to run the training loop or just return the system and trainer
    :return: System and trainer objects
    """

    trainer_config = config["trainer"]
    seed_everything(config["random_seed"])
    system = system_class(config, accelerator=accelerator)

    experiment_dir = Path(f"{experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=trainer_config['save_top_k'],
        save_last=trainer_config['save_last'],
        monitor=config['trainer']['monitor'],
        mode=config['trainer']['mode'],
        dirpath=f"{experiment_dir}/models",
        filename="bell-{epoch:02d}-{val_loss:.2f}",
        auto_insert_metric_name=True
    )

    trainer = Trainer(
        max_epochs=trainer_config["max_epochs"],
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config['trainer']['monitor'], patience=trainer_config["patience"]),
            TQDMProgressBar(refresh_rate=20)
        ],
        default_root_dir=str(experiment_dir),
        accelerator=accelerator,
        devices=devices
    )
    if run:
        trainer.fit(system, ckpt_path=ckpt_path)
    return system, trainer
