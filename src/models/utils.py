import os

from emg_decoder.src.models.rtdecoder import RTDecoder
from emg_decoder.src.models.train import train
from emg_decoder.src.utils import load_config


class Experiment:
    def __init__(self, system_type, expt_dir: str, version: int = 0):
        """
        Class to load and store an experiment.
        :param system_type: System type for the experiment.
        :param expt_dir: Directory containing the experiment data.
        :param version: Version of the experiment to load.
        """
        self.system_type = system_type
        self.expt_dir = expt_dir
        self.version = version

        dir_list = [version_dir[-1] for version_dir in os.listdir(f"{expt_dir}/lightning_logs")]
        if len(dir_list) > 0 and str(version) in dir_list:
            print(f"Versions: {dir_list} available, loading version {version}...")
            self.config = load_config(f"{expt_dir}/lightning_logs/version_{version}/hparams.yaml")
        elif len(dir_list) > 0:
            raise FileNotFoundError(f"Specified version {version} not available. Versions: {dir_list} available.")
        else:
            raise FileNotFoundError(f"No versions available in {expt_dir}/lightning_logs. Check experiment directory.")

    def list_checkpoints(self):
        """
        List all checkpoints for the experiment.
        :return: List of checkpoints.
        """
        return os.listdir(f"{self.expt_dir}/models")

    def load_checkpoint(self, checkpoint_name: str, accelerator: str = 'gpu', devices: list = [0]):
        """
        Load a checkpoint from the experiment.
        :param checkpoint_name: Name of the checkpoint to load.
        :param devices: Indices of the device(s) to use.
        :param accelerator: Accelerator to use.
        :return: Model loaded from checkpoint.
        """
        system, trainer = train(
            self.system_type,
            self.config,
            experiment_dir=self.expt_dir,
            accelerator=accelerator,
            devices=devices,
            ckpt_path=f"{self.expt_dir}/models/{checkpoint_name}",
            run=False
        )

        return system, trainer
