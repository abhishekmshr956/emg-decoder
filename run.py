import argparse
import os
from typing import Tuple, Type, Union

from pytorch_lightning import Trainer, LightningModule

from emg_decoder.src.models.conkeydecoder import ConKeyDecoder
from emg_decoder.src.models.rtdecoder import RTDecoder
from emg_decoder.src.models.keydecoder import KeyDecoder
from emg_decoder.src.models.train import train
from omegaconf import OmegaConf

repo_dir = os.path.dirname(os.path.realpath(__file__))


def main(config: dict,
         accelerator: str = None,
         devices: Union[int, list] = None,
         checkpoint: str = None,
         experiment_dir: str = None) -> Tuple[Type[LightningModule], Trainer]:
    """
    Main function for training an EMG decoding model.
    :param experiment_dir: Directory to hold checkpoints and logs
    :param config: Config dict containing all training hyperparameters.
    :param accelerator: Type of accelerator to use (e.g. "cpu", "gpu", or "mps").
    :param devices: Indices of devices to use (e.g. [0, 1, 2, 3]).
    :param checkpoint: Checkpoint path to load.
    :return: System and trainer objects.
    """
    if accelerator == 'gpu' or accelerator == 'cuda':
        devices = [int(devices)]

    if experiment_dir is None:
        experiment_dir = f"{repo_dir}/models/{config['name']}"

    if config['system'].lower() == 'keydecoder':
        system_type = KeyDecoder
    elif config['system'].lower() == 'rtdecoder':
        system_type = RTDecoder
    elif config['system'].lower() == 'conkeydecoder':
        system_type = ConKeyDecoder
    else:
        raise NotImplementedError(f"System type {config['system']} not implemented.")

    system, trainer = train(
        system_type,
        config,
        experiment_dir=experiment_dir,
        accelerator=accelerator,
        devices=devices,
        ckpt_path=checkpoint
    )
    return system, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='EMG Decoder Training',
        description='Train an EMG decoding model based on a provided config file')
    parser.add_argument('config')
    parser.add_argument('-a', '--accelerator', default='cpu')
    parser.add_argument('-d', '--devices', default="auto")
    parser.add_argument('-c', '--checkpoint', default=None)
    args = parser.parse_args()

    if os.path.isfile(args.config):
        hparams = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        main(hparams, args.accelerator, args.devices, args.checkpoint)

    else:
        print(f"Config argument is a directory, running all configs found in {args.config}")
        for config in os.listdir(args.config):
            if os.path.splitext(config)[-1].lower() == '.yaml':
                print(f"Running {config}")
                try:
                    hparams = OmegaConf.to_container(OmegaConf.load(f"{args.config}/{config}"), resolve=True)
                    batch_dir_name = os.path.basename(os.path.normpath(args.config))
                    main(hparams,
                         accelerator=args.accelerator,
                         devices=args.devices,
                         checkpoint=args.checkpoint,
                         experiment_dir=f"{repo_dir}/models/{batch_dir_name}/{hparams['name']}")
                except Exception as e:
                    print(f"Error running {config}: {e}, continuing...")
                    continue
