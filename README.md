# EMG Decoder

A package for decoding EMG signals into keypresses.

### Installation

Install required packages and run setup. (Make sure that you are using `python --version >= 3.10`)
```bash
pip install -r requirements.txt
pip install -e .
```

Note: If running on a new environment make sure to have tensorflow installed (this will also install a local version of
tensorboard in your conda environment) with `conda install tensorflow` for logging purposes.

### Usage
Create a configuration file for your data based on the example in `configs/example.yaml`.
From the time of writing, note that new runs will automatically save the preprocessed data to the directory 
`data/processed/config['name']`, but you will need to manually paste this path into
`config['data']['preprocessed_data_dir']` to load the saved data in future runs.

From the repository directory, run the training script with
```bash
python run.py path_to_config.yaml -a accelerator_type -d device_index -c optional_path_to_checkpoint
```

`run.py` will call the `train` function, which sets up an instance of `RTDecoder` (a subclass of 
[LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)), a 
[Pytorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) object, associated data, 
logging, and model checkpoint folders, and calls `trainer.fit()` on the `RTDecoder` instance. The instantiation of
`RTDecoder` will process and save the data from raw (or load existing preprocessed data) and set up the model, optimizer, loss function, and train/val/test steps. 
The `Trainer` object will fit the model to the data, log the results to [TensorBoard](https://www.tensorflow.org/tensorboard), 
and save checkpoints as specified by the configuration file. These logs and model checkpoints can be found in 
`lightning_logs` and `models` within `root_dir/models/name`, where `root_dir` and `name` are specified in the top level 
of the configuration file.
