import itertools
import math

from omegaconf import OmegaConf


def load_config(config_path: str) -> dict:
    """
    Load a yaml config file.
    :param config_path: Path to the yaml config file.
    :return: Dictionary of the config.
    """
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


def logsumexp(*args):
    """
    Compute sum of log-transformed input probabilities, copied from Awni Hannun at
    https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0
    :param args: Log-transformed probabilities.
    :return: Sum of log-transformed probabilities.
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def generate_grid_search_configs(base_config_path: str,
                                 free_params: dict,
                                 expt_dir: str) -> None:
    """
    Generate a directory of configs for a grid search based on a base config and a dictionary of free parameters.
    :param base_config_path: The path to the base config with all other params set.
    :param free_params: Dictionary of free parameters and a list of their values for the grid search.
    :param expt_dir: Directory to save the configs to.
    """
    base_config = load_config(base_config_path)
    param_names = []
    value_lists = []
    for param, values in free_params.items():
        param_names.append(param)
        value_lists.append(values)

    permutations = list(itertools.product(*value_lists))
    for permutation in permutations:
        config_name = ''
        for idx, param_value in enumerate(permutation):
            param_name = param_names[idx]
            base_config[param_name] = param_value
            config_name += f'{param_name}_{param_value}_'
        config_name = config_name[:-1]
        OmegaConf.save(OmegaConf.create(base_config), f'{expt_dir}/config_{config_name}.yaml')
