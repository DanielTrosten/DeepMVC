import pickle
from argparse import ArgumentParser, SUPPRESS

import helpers
from .constants import *
from .config import Config
from .templates.experiment import Experiment
from . import experiments


DEFAULT_SEP = "/"
SKIP_KEYS = ("illegal_vars",)


def parse_config_name_arg():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_name", required=True)
    return parser.parse_known_args()[0].config_name


def dict_from_cfg(cfg, sep=DEFAULT_SEP):
    flat = {}
    nested = cfg.dict()
    _insert_from_dict_or_iterable(flat, nested, prefix="", sep=sep)
    return flat


def _insert_from_dict_or_iterable(dct, other, prefix, sep):
    legal_types = (dict, list, tuple)
    iterable_types = (list, tuple)
    assert isinstance(other, legal_types), f"Illegal type: {type(other)}"
    if isinstance(other, iterable_types):
        # Convert iterables to a dict where keys are indices
        other = {str(i): elem for i, elem in enumerate(other)}

    for key, value in other.items():
        if key in SKIP_KEYS:
            continue

        full_key = prefix + sep + key if prefix else key
        if isinstance(value, legal_types):
            _insert_from_dict_or_iterable(dct, value, prefix=full_key, sep=sep)
        else:
            dct[full_key] = value


def parse_cli_args(cfg_dict):
    parser = ArgumentParser(argument_default=SUPPRESS)

    for key, value in cfg_dict.items():
        if isinstance(value, bool):
            value_type = helpers.str2bool
        elif isinstance(value, (int, float)):
            value_type = type(value)
        elif isinstance(value, str):
            value_type = str
        else:
            value_type = None
        parser.add_argument("--" + key, dest=key, type=value_type)

    args, unknown = parser.parse_known_args()
    # The only unknown argument we expect to find is "-c" or "--config".
    assert (len(unknown) == 2) and (unknown[0] in ("-c", "--config")), f"Got unexpected unknown arguments: {unknown}"
    args = vars(args)
    return args


def insert_tied_args(cli_args):
    # Syntax: --tied_args base_arg1[copy1,copy2]|base_arg2[copy1,copy2]
    ta = cli_args["tied_args"]
    for arg in ta.split("|"):
        print(f"Processing {arg}")
        base, copies = arg.split("[")
        copies = copies.rstrip("]")
        copies = copies.split(",")

        assert base in cli_args, f"Base argument '{base}' for tied arguments '{arg}' not found in provided arguments."

        for copy in copies:
            assert copy not in cli_args, f"Already found copy '{copy}' of tied arguments '{arg}' in provided " \
                                         f"arguments. Overriding given arguments with tied arguments is not allowed."

            cli_args[copy] = cli_args[base]


def set_cfg_value(cfg, key_list, value):
    last_key = key_list[-1]
    assert last_key not in cfg.illegal_vars, f"Setting attribute '{last_key}' on Config of type {type(cfg)} " \
                                             f"is not allowed."

    sub_cfg = cfg
    for key in key_list[:-1]:
        if isinstance(sub_cfg, list):
            sub_cfg = sub_cfg[int(key)]
        else:
            sub_cfg = getattr(sub_cfg, key)

    if isinstance(sub_cfg, list):
        sub_cfg[int(last_key)] = value
    else:
        setattr(sub_cfg, last_key, value)


def update_cfg(cfg, cli_args, sep=DEFAULT_SEP):
    for key, value in cli_args.items():
        set_cfg_value(cfg, key_list=key.split(sep), value=value)


def get_config_by_name(name):
    try:
        cfg = getattr(experiments, name)
        assert isinstance(cfg, Experiment), f"Found config with invalid type: '{type(cfg)}' (Expected {Experiment})."
    except Exception as err:
        raise RuntimeError(f"Config not found: {name}") from err
    return cfg


def get_config_from_file(name=None, tag=None, file_path=None, run=0):
    if file_path is None:
        file_path = MODELS_DIR / f"{name}-{tag}" / f"run-{run}" / "config.pkl"
    with open(file_path, "rb") as f:
        cfg = pickle.load(f)
    return cfg


def get_experiment_config():
    # Get the config name from cli-args
    name = parse_config_name_arg()
    # Get the config object for the specified name
    cfg = get_config_by_name(name)
    # Convert the config object to a flat dict
    cfg_dict = dict_from_cfg(cfg)
    # Get config cli-args
    cli_args = parse_cli_args(cfg_dict)
    # If we got any arguments that are supposed to be tied. Expand these in the 'cli_args' dict.
    if cli_args.get("tied_args", None) is not None:
        insert_tied_args(cli_args)
    # Update the config-object with the cli args
    update_cfg(cfg, cli_args)
    # Set global parameters
    cfg.set_globs()
    return name, cfg
