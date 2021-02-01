"""
Common test utilities
"""

import collections
import pathlib
import shutil


def get_directory(module, file, name):
    """
    Return a temporary test directory for a module
    """
    path = pathlib.Path(
        "tests",
        "temp",
        module,
        file,
        name,
    )
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=False)
    return path


def get_mock_args(args):
    """
    Return argparse default arguments as a namedtuple
    """
    return collections.namedtuple(
        "args",
        (arg[2:].replace("-", "_") for arg in args.keys()),
        defaults=(args[arg] for arg in args.keys()),
    )
