import collections
import pathlib
import shutil


def get_directory(module, name):
    path = pathlib.Path(
        "tests",
        "temp",
        module,
        pathlib.Path(__file__).stem,
        name,
    )
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=False)
    return path


def get_mock_args(args):
    return collections.namedtuple(
        "args",
        (arg[2:].replace("-", "_") for arg in args.keys()),
        defaults=(args[arg] for arg in args.keys()),
    )
