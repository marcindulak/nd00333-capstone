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
