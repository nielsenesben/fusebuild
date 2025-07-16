import functools
from pathlib import Path

from fusebuild.core.libfusebuild import output_folder_root_str


@functools.cache
def topdir(here: Path = Path(".")) -> Path:
    here = Path(here).absolute()
    if here.is_dir():
        best = here
    else:
        best = here.parent

    for p in here.absolute().parents:
        if (p / "FUSEBUILD.py").exists():
            best = p

    return best


@functools.cache
def top_output_dir(here: Path = Path(".")) -> Path:
    td = topdir(here)
    return Path(output_folder_root_str + str(td.absolute()))
