from glob import glob
from pathlib import Path

from fusebuild import shell_action, top_output_dir
from fusebuild.python.pyc import pyc_mappings


def mypy_actions() -> None:
    pyfiles = glob("*.py")
    for p in pyfiles:
        shell_action(
            name=f"{p}.mypy",
            cmd="\n".join(
                [
                    f". {top_output_dir()}/venv/bin/activate",
                    f"echo starting mypy {p}",
                    f"mypy {p}",
                    f"echo mypy {p} done",
                ]
            ),
            category="linting",
            mappings=pyc_mappings(),
        )
