from fusebuild import topdir, top_output_dir, PatternRemapToOutput, shell_action
from pathlib import Path
from glob import glob


def pyc_actions(t: Path = topdir(Path("."))):
    shell_action(name="pycache_gen_dir", cmd="true")

    pyfiles = glob("**/*.py") + glob("*.py")
    for p in pyfiles:
        shell_action(
            name=f"{p[:-3]}_gen_pyc",
            cmd="\n".join(
                [
                    f". {top_output_dir(t)}/venv/bin/activate",
                    "mkdir -p $OUTPUT_DIR/__pycache__",
                    f"python -m compileall {p}",
                ]
            ),
            category="build",
            mappings=pyc_mappings(),
        )


def pyc_mappings(td: Path = topdir(Path("."))) -> list[PatternRemapToOutput]:
    return [
        PatternRemapToOutput(
            f"({td.absolute()}.*)/__pycache__/(.*)\.(.*)\.pyc(.*)",
            "\\1/\\2_gen_pyc/\\2.\\3.pyc\\4",
        ),
        PatternRemapToOutput(
            f"({td.absolute()}.*)/__pycache__", f"\\1/pycache_gen_dir"
        ),
    ]
