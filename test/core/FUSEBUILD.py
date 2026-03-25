import glob

from fusebuild import BwrapSandbox, NoSandbox, get_action, shell_action
from fusebuild.python import mypy_actions, pyc_actions, pyc_mappings

pyc_actions()
mypy_actions()

for test in glob.glob("test_*.py"):
    shell_action(
        name=test[0:-3],
        cmd="\n".join(
            [
                ". $OUTPUT_DIR/../../../venv/bin/activate",
                f"python {test}",
            ]
        ),
        category="test",
        mappings=pyc_mappings(),
    )
