from fusebuild import BwrapSandbox, NoSandbox, get_action, shell_action
from fusebuild.python import mypy_actions, pyc_actions, pyc_mappings

pyc_actions()
mypy_actions()

shell_action(
    name="test_utils",
    cmd="\n".join(
        [
            ". $OUTPUT_DIR/../../../venv/bin/activate",
            "python -B test_utils.py",
        ]
    ),
    category="test",
    mappings=pyc_mappings(),
)
