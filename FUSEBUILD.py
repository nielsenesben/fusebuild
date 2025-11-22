from glob import glob

from fusebuild import NoSandbox, action, shell_action
from fusebuild.python import mypy_actions, pyc_actions, pyc_mappings

print(f"Loading {__file__}")
action(name="xeyes", cmd=["xeyes"], category="demo")

shell_action(
    name="depend_on_xeyes",
    cmd="test ! -f $OUTPUT_DIR/../xeyes/nonexistantfile",
    category="demo",
)

shell_action(
    name="venv",
    cmd="\n".join(
        [
            "pwd",
            "python -m venv $OUTPUT_DIR",
            ". $OUTPUT_DIR/bin/activate",
            "pip install -r requirements.txt",
        ]
    ),
)


pyc_actions()
mypy_actions()


shell_action(
    name="sleep",
    cmd="\n".join(["ls $OUTPUT_DIR/../venv", "sleep 60"]),
    category="demo",
)

shell_action(name="fail", cmd="false", category="shallfail", sandbox=NoSandbox())
