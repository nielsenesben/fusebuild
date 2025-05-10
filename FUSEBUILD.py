from glob import glob
from fusebuild import shell_action, action, NoSandbox
from fusebuild.python import pyc_mappings, pyc_actions, mypy_actions

print(f"Loading {__file__}")
action(name="xeyes", cmd=["xeyes"], category="demo")

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
