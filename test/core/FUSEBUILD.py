from fusebuild import BwrapSandbox, NoSandbox, get_action, shell_action
from fusebuild.python import mypy_actions, pyc_actions, pyc_mappings

pyc_actions()
mypy_actions()

shell_action(
    name="test_libfusebuild",
    cmd="\n".join(
        [
            ". $OUTPUT_DIR/../../../venv/bin/activate",
            "python -B test_libfusebuild.py",
        ]
    ),
    category="test",
    mappings=pyc_mappings(),
)

shell_action(
    name="test_graph",
    cmd="\n".join(
        [
            ". $OUTPUT_DIR/../../../venv/bin/activate",
            "export FUSEBUILD_CACHE_DIR=$OUTPUT_DIR/fusebuild_cache",
            "ls -altr",
            "python -B test_graph.py",
            "ls -altr",
        ]
    ),
    category="test",
    mappings=pyc_mappings(),
)
