from fusebuild import shell_action, get_action, NoSandbox, BwrapSandbox
from fusebuild.python import pyc_mappings, pyc_actions

pyc_actions()

shell_action(
    name="run_test_example1_nosandbox",
    cmd="\n".join(
        [
            "export FUSEBUILD_CACHE_DIR=$(mktemp -d)",
            ". $OUTPUT_DIR/../../venv/bin/activate",
            "python test_example1.py",
        ]
    ),
    category="test",
    sandbox=NoSandbox(),
    mappings=pyc_mappings(),
)

shell_action(
    name="test_graph",
    cmd="\n".join(
        [
            ". $OUTPUT_DIR/../../venv/bin/activate",
            "export FUSEBUILD_CACHE_DIR=$OUTPUT_DIR/fusebuild_cache",
            "ls -altr",
            "python -B test_graph.py",
            "ls -altr",
        ]
    ),
    category="test",
    mappings=pyc_mappings(),
)

shell_action(
    name="test_action",
    cmd="\n".join(
        [
            ". $OUTPUT_DIR/../../venv/bin/activate",
            "export FUSEBUILD_CACHE_DIR=$OUTPUT_DIR/fusebuild_cache",
            "python test_action.py",
        ]
    ),
    category="test",
    mappings=pyc_mappings(),
)
shell_action(
    name="run_test_example1_bwrap",
    cmd="\n".join(
        [
            ". $OUTPUT_DIR/../../venv/bin/activate",
            "export FUSEBUILD_CACHE_DIR=$OUTPUT_DIR/fusebuild_cache",
            "python test_example1.py",
        ]
    ),
    category="test",
    sandbox=BwrapSandbox(run_as_root=True),
    mappings=pyc_mappings(),
)


shell_action(
    name="run_test_example2_bwrap",
    cmd="\n".join(
        [
            ". $OUTPUT_DIR/../../venv/bin/activate",
            "export FUSEBUILD_CACHE_DIR=$OUTPUT_DIR/fusebuild_cache",
            "python test_example2.py",
        ]
    ),
    category="test",
    sandbox=BwrapSandbox(run_as_root=True),
    mappings=pyc_mappings(),
)

provider = get_action("./test_providers/", "hasprovider").providers["testprovider"]

shell_action(
    name="testproviderotherdirectory",
    cmd=f"test hallo = $(cat {provider.get_some_file()})",  # type: ignore
)
