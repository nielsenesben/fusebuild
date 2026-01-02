import logging
import os
import shutil
import subprocess
import tempfile
from inspect import getframeinfo, stack
from pathlib import Path

from absl.testing.absltest import TestCase, main

from fusebuild import output_folder_root
from fusebuild.core.logger import getLogger

logger = getLogger(__name__)
verboses = 0
logger.setLevel(logging.ERROR - 10 * verboses)
verbose = ["-v"] * verboses


def run(cmd: list[str]) -> int:
    caller = getframeinfo(stack()[1][0])
    p = subprocess.Popen(cmd)
    logger.info(f"{caller.filename}:{caller.lineno} {p.pid=}")
    return p.wait()


class TestExample2(TestCase):
    tempdir = tempfile.TemporaryDirectory()

    def setUp(self) -> None:
        ## First copy the test project to a tmp working dir
        self.workdir = Path(self.tempdir.name) / "example2"
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        logger.info(f"Copying to {self.workdir}")
        ret = shutil.copytree(Path(__file__).parent / "test_example2", self.workdir)
        self.assertEqual(ret, self.workdir)

    def test_depend_on_action_in_parent(self) -> None:
        output_file = (
            Path(str(output_folder_root) + str(self.workdir))
            / "subdir"
            / "dependoneup"
            / "output2.txt"
        )

        build_cmd = (
            [
                "python3",
                "-m",
                "fusebuild",
            ]
            + verbose
            + [
                "build",
                str(self.workdir) + "/subdir/dependoneup",
            ]
        )

        ret = run(build_cmd)
        self.assertEqual(ret, 0)
        self.assertEqual(output_file.read_text(), "something\n")
        old_stat = os.stat(output_file)

        ret = run(build_cmd)
        self.assertEqual(ret, 0)
        new_stat = os.stat(output_file)
        self.assertEqual(new_stat, old_stat)  # Not remade when no changes

        with (self.workdir / "FUSEBUILD.py").open("w") as f:
            f.write(
                """from fusebuild import *

shell_action(name="someaction",
	     cmd = "echo newtext > $OUTPUT_DIR/output1.txt",
             category="dontbuilddirectly", tmp=None)
"""
            )

        ret = run(build_cmd)
        self.assertEqual(ret, 0)
        self.assertEqual(output_file.read_text(), "newtext\n")
        old_stat = os.stat(output_file)

        ret = run(build_cmd)
        self.assertEqual(ret, 0)
        new_stat = os.stat(output_file)
        self.assertEqual(new_stat, old_stat)

        with (self.workdir / "FUSEBUILD.py").open("w") as f:
            f.write(
                """from fusebuild import *

shell_action(name="someaction",
	     cmd = "false",
             category="dontbuilddirectly", tmp=None)
"""
            )

        build_cmd = (
            [
                "python3",
                "-m",
                "fusebuild",
            ]
            + verbose
            + [
                "build",
                str(self.workdir) + "/subdir/dependoneup",
            ]
        )

        ret = run(build_cmd)
        self.assertEqual(ret, 1)

        # Test that failed FUSEBUILD.py gives a failed build
        with (self.workdir / "FUSEBUILD.py").open("w") as f:
            f.write(
                """from fusebuild import *

shell_action(name="someaction",
	     cmd = "false",
             category="dontbuilddirectly", tmp=Nonesense)
"""
            )

        build_cmd = (
            [
                "python3",
                "-m",
                "fusebuild",
            ]
            + verbose
            + [
                "build",
                str(self.workdir) + "/subdir/dependoneup",
            ]
        )

        ret = run(build_cmd)
        self.assertEqual(ret, 1)


if __name__ == "__main__":
    main()
