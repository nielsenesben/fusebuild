import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

from absl.testing.absltest import TestCase, main  # type: ignore

from fusebuild import output_folder_root

# Load internal stuff to test internal state
from fusebuild.core.access_recorder import load_action_deps
from fusebuild.core.logger import getLogger

# logging.basicConfig(level=logging.DEBUG)


logger = getLogger(__name__)


class TestCircular(TestCase):
    tempdir = tempfile.TemporaryDirectory()

    def setUp(self):
        ## First copy the test project to a tmp working dir
        self.workdir = Path(self.tempdir.name) / "circular"
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        logger.info(f"Copying to {self.workdir}")
        ret = shutil.copytree(Path(__file__).parent / "test_circular", self.workdir)
        self.assertEqual(ret, self.workdir)

    def test_build_circular(self):
        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "circular",
                str(self.workdir / "A"),
            ]
        )
        self.assertEqual(ret.returncode, 1)

        # Now testing that breaking the circular dependency will fix the build
        (self.workdir / "FUSEBUILD.py").write_text(
            """
from fusebuild import shell_action

shell_action(
    name="A",
    cmd="echo somethong > $OUTPUT_DIR/file.txt",
    category="circular",
    tmp=None,
)

shell_action(
    name="B",
    cmd="cp $OUTPUT_DIR/../A/file.txt $OUTPUT_DIR/file.txt",
    category="circular",
    tmp=None,
)

shell_action(
    name="C",
    cmd="cp $OUTPUT_DIR/../B/file.txt $OUTPUT_DIR/file.txt",
    category="circular",
    tmp=None,
)
"""
        )

        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "circular",
                str(self.workdir / "A"),
            ]
        )
        self.assertEqual(ret.returncode, 0)

        # Now reintroduce the issue
        (self.workdir / "FUSEBUILD.py").unlink()
        ret = shutil.copyfile(
            Path(__file__).parent / "test_circular" / "FUSEBUILD.py",
            self.workdir / "FUSEBUILD.py",
        )
        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "circular",
                str(self.workdir / "A"),
            ]
        )
        self.assertEqual(ret.returncode, 1)

    def test_build_circular_from_out_of_circle(self):
        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "circular",
                str(self.workdir / "D"),
            ]
        )
        self.assertEqual(ret.returncode, 1)

    def test_circular_fusebuild_files(self):
        # Avoid original error
        (self.workdir / "FUSEBUILD.py").write_text("")
        (self.workdir / "subA").mkdir()
        (self.workdir / "subA" / "FUSEBUILD.py").write_text(
            """from fusebuild import shell_action, get_action


shell_action(
    name="A",
    cmd="echo ok A",
    category="circular",
    tmp=None,
)

get_action("../subB", "B")
"""
        )
        (self.workdir / "subB").mkdir()
        (self.workdir / "subB" / "FUSEBUILD.py").write_text(
            """from fusebuild import shell_action, get_action


shell_action(
    name="B",
    cmd="echo ok B",
    category="circular",
    tmp=None,
)

get_action("../subA", "A")
"""
        )
        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "build",
                str(self.workdir),
            ]
        )
        self.assertEqual(ret.returncode, 1)


if __name__ == "__main__":
    main()
