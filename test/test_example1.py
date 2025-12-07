import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

from absl.testing.absltest import TestCase, main  # type: ignore

from fusebuild import ActionLabel, output_folder_root

# Load internal stuff to test internal state
from fusebuild.core.access_recorder import load_action_deps
from fusebuild.core.logger import getLogger

# logging.basicConfig(level=logging.DEBUG)


logger = getLogger(__name__)


def create_tcp_server() -> socket.socket:
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ("localhost", 0)
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    return sock


class TestExample1(TestCase):
    tempdir = tempfile.TemporaryDirectory()

    def setUp(self) -> None:
        ## First copy the test project to a tmp working dir
        self.workdir = Path(self.tempdir.name) / "example1"
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        logger.info(f"Copying to {self.workdir}")
        ret = shutil.copytree(Path(__file__).parent / "test_example1", self.workdir)
        self.assertEqual(ret, self.workdir)

    def test_rebuilding(self) -> None:
        output_file = (
            Path(str(output_folder_root) + str(self.workdir))
            / "copyfile"
            / "file1_copied.txt"
        )

        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "build",
                str(self.workdir) + "/copyfile",
            ]
        )
        self.assertEqual(ret.returncode, 0)
        self.assertEqual(output_file.read_text(), "This is a test\n")
        old_stat = os.stat(output_file)

        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "build",
                str(self.workdir) + "/copyfile",
            ]
        )
        self.assertEqual(ret.returncode, 0)
        new_stat = os.stat(output_file)
        self.assertEqual(new_stat, old_stat)  # Not remade when no changes

        (self.workdir / "file1.txt").write_text("Changed text")
        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "build",
                str(self.workdir) + "/copyfile",
            ]
        )
        self.assertEqual(ret.returncode, 0)
        self.assertEqual(output_file.read_text(), "Changed text")
        old_stat = os.stat(output_file)

        with (self.workdir / "FUSEBUILD.py").open("a") as f:
            f.write(
                """shell_action(
    name="new_action",
    cmd="echo foo"
)
"""
            )
        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "build",
                str(self.workdir) + "/copyfile",
            ]
        )
        self.assertEqual(ret.returncode, 0)
        new_stat = os.stat(output_file)
        self.assertEqual(
            new_stat, old_stat
        )  # Not remade, when only FUSEBUILD.py have changed

    def test_reference_to_non_existant_target(self) -> None:
        self.assertTrue(self.workdir.exists())
        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "build",
                str(self.workdir) + "/access_non_existant_rule",
            ]
        )
        self.assertEqual(ret.returncode, 0)
        deps = load_action_deps(ActionLabel(self.workdir, "access_non_existant_rule"))
        logger.info(f"{deps=}")
        self.assertGreaterEqual(len(deps), 1)
        self.assertTrue(ActionLabel(self.workdir, "nonexistant_rule") in deps)

    def test_failed_action(self) -> None:
        self.assertTrue(self.workdir.exists())
        subprocess.run(["ls", str(self.workdir)])
        ret = subprocess.run(
            [
                "python3",
                "-m",
                "fusebuild",
                "failing",
                str(self.workdir) + "/fail",
            ]
        )
        self.assertEqual(ret.returncode, 1)

    def test_handle_change_to_input_under_build(self) -> None:
        with create_tcp_server() as sock:
            print(f"{sock=}")
            _, port = sock.getsockname()

            print(f"Server is listening on port {port}")
            new_env = os.environ.copy()
            new_env.update({"TEST_PORT": str(port)})
            p = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "fusebuild",
                    "testspecial",
                    str(self.workdir) + "/copyfile_twice",
                ],
                env=new_env,
            )
            print("Waiting for socket to connect")
            client_sock, _ = sock.accept()
            print("Replacing file")
            with (self.workdir / "file1.txt").open("w") as f:
                f.write("Some changed text\n")
                f.flush()
                f.close()
            client_sock.send(b"hello\n")
            client_sock.close()
            print("Wating for action to continue")
            p.wait(timeout=60)
            self.assertEqual(p.returncode, 0)
            output_file1 = (
                Path(str(output_folder_root) + str(self.workdir))
                / "copyfile_twice"
                / "file1_copied1.txt"
            )
            self.assertEqual(output_file1.read_text(), "This is a test\n")
            old_stat1 = os.stat(output_file1)
            output_file2 = (
                Path(str(output_folder_root) + str(self.workdir))
                / "copyfile_twice"
                / "file1_copied2.txt"
            )
            # Don't know if the file have been changed due to caching in fuse
            self.assertTrue(
                output_file2.read_text() in ["This is a test\n", "Some changed text\n"]
            )
            old_stat2 = os.stat(output_file2)

        with create_tcp_server() as sock:
            print(f"{sock=}")
            _, port = sock.getsockname()

            print(f"Server is listening on port {port}")
            new_env = os.environ.copy()
            new_env.update({"TEST_PORT": str(port)})
            p = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "fusebuild",
                    "testspecial",
                    str(self.workdir) + "/copyfile_twice",
                ],
                env=new_env,
            )
            client_sock, _ = sock.accept()
            print(f"Got connection 2: {client_sock=}")
            try:
                client_sock.send("hello 2".encode())
            except Exception as e:
                print(e)
            client_sock.close()
            p.wait(timeout=60)
            self.assertEqual(p.returncode, 0)
            new_stat1 = os.stat(output_file1)
            self.assertNotEqual(new_stat1, old_stat1)  # Remade
            self.assertEqual(output_file1.read_text(), "Some changed text\n")
            new_stat2 = os.stat(output_file2)
            self.assertNotEqual(new_stat2, old_stat2)  # Remade
            self.assertEqual(output_file2.read_text(), "Some changed text\n")

            p = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "fusebuild",
                    "testspecial",
                    str(self.workdir) + "/copyfile_twice",
                ],
                env=new_env,
            )
            p.wait(timeout=60)
            self.assertEqual(p.returncode, 0)
            newnew_stat1 = os.stat(output_file1)
            self.assertEqual(newnew_stat1.st_mtime, new_stat1.st_mtime)  # Not remade
            newnew_stat2 = os.stat(output_file2)
            self.assertEqual(newnew_stat2.st_mtime, new_stat2.st_mtime)  # Not remade


if __name__ == "__main__":
    main()
