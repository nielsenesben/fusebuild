from pathlib import Path
from unittest.mock import MagicMock

from absl.testing.absltest import TestCase, main

from fusebuild.core.action import Action
from fusebuild.core.action_invoker import ActionInvoker
from fusebuild.core.libfusebuild import get_action


class TestGetAction(TestCase):
    def common_test_fusebuild_py_action(self, path_in: Path | str) -> None:
        if isinstance(path_in, str):
            path = Path(path_in)
        else:
            path = path_in

        invoker = MagicMock(ActionInvoker)
        action = get_action(path, "FUSEBUILD.py", invoker)
        self.assertIsInstance(action, Action)
        assert isinstance(action, Action)
        self.assertEqual(action.category, "")
        self.assertEqual(
            action.cmd,
            [
                "python",
                "-m",
                "fusebuild.core.load_build_file",
                str(path.resolve() / "FUSEBUILD.py"),
            ],
        )

    def test_fusebuild_existing_dir(self) -> None:
        """Tests that we get FUSEBUILD.py action on existing FUSEBUILD.py file"""
        self.common_test_fusebuild_py_action(Path(__file__).parent)

    def test_fusebuild_py_nonexisting_dir(self) -> None:
        """Tests that we can always hit the special branch of FUSEBUIlD.py
        even though the directory doesn't exists"""
        self.common_test_fusebuild_py_action("nonexistingdir")


if __name__ == "__main__":
    main()
