import asyncio
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fusebuild.core.action import ActionLabel
from fusebuild.core.action_invoker import DummyInvoker
from fusebuild.core.errorcodes import ErrorCode
from fusebuild.core.file_layout import FUSEBUILD_INVOCATION_DIR
from fusebuild.core.main import ActionExecuterImpl, BuildAction, BuildActionStatus
from fusebuild.core.utils import run_action_cmd_env


def checkInvariances(executer: ActionExecuterImpl) -> None:
    def in_started(action: BuildAction) -> bool:
        for task, (p, a) in executer.started.items():
            if action == a:
                return True
        return False

    for label, action in executer.actions.items():
        assert label == action.label
        match action.status:
            case BuildActionStatus.WAITING:
                assert label in executer.waiting
                assert label not in executer.runable
                assert label not in executer.blocked
                assert label not in executer.blocked_runable
                assert action not in executer.failures
                assert not in_started(action)
            case BuildActionStatus.RUNABLE:
                assert label not in executer.waiting
                assert label in executer.runable
                assert label not in executer.blocked
                assert label not in executer.blocked_runable
                assert action not in executer.failures
                assert not in_started(action)
            case BuildActionStatus.RUNNING:
                assert label not in executer.waiting
                assert label not in executer.runable
                assert label not in executer.blocked
                assert label not in executer.blocked_runable
                assert action not in executer.failures
                assert in_started(action)
            case BuildActionStatus.SUCCESSFULL:
                print(f"{label=} {action.label=} {action.status=}")
                assert label not in executer.waiting
                assert label not in executer.runable
                assert label not in executer.blocked
                assert label not in executer.blocked_runable
                assert action not in executer.failures
                assert not in_started(action)
            case _:
                assert False


async def background_worker(
    executer: ActionExecuterImpl, stop_event: asyncio.Event
) -> None:
    """Runs continuously until told to stop."""
    while not stop_event.is_set():
        checkInvariances(executer)
        await asyncio.sleep(0.0)  # simulate periodic work


class TestActionExecuter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.invovation_dir = Path(tempfile.mkdtemp())
        os.environ[FUSEBUILD_INVOCATION_DIR] = str(self.invovation_dir)

        self.executer = ActionExecuterImpl(
            max_running=1, invocation_dir=self.invovation_dir
        )
        self.fusebuild_py_label = ActionLabel(Path("/"), "FUSEBUILD.py")
        self.stop_event = asyncio.Event()
        self.bg_task = asyncio.create_task(
            background_worker(self.executer, self.stop_event)
        )

    async def asyncTearDown(self) -> None:
        self.stop_event.set()
        await self.bg_task
        shutil.rmtree(self.invovation_dir)

    @patch("fusebuild.core.main.run_action_cmd_env")
    async def test_simple_action_runs(self, mock_run_action_cmd_env: Any) -> None:
        mock_run_action_cmd_env.return_value = (["true"], os.environ)
        action_label = ActionLabel(Path("/"), "action")

        action = BuildAction(action_label, needed=True)

        self.executer.schedule_action(action)
        self.assertIn(action_label, self.executer.actions)
        self.assertEqual(self.executer.actions[action_label], action)
        # Test that the corresponding FUSEBUILD.py action is added
        self.assertIn(self.fusebuild_py_label, self.executer.actions)
        self.assertEqual(
            self.executer.actions[self.fusebuild_py_label].status,
            BuildActionStatus.RUNABLE,
        )
        self.assertEqual(action.status, BuildActionStatus.WAITING)

        res = await self.executer.run()
        self.assertEqual(res, ErrorCode.SUCCESS)
        self.assertEqual(action.status, BuildActionStatus.SUCCESSFULL)


if __name__ == "__main__":
    unittest.main()
