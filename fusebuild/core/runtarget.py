import importlib.util
import os
import sys
import time
import uuid
from pathlib import Path

from fusebuild.core.action_invoker import ActionInvoker, DummyInvoker
from fusebuild.core.libfusebuild import ActionBase, Status, get_rule_action
from fusebuild.core.logger import getLogger

logger = getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


def _check_for_deadlock(
    status: Status, target: tuple[Path, str], try_to_lock_all
) -> bool:
    return False


def _runtarget(buildfile: Path, target: str, invoker: ActionInvoker) -> int | None:
    # os.setpgrp()
    logger.debug(f"Runtarget {buildfile=} {target=}")
    assert buildfile.is_file()
    label = (buildfile.parent, target)
    action = get_rule_action(buildfile.parent, target, invoker)
    if action is None:
        return -1
    return_code = action.run_if_needed(invoker, "?")
    return return_code


if __name__ == "__main__":
    invoker: ActionInvoker
    if len(sys.argv) > 3:
        invoker = ActionBase((Path(sys.argv[3]), sys.argv[4]))
    else:
        invoker = DummyInvoker()
    sys.exit(_runtarget(Path(sys.argv[1]), sys.argv[2], invoker))
