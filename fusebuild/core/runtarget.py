import importlib.util
import os
import sys
import time
import uuid
from pathlib import Path

from fusebuild.core.action import ActionLabel
from fusebuild.core.action_invoker import ActionInvoker, DummyInvoker
from fusebuild.core.libfusebuild import ExecuterBase, Status, get_action_executer
from fusebuild.core.logger import getLogger, setLoggerPrefix

logger = getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


def _runtarget(buildfile: Path, target: str, invoker: ActionInvoker) -> int | None:
    # os.setpgrp()
    logger.debug(f"Runtarget {buildfile=} {target=} {invoker=}")
    assert buildfile.is_file()
    executer = get_action_executer(buildfile.parent, target, invoker)
    if executer is None:
        return -1
    return_code = executer.run_if_needed(
        invoker, f"building {[str(a) for a in invoker.runtarget_args()]}"
    )
    return return_code


if __name__ == "__main__":
    invoker: ActionInvoker
    if len(sys.argv) > 3:
        invoker = ExecuterBase(ActionLabel(Path(sys.argv[3]), sys.argv[4]))
    else:
        invoker = DummyInvoker()
    setLoggerPrefix(f"runtarget {sys.argv[2]}")
    result = _runtarget(Path(sys.argv[1]), sys.argv[2], invoker)
    logger.debug(f"Result of {sys.argv}: {result}")
    sys.exit(result)
