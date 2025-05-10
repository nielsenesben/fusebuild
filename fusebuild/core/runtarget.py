import sys
import os
import importlib.util
import time
from pathlib import Path
from libfusebuild import get_rule_action, Status
import uuid
from logger import getLogger

logger = getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


def _check_for_deadlock(
    status: Status, target: tuple[Path, str], try_to_lock_all
) -> bool:
    return False


def runtarget(buildfile: Path, target: str) -> int | None:
    # os.setpgrp()
    logger.debug(f"Runtarget {buildfile=} {target=}")
    count = 0
    deadlock_count = 0
    while True:
        action = get_rule_action(buildfile, target)
        if action is None:
            return -1

        res, status = action.require_for_build()
        if res:
            break
        count += 1
        if count > 2:
            logger.info(
                f"Waiting for {(Path(buildfile.parent), target)} with status {status}"
            )
        if _check_for_deadlock(status, (buildfile.parent, target), deadlock_count):
            deadlock_count += 1
        else:
            deadlock_count = 0

        time.sleep(1)

    action = get_rule_action(buildfile.parent, target)
    if action is None:
        return -1
    return_code = action.run_if_needed("?")
    return return_code


if __name__ == "__main__":
    sys.exit(runtarget(Path(sys.argv[1]), sys.argv[2]))
