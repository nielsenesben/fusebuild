import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

import fusebuild.core.logger as logger_module

from .action_invoker import ActionInvoker

logger = logger_module.getLogger(__name__)


def os_environ() -> dict[str, str]:
    return {k: os.environ[k] for k in os.environ}


def check_pid(pid: int) -> bool:
    """Check For the existence of a unix pid."""
    psutil.process_iter.cache_clear()
    return psutil.pid_exists(pid)


def kill_subprocess(process: subprocess.Popen[bytes] | psutil.Popen) -> None:
    pid = process.pid

    if not check_pid(pid):
        return

    logger.debug(f"Trying to terminate group {pid=}")
    try:
        os.kill(pid, signal.SIGINT)
    except BaseException as e:
        logger.warning(f"Failed to send sigterm to {pid=}: {type(e)=} {e}")
        for p in psutil.process_iter():
            logger.debug(f"   {p}")

    logger.debug(f"Trying to terminate group {pid=}")
    try:
        os.killpg(pid, signal.SIGTERM)
    except BaseException as e:
        logger.warning(f"Failed to send sigterm to {pid=}: {type(e)=} {e}")
        for p in psutil.process_iter():
            logger.debug(f"   {p}")

    start = time.time()
    while check_pid(pid):
        time.sleep(0.1)
        if time.time() - start >= 1.0:
            break

    if not check_pid(pid):
        return

    logger.debug(f"Trying to kill process group {pid=}")
    try:
        os.killpg(pid, signal.SIGKILL)
    except BaseException as e:
        logger.warning(f"Failed to send sigkill to {pid=}:  {type(e)=}  {e}")


def run_action_cmd_env(
    directory: Path, target: str, invoker: ActionInvoker
) -> tuple[list[str], Any]:
    return [
        "python3",
        str(Path(__file__).parent / "runtarget.py"),
        str(directory.absolute() / "FUSEBUILD.py"),
        target,
    ] + invoker.runtarget_args(), os.environ


def run_action(directory: Path, target: str, invoker: ActionInvoker) -> int:
    logger.debug(f"Run action {directory} / {target}")
    process = None
    try:
        cmd, env = run_action_cmd_env(directory, target, invoker)
        process = psutil.Popen(cmd, env=env)
        res = process.wait()
    except:
        logger.error(f"Something went wrong when invoking {directory} / {target}")
        if process is not None:
            kill_subprocess(process)
        raise
    return res
