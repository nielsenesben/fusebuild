import logging

logging.basicConfig(format="%(process)d %(filename)s %(lineno)d: %(message)s")

import argparse
import inspect
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import TracebackType
from typing import Any

import filelock
import psutil
from result import Err, Ok

from .access_recorder import load_action_deps
from .action import Action, ActionLabel, label_from_line
from .action_invoker import DummyInvoker
from .file_layout import action_dir
from .graph import sort_graph
from .libfusebuild import (
    FUSEBUILD_INVOCATION_DIR,
    ActionBase,
    BasicAction,
    StatusEnum,
    all_actions,
    check_build_file,
    find_all_actions,
    get_action_from_path,
    get_rule_action,
    run_action,
)
from .logger import FUSEBUILD_LOG_LEVEL, getLogger

logger = getLogger(__name__)


def copy_file_to_stderr(name: Path) -> None:
    with name.open("rb") as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            sys.stderr.buffer.write(data)


def print_output(label: ActionLabel) -> None:
    stderr_out = action_dir(label) / "stderr"
    if stderr_out.exists():
        print(f"Stderr of {label}:", file=sys.stderr)
        copy_file_to_stderr(stderr_out)
    stdout_out = action_dir(label) / "stdout"
    if stdout_out.exists():
        print(f"Stdout of {label}:", file=sys.stderr)
        copy_file_to_stderr(stdout_out)


def print_failure(label: ActionLabel, seen: set[ActionLabel]) -> None:
    if label in seen:
        print(f"Deadlock detected at {label}", file=sys.stderr)
        print_output(label)
        return
    action = ActionBase(label)
    try:
        with action.get_status_lock_file().acquire(blocking=False):
            status = action._read_status()
            assert status is not None  # Must have run now, so status can't be None
            if status.status != StatusEnum.DONE:
                print(
                    f"Some other is building {label} (pid={status.running_pid}) such that failure can't be printed"
                )
                return
            subbuild_failed_file = action_dir(label) / "subbuild_failed"
            if subbuild_failed_file.exists():
                seen2 = seen.union({label})
                with subbuild_failed_file.open("r") as f:
                    for line in set(f.readlines()):
                        failed_label = label_from_line(line)
                        print(f"{label} failed due to {failed_label}", file=sys.stderr)
                        print_failure(failed_label, seen2)
            else:
                print_output(label)
    except filelock.Timeout as to:
        subprocess.run(["fuser", str(action.storage / "status.lck")])
        subprocess.run(["ps", "auxfwwww"])
        print(
            f"Can't get lock in {label}, and print further information.",
            file=sys.stderr,
        )
        print(f"This is usually due to a deadlock or a another fusebuild running.")
        print_output(label)


def main_inner(args: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="count", default=0)
    parser.add_argument("category", type=str)
    parser.add_argument("target", nargs="+", type=Path)
    arg = parser.parse_args(args=args)

    log_level = logging.ERROR - 10 * arg.verbose
    os.environ[FUSEBUILD_LOG_LEVEL] = str(log_level)
    os.putenv(FUSEBUILD_LOG_LEVEL, str(log_level))
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(log_level)
        logger.info(f"{logger.name} {log_level=}")
    logger.info(f"{arg.verbose=} {logging.getLevelName(log_level)}")

    categories = arg.category.split(",")

    invoker = DummyInvoker()
    targets: dict[ActionLabel, Action] = {}
    for ti in arg.target:
        t: Path = ti.absolute()
        logger.debug(f"Processing {ti} at {os.getcwd()=}: {t=}")
        if t.exists():
            if not t.is_dir():
                print(f"{t} is an file, not an action", file=sys.stderr)
                sys.exit(1)
            ret = find_all_actions(t, invoker)
            match ret:
                case Err(failed):
                    print(f"Failed to load {failed}.", file=sys.stderr)
                    sys.exit(1)
                case Ok(actions):
                    targets.update(actions)
        else:
            label_action = get_action_from_path(t, invoker)
            if label_action is None:
                print(f"Can't find action for {t}", file=sys.stderr)
                return 1
            targets[label_action[0]] = label_action[1]

    graph: dict[ActionLabel, list[ActionLabel]] = {}

    def helper(l: ActionLabel) -> None:
        deps = load_action_deps(l)
        graph[l] = deps
        logger.debug(f"Graph {l}: {deps}")
        for d in deps:
            if d not in graph:
                helper(d)

    for label, action in targets.items():
        if action.category in categories:
            if label not in graph:
                helper(label)

    logger.debug(f"{graph=}")
    sorted_actions = sort_graph(graph)
    logger.info(f"{sorted_actions=}")
    for label in sorted_actions:
        rule_action = get_rule_action(label.path, label.name, invoker)
        if rule_action is None:
            if label in targets:
                print(f"Can't find action for {label}", file=sys.stderr)
                return 1
            else:
                # Just ignore it, everything might work anyway
                logger.info(f"Old dependency to now missing action {label}")
                continue

        print(f"{label}....", end="")
        needs_rebuild_result = rule_action.needs_rebuild("from main loop")
        rule_action.release_lock()
        match needs_rebuild_result:
            case Err(b):
                logger.warning(
                    f"Got error when calling need_rebuild on {rule_action.label}"
                )
                sys.exit(1)
            case Ok(needs_rebuild):
                if not needs_rebuild:
                    print(".. Unchanged")
                    continue
            case _:
                assert False

        res = run_action(label.path, label.name, invoker)
        if res != 0:
            print(".. failed")
            if label in targets:
                print_failure(label, set([]))
                return 1
            # otherwise just continue - the _targets_ _might_ be ok
        else:
            print("..Ok")

    return 0


def kill_process(process: psutil.Process, signal: int, tmp_dir: str) -> None:
    logger.info(f"Killing {process.pid} with {signal=}")
    try:
        env = process.environ()

        if FUSEBUILD_INVOCATION_DIR in env and env[FUSEBUILD_INVOCATION_DIR] == tmp_dir:
            logger.info(f"Killing {process.pid} with {FUSEBUILD_INVOCATION_DIR}")
        else:
            logger.info(f"Killing {process.pid} with {FUSEBUILD_INVOCATION_DIR}")
    except psutil.AccessDenied as e:
        logger.error(f"No access to {process.pid=} {e}")
    except psutil.NoSuchProcess as e:
        logger.error(f"Process gone {process.pid=} {e}")

    try:
        os.kill(process.pid, signal)
    except ProcessLookupError as e:
        logger.debug(f"Process {process.pid} already gone")


def kill_recursive(process: psutil.Process, signal: int, tmp_dir: str) -> None:
    for c in process.children():
        kill_recursive(c, signal, tmp_dir)
        kill_process(c, signal, tmp_dir)


def status(p: psutil.Process) -> str:
    try:
        return p.status()
    except psutil.NoSuchProcess:
        return "gone"


def signal_handler(
    tmp_dir: str, signumber: int, frame: Any = None
) -> None:  # TODO: type of frame
    logger.info(f"Got signal {signumber}")
    children = psutil.Process().children(recursive=True)
    for c in children:
        kill_process(c, signumber, tmp_dir)
    start = time.monotonic()
    while True:
        for p in children:
            logger.debug(f"{p.pid}: {status(p)}")
        remaining = [
            p
            for p in children
            if status(p)
            not in [psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE, "terminated", "gone"]
        ]
        if len(remaining) == 0:
            break
        logger.info(f"{remaining=}")
        if time.monotonic() - start > 2:
            for c in remaining:
                kill_process(c, signal.SIGKILL, tmp_dir)


def main(args: list[str]) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.environ[FUSEBUILD_INVOCATION_DIR] = tmp_dir
        handler = lambda signum, frame: signal_handler(tmp_dir, signum, frame)
        signal.signal(signal.SIGHUP, handler)
        signal.signal(signal.SIGINT, handler)
        try:
            ret = main_inner(args)
            return ret
        finally:
            signal_handler(tmp_dir, signal.SIGHUP)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
