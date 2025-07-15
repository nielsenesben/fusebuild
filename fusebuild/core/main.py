import logging

logging.basicConfig(format="%(process)d %(filename)s %(lineno)d: %(message)s")

import sys
import time
import inspect
from pathlib import Path
from .libfusebuild import (
    BasicAction,
    FUSEBUILD_INVOCATION_DIR,
    all_actions,
    check_build_file,
)
from .action import Action, ActionLabel
from .libfusebuild import (
    get_rule_action,
    find_all_actions,
    run_action,
    get_action_from_path,
    load_action_deps,
)
from result import Ok, Err
import tempfile
import os
import subprocess
import psutil
import signal
import time
import argparse
from .graph import sort_graph

from .logger import getLogger, FUSEBUILD_LOG_LEVEL

logger = getLogger(__name__)


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

    targets: dict[ActionLabel, Action] = {}
    for ti in arg.target:
        t: Path = ti.absolute()
        logger.debug(f"Processing {ti} at {os.getcwd()=}: {t=}")
        if t.exists():
            if not t.is_dir():
                print(f"{t} is an file, not an action", file=sys.stderr)
                sys.exit(1)
            ret = find_all_actions(t)
            match ret:
                case Err(failed):
                    print(f"Failed to load {failed}.", file=sys.stderr)
                    sys.exit(1)
                case Ok(actions):
                    targets.update(actions)
        else:
            label_action = get_action_from_path(t)
            if label_action is None:
                print(f"Can't find action for {t}", file=sys.stderr)
                return 1
            targets[label_action[0]] = label_action[1]

    graph: dict[ActionLabel, list[ActionLabel]] = {}

    def helper(l: ActionLabel):
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
        rule_action = get_rule_action(label[0], label[1])
        if rule_action is None:
            if label in targets:
                print(f"Can't find action for {label}", file=sys.stderr)
                return 1
            else:
                # Just ignore it, everything might work anyway
                logger.info(f"Old dependency to now missing action {label}")
                continue

        print(f"{str(label[0])}/{label[1]}....", end="")
        needs_rebuild = rule_action.needs_rebuild("from main loop")
        rule_action.release_lock()
        if not needs_rebuild:
            print(".. Unchanged")
        else:
            res = run_action(label[0], label[1])
            if res != 0:
                print(".. failed")
                if label in targets:
                    return 1
                # otherwise just continue - the _targets_ _might_ be ok
            else:
                print("..Ok")

    return 0


def kill_process(process: psutil.Process, signal: int, tmp_dir: str):
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


def kill_recursive(process: psutil.Process, signal: int, tmp_dir: str):
    for c in process.children():
        kill_recursive(c, signal, tmp_dir)
        kill_process(c, signal, tmp_dir)


def status(p: psutil.Process) -> str:
    try:
        return p.status()
    except psutil.NoSuchProcess:
        return "gone"


def signal_handler(tmp_dir: str, signumber: int, frame=None) -> None:
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
