import logging

logging.basicConfig(format="%(process)d %(filename)s %(lineno)d: %(message)s")

import argparse
import asyncio
import inspect
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import cpu_count
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Iterable, Protocol

import filelock
import psutil
from result import Err, Ok

from .access_recorder import load_action_deps
from .action import Action, ActionLabel, label_from_line
from .action_invoker import ActionInvoker, DummyInvoker
from .file_layout import (
    FUSEBUILD_INVOCATION_DIR,
    action_dir,
    status_lock_file,
    stderr_file,
    stdout_file,
    subbuild_failed_file,
)
from .graph import sort_graph
from .libfusebuild import (
    BasicExecuter,
    ExecuterBase,
    Status,
    StatusEnum,
    get_action_executer,
    load_actions,
)
from .logger import FUSEBUILD_LOG_LEVEL, getLogger
from .utils import run_action_cmd_env

logger = getLogger(__name__)


def copy_file_to_stderr(name: Path) -> None:
    with name.open("rb") as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            sys.stderr.buffer.write(data)


def print_output(label: ActionLabel) -> None:
    stderr_out = stderr_file(label)
    if stderr_out.exists():
        print(f"Stderr of {label}:", file=sys.stderr)
        copy_file_to_stderr(stderr_out)
    stdout_out = stdout_file(label)
    if stdout_out.exists():
        print(f"Stdout of {label}:", file=sys.stderr)
        copy_file_to_stderr(stdout_out)


def print_failure(label: ActionLabel, seen: set[ActionLabel]) -> None:
    if label in seen:
        print(f"Deadlock detected at {label}", file=sys.stderr)
        print_output(label)
        return
    action = ExecuterBase(label)
    try:
        with action.get_status_lock_file().acquire(blocking=False):
            status = action._read_status()
            if status is None:
                # Must have run now, but on failures
                print(f"{label} isn't defined")
                return
            if status.status != StatusEnum.DONE:
                print(
                    f"Some other is building {label} (pid={status.running_pid}) such that failure can't be printed"
                )
                return
            subbuild_failed_path = subbuild_failed_file(label)
            if subbuild_failed_path.exists():
                seen2 = seen.union({label})
                with subbuild_failed_path.open("r") as f:
                    for line in set(f.readlines()):
                        failed_label = label_from_line(line)
                        print(f"{label} failed due to {failed_label}", file=sys.stderr)
                        print_failure(failed_label, seen2)
            else:
                print_output(label)
    except filelock.Timeout as to:
        subprocess.run(["fuser", str(status_lock_file(label))])
        subprocess.run(["ps", "auxfwwww"])
        print(
            f"Can't get lock in {label}, and print further information.",
            file=sys.stderr,
        )
        print(f"This is usually due to a deadlock or a another fusebuild running.")
        print_output(label)


class BuildActionStatus(Enum):
    IN_QUEUE = 0
    RUNNING = 1
    SUCCESSFULL = 2
    FAILED = 3


@dataclass
class BuildAction:
    label: ActionLabel
    needed: bool
    status: BuildActionStatus = BuildActionStatus.IN_QUEUE
    deps: set[ActionLabel] = field(default_factory=set)
    done_actions: set[Callable[[], None]] = field(default_factory=set)


class ActionExecuter(Protocol):
    def schedule_action(self, action: BuildAction) -> None:
        """Execute action if it matches category"""
        ...


class ActionExecuterImpl(ActionExecuter):
    actions: dict[ActionLabel, BuildAction]
    running: dict[asyncio.Task[Any], tuple[asyncio.subprocess.Process, BuildAction]]
    waiting: list[BuildAction]
    need_resort: bool
    failures: list[BuildAction]
    max_running: int
    invoker: ActionInvoker

    def __init__(self, max_running: int) -> None:
        self.actions = {}
        self.running = {}
        self.waiting = []
        self.need_resort = False
        self.failures = []
        self.max_running = max_running
        self.invoker = DummyInvoker()

    def schedule_action(self, action: BuildAction) -> None:
        self.need_resort = True
        logger.debug(f"Scheduling {action.label}")
        if action.label in self.actions:
            old_action = self.actions[action.label]
            old_action.deps.update(action.deps)
            old_action.done_actions.update(action.done_actions)
            old_action.needed = old_action.needed or action.needed
        else:
            self.actions[action.label] = action
            self.waiting.append(action)
            for d in load_action_deps(action.label):
                logger.debug(f"Adding dependency {d} for {action.label}")
                action.deps.add(d)
                if d not in self.actions:
                    self.schedule_action(BuildAction(d, needed=False))
            if action.label.name != "FUSEBUILD.py":
                bf_label = ActionLabel(action.label.path, "FUSEBUILD.py")
                logger.debug(f"Adding {bf_label} for {action.label}")

                action.deps.add(bf_label)
                if bf_label not in self.actions:
                    self.schedule_action(BuildAction(bf_label, needed=True))

    def sort_waiting(self) -> None:
        graph: dict[ActionLabel, list[ActionLabel]] = {}

        def members() -> Iterable[BuildAction]:
            for a in self.waiting:
                assert a.status == BuildActionStatus.IN_QUEUE
                yield a
            for ta in self.running.values():
                assert ta[1].status == BuildActionStatus.RUNNING
                yield ta[1]

        for a in members():
            graph[a.label] = [
                d
                for d in a.deps
                if self.actions[d].status
                in [BuildActionStatus.IN_QUEUE, BuildActionStatus.RUNNING]
            ]
        assert len(graph) == len(self.waiting) + len(self.running)
        sorted_actions = sort_graph(graph)
        assert len(sorted_actions) == len(graph)
        waiting_new = [
            self.actions[al]
            for al in sorted_actions
            if self.actions[al].status == BuildActionStatus.IN_QUEUE
        ]
        waiting_new_labels = set([a.label for a in waiting_new])
        waiting_old_labels = set([a.label for a in self.waiting])
        if waiting_new_labels != waiting_old_labels:
            logger.error(
                f"Missing in new labels: {waiting_old_labels - waiting_new_labels}"
            )
            logger.error(
                f"New in new labels: {waiting_new_labels - waiting_old_labels}"
            )
            assert False
        self.waiting = waiting_new

    async def start_running(self, action: BuildAction) -> None:
        logger.debug(f"Starting {action.label}")
        print(f"{action.label}..")
        assert action.status == BuildActionStatus.IN_QUEUE
        action.status = BuildActionStatus.RUNNING
        cmd, env = run_action_cmd_env(
            action.label.path, action.label.name, self.invoker
        )
        proc = await asyncio.create_subprocess_exec(*cmd, env=env)
        logger.debug(f"Running {cmd} with env {env} in {proc.pid=}")
        task = asyncio.create_task(proc.wait())
        self.running[task] = (proc, action)
        action.status = BuildActionStatus.RUNNING

    def action_done(self, task: asyncio.Task[Any]) -> None:
        assert task in self.running
        process, action = self.running.pop(task)
        logger.debug(f"{action.label}: {process.returncode}")
        if process.returncode == 0:
            action.status = BuildActionStatus.SUCCESSFULL
            print(f"{action.label} ... Ok")
            for done_cb in action.done_actions:
                done_cb()
        else:
            print(f"{action.label} ... Failed")
            action.status = BuildActionStatus.FAILED
            if action.needed:
                self.failures.append(action)

    async def run(self) -> int:
        while True:
            if len(self.failures) > 0:
                failure = self.failures[0]
                print_failure(failure.label, set([]))
                return 1

            pre_sort = len(self.waiting)
            if self.need_resort:
                self.sort_waiting()
            assert pre_sort == len(self.waiting)

            while len(self.waiting) > 0 and len(self.running) < self.max_running:
                await self.start_running(self.waiting[0])
                self.waiting = self.waiting[1:]

            if len(self.running) == 0 and len(self.waiting) == 0:
                return 0

            logger.debug(f"Waiting for one of {len(self.running)} actions.")
            done, pending = await asyncio.wait(
                [t for t in self.running.keys()],
                timeout=10,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if len(done) == 0:
                logger.debug("timeout")
                continue

            for d in done:
                self.action_done(d)


@dataclass(frozen=True)
class ScheduleAll:
    executer: ActionExecuter
    bf_label: ActionLabel
    categories: frozenset[str]

    def __call__(self) -> None:
        actions = load_actions(self.bf_label.path)
        print(f"Loading all actions {self.bf_label}")
        for label, action in actions.items():
            if action.category in self.categories:
                self.executer.schedule_action(BuildAction(label, needed=True))


def main_inner(args: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-j", "--parallel", type=int, default=0)
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

    logger.info(f"Using {os.environ[FUSEBUILD_INVOCATION_DIR]} as invocation dir.")

    categories = frozenset(arg.category.split(","))

    invoker = DummyInvoker()
    max_running = arg.parallel
    if max_running <= 0:
        max_running = cpu_count()
    executer = ActionExecuterImpl(max_running=max_running)
    for ti in arg.target:
        t: Path = ti.absolute()
        logger.debug(f"Processing {ti} at {os.getcwd()=}: {t=}")
        if t.exists():
            if not t.is_dir():
                print(f"{t} is an file, not an action", file=sys.stderr)
                sys.exit(1)
            build_files = t.glob("**/FUSEBUILD.py")
            for bf in build_files:
                bf_label = ActionLabel(bf.parent, bf.name)
                action = BuildAction(
                    bf_label,
                    needed=True,
                    done_actions={ScheduleAll(executer, bf_label, categories)},
                )
                executer.schedule_action(action)

        else:
            while True:
                name = t.name
                t_next = t.parent
                if t_next == t:
                    print(f"Can't find build file matching {ti}", file=sys.stderr)
                    return 1
                t = t_next
                logger.debug(f"{t=} {name=}")
                build_file = t / "FUSEBUILD.py"
                if build_file.exists():
                    executer.schedule_action(
                        BuildAction(ActionLabel(t, name), needed=True)
                    )
                    break
    return asyncio.run(executer.run())


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
            logger.info(f"Result of main: {ret}")
            return ret
        finally:
            signal_handler(tmp_dir, signal.SIGHUP)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
