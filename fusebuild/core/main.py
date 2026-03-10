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
from typing import Any, Awaitable, Callable, Iterable, Protocol

import filelock
import psutil
from result import Err, Ok

from .access_recorder import load_action_deps
from .action import Action, ActionLabel, label_from_line
from .action_invoker import ActionInvoker, DummyInvoker
from .file_layout import (
    FUSEBUILD_INVOCATION_DIR,
    action_dir,
    socket_path,
    status_lock_file,
    stderr_file,
    stdout_file,
    subbuild_failed_file,
)
from .libfusebuild import (
    BasicExecuter,
    ExecuterBase,
    Status,
    StatusEnum,
    get_action_executer,
    load_actions,
)
from .logger import FUSEBUILD_LOG_LEVEL, getLogger
from .utils import run_action_cmd_env, unescape_whitespace

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
                    f"Some other is building {label} (pid={status.running_pid}) such that failure can't be printed reliable"
                )
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
    WAITING = 0
    RUNABLE = 1
    RUNNING = 2
    BLOCKED = 3
    BLOCKED_RUNABLE = 4
    SUCCESSFULL = 5
    FAILED = 6


# Possible transitions:
# WAITING -> RUNABLE when all deps are done
# RUNABLE -> RUNNING when start runnning
# RUNNING -> BLOCKED when unknown deps are found while running
# BLOCKED -> BLOCKED_RUNABLE when blocker have finished
# BLOCKED_RUNABLE -> RUNNING when it is scheduled to continue
# RUNNING -> SUCCESSFULL when done with return code 0
# RUNNING -> FAILED when done with return code non-zero 0


def waiting_status(s: BuildActionStatus) -> bool:
    return s == BuildActionStatus.WAITING or s == BuildActionStatus.BLOCKED


def have_not_started(s: BuildActionStatus) -> bool:
    return s == BuildActionStatus.WAITING or s == BuildActionStatus.RUNABLE


def runable_status(s: BuildActionStatus) -> bool:
    return s == BuildActionStatus.RUNABLE or s == BuildActionStatus.BLOCKED_RUNABLE


def finished_status(s: BuildActionStatus) -> bool:
    return s == BuildActionStatus.SUCCESSFULL or s == BuildActionStatus.FAILED


@dataclass
class BuildAction:
    label: ActionLabel
    needed: bool
    status: BuildActionStatus = BuildActionStatus.WAITING
    deps: set[ActionLabel] = field(default_factory=set)
    hard_deps: set[ActionLabel] = field(default_factory=set)
    dependers: set[ActionLabel] = field(default_factory=set)
    done_actions: set[Callable[[], Awaitable[None]]] = field(default_factory=set)
    connections: set[asyncio.StreamWriter] = field(default_factory=set)


class ActionExecuter(Protocol):
    async def schedule_action(self, action: BuildAction) -> None:
        """Execute action if it matches category"""
        ...


class ActionExecuterImpl(ActionExecuter):
    actions: dict[ActionLabel, BuildAction]
    waiting: set[ActionLabel]
    runable: set[ActionLabel]
    started: dict[asyncio.Task[Any], tuple[asyncio.subprocess.Process, BuildAction]]
    blocked: set[ActionLabel]
    blocked_runable: set[ActionLabel]
    failures: list[BuildAction]
    max_running: int
    invoker: ActionInvoker
    invocation_dir: Path

    def __init__(self, max_running: int, invocation_dir: Path) -> None:
        self.actions = {}
        self.waiting = set([])
        self.runable = set([])
        self.started = {}
        self.blocked = set([])
        self.blocked_runable = set([])
        self.need_resort = False
        self.failures = []
        self.deadlock_detected = False
        self.max_running = max_running
        self.invoker = DummyInvoker()
        self.invocation_dir = invocation_dir
        self.open_connections: set[asyncio.StreamWriter] = set([])
        self.connection_reader_tasks: set[asyncio.Task[Any]] = set()
        self.pending: list[BuildAction] = []
        self.wakeup: asyncio.Queue[None] = asyncio.Queue()

    def _update_dependers(self, action: BuildAction) -> None:
        for d in action.deps:
            self.actions[d].dependers.add(action.label)

    async def _waiting_or_runable(self, action: BuildAction) -> None:
        if waiting_status(action.status) or runable_status(action.status):
            undone_dep = False
            for d in action.deps:
                logger.debug(
                    f"{action.label} depends on {d} with status {self.actions[d].status}"
                )
                if not finished_status(self.actions[d].status):
                    logger.debug(f"{action.label} must wait for {d} to finish.")
                    undone_dep = True
                    break

            match action.status:
                case BuildActionStatus.BLOCKED | BuildActionStatus.BLOCKED_RUNABLE:
                    if undone_dep:
                        action.status = BuildActionStatus.BLOCKED
                        self.blocked.add(action.label)
                        self.blocked_runable.discard(action.label)
                    else:
                        action.status = BuildActionStatus.BLOCKED_RUNABLE
                        self.blocked_runable.add(action.label)
                        self.blocked.discard(action.label)

                case BuildActionStatus.WAITING | BuildActionStatus.RUNABLE:
                    if undone_dep:
                        action.status = BuildActionStatus.WAITING
                        self.waiting.add(action.label)
                        self.runable.discard(action.label)
                    else:
                        action.status = BuildActionStatus.RUNABLE
                        self.runable.add(action.label)
                        self.waiting.discard(action.label)
                case _:
                    assert False

        await self.wakeup.put(None)
        logger.info(f"{action.label} is now in status {action.status}")

    def running(self) -> int:
        return len(self.started) - len(self.blocked) - len(self.blocked_runable)

    def schedule_action_nonasync(self, action: BuildAction) -> None:
        self.pending.append(action)

    async def schedule_action(self, action: BuildAction) -> None:
        self.need_resort = True
        logger.debug(f"Scheduling {action.label}")
        if action.label in self.actions:
            old_action = self.actions[action.label]
            old_action.deps.update(action.deps)
            old_action.done_actions.update(action.done_actions)
            old_action.needed = old_action.needed or action.needed
            # old_action dependenders already done, only new ones
            self._update_dependers(action)
            await self._waiting_or_runable(old_action)
        else:
            self.actions[action.label] = action
            self.waiting.add(action.label)
            for d in load_action_deps(action.label):
                logger.debug(f"Adding dependency {d} for {action.label}")
                action.deps.add(d)
                if d not in self.actions:
                    await self.schedule_action(BuildAction(d, needed=False))
            if action.label.name != "FUSEBUILD.py":
                bf_label = ActionLabel(action.label.path, "FUSEBUILD.py")
                logger.debug(f"Adding {bf_label} for {action.label}")

                action.deps.add(bf_label)
                action.hard_deps.add(bf_label)
                if bf_label not in self.actions:
                    await self.schedule_action(BuildAction(bf_label, needed=True))
                self._update_dependers(action)

            self._update_dependers(action)
            await self._waiting_or_runable(action)

    async def start_running(self, action: BuildAction) -> None:
        logger.debug(f"Starting {action.label}")
        print(f"{action.label}..")
        assert have_not_started(action.status)
        self.waiting.discard(action.label)
        self.runable.discard(action.label)
        action.status = BuildActionStatus.RUNNING
        cmd, env = run_action_cmd_env(
            action.label.path, action.label.name, self.invoker
        )
        proc = await asyncio.create_subprocess_exec(*cmd, env=env)
        logger.debug(f"Running {cmd} with env {env} in {proc.pid=}")
        task = asyncio.create_task(proc.wait())
        self.started[task] = (proc, action)

    async def action_done(self, task: asyncio.Task[Any]) -> None:
        assert task in self.started

        process, action = self.started.pop(task)
        logger.debug(f"{action.label}: {process.returncode}")
        if process.returncode == 0:
            action.status = BuildActionStatus.SUCCESSFULL
            print(f"{action.label} ... Ok")
            for done_cb in action.done_actions:
                await done_cb()
        else:
            print(f"{action.label} ... Failed")
            action.status = BuildActionStatus.FAILED
            if action.needed:
                self.failures.append(action)

        logger.info(f"{action.label} is now in status {action.status}")
        for d in action.dependers:
            await self._waiting_or_runable(self.actions[d])

    def _check_for_deadlock_inner(
        self, at: ActionLabel, seen: list[ActionLabel]
    ) -> bool:
        if at in seen:
            print(f"Deadlock: {at} ->", file=sys.stderr)
            for p in seen[::-1]:
                print(f"   {p} ->", file=sys.stderr)
            return True

        seen_now = seen + [at]
        for hd in self.actions[at].hard_deps:
            if self._check_for_deadlock_inner(hd, seen_now):
                return True

        return False

    def _check_for_deadlock(self, start: ActionLabel) -> bool:
        return self._check_for_deadlock_inner(start, [])

    async def _read_from_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peername = writer.get_extra_info("peername")
        logger.info(f"Start reading from {peername}")
        try:
            while not reader.at_eof():
                logger.debug(f"Before reading from {peername}")
                line_bytes = await reader.readline()
                logger.debug(f"After reading from {peername}: {line_bytes=}")

                if not line_bytes:
                    break
                line: str = line_bytes.decode().rstrip()
                logger.debug(f"Received from {peername}: {line}")
                split = line.split(" ")
                if split[0] == "needs:":
                    invoker_label = label_from_line(unescape_whitespace(split[1]))
                    to_build = label_from_line(unescape_whitespace(split[2]))
                    hard = len(split) > 3 and split[3] == "hard"
                    invoking_action = self.actions[invoker_label]
                    await self.schedule_action(
                        BuildAction(to_build, invoking_action.needed)
                    )
                    invoking_action.deps.add(to_build)
                    self._update_dependers(invoking_action)

                    invoking_action.connections.add(writer)
                    invoking_action.status = BuildActionStatus.BLOCKED
                    self.blocked.add(invoking_action.label)

                    if hard:
                        invoking_action.hard_deps.add(to_build)
                        if self._check_for_deadlock(invoking_action.label):
                            invoking_action.status = BuildActionStatus.FAILED
                            self.deadlock_detected = True
                            self.failures.append(invoking_action)

                    await self._waiting_or_runable(invoking_action)
                else:
                    logger.error("Got unknwown command on internal socket:" + line)
        except asyncio.CancelledError:
            logger.info(f"Reader task for {peername} cancelled.")
        except ConnectionResetError:
            logger.info(f"Reader task for {peername} closed.")
        except BrokenPipeError:
            logger.info(f"Reader task for {peername} closed (broken pipe).")
        except Exception as ex:
            logger.error("An error occurred", exc_info=True)
            logger.error(f"Error reading from {peername}: {type(ex)}")
        finally:
            logger.info(f"Closing connection from {peername}")
            self.open_connections.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass
            except BrokenPipeError:
                pass

    def remove_connection_reader_task(self, task: asyncio.Task[Any]) -> None:
        logger.debug("Removing connection")
        self.connection_reader_tasks.discard(task)

    async def handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peername = writer.get_extra_info("peername")
        logger.info(f"Received connection from {peername}")
        self.open_connections.add(writer)

        task = asyncio.create_task(self._read_from_connection(reader, writer))
        self.connection_reader_tasks.add(task)
        logger.debug(
            f"Now there are {len(self.connection_reader_tasks)}/{len(self.open_connections)} connections"
        )
        task.add_done_callback(self.remove_connection_reader_task)

    async def unblock(self, action: BuildAction) -> None:
        assert len(action.connections) > 0
        for c in action.connections:
            logger.debug(f"Sending 'try again' to {action.label}")
            c.write(b"try again\n")
        self.blocked_runable.remove(action.label)
        self.status = BuildActionStatus.RUNNING

    def pick_waiter(self) -> BuildAction:
        return self.pick_one(self.waiting)

    def pick_one(self, runables: set[ActionLabel]) -> BuildAction:
        best = None
        best_count = -1
        for label in runables:
            action = self.actions[label]
            c = len(
                [
                    l
                    for l in action.dependers
                    if not finished_status(self.actions[l].status)
                ]
            )
            if c > best_count:
                best_count = c
                best = action

        assert best is not None
        return best

    async def run(self) -> int:
        s_path = socket_path()
        assert s_path is not None
        server = await asyncio.start_unix_server(
            self.handle_connection, path=str(s_path)
        )
        logger.info(f"Listening on unix socket {socket_path}")
        for action in self.pending:
            await self.schedule_action(action)
        self.pending = []
        next_print = time.monotonic()
        try:
            while True:
                if len(self.failures) > 0:
                    failure = self.failures[0]
                    print_failure(failure.label, set([]))
                    if self.deadlock_detected:
                        return 4
                    else:
                        return 3
                now = time.monotonic()
                if now > next_print:
                    next_print += 1
                    for a in self.waiting:
                        print(f"{a} waiting")
                    for a in self.runable:
                        print(f"{a} runable")
                    for a in self.blocked:
                        print(f"{a} blocked")
                    for a in self.blocked_runable:
                        print(f"{a} blocked runable")
                    for _, action in self.started.values():
                        if action.status == BuildActionStatus.RUNNING:
                            print(f"{action.label} running")

                logger.debug(
                    f"{len(self.waiting)=} {len(self.runable)=} {len(self.started)=} {len(self.blocked)=}  {len(self.blocked_runable)=}"
                )
                while self.running() < self.max_running:
                    if len(self.blocked_runable) > 0:
                        await self.unblock(self.pick_one(self.blocked_runable))
                    elif len(self.runable) > 0:
                        await self.start_running(self.pick_one(self.runable))
                    else:
                        break

                if (
                    self.running() == 0
                    and len(self.runable) == 0
                    and len(self.waiting) > 0
                ):
                    # We are stuck - cyclic dependencies
                    # We can try to force one to start
                    to_run = self.pick_waiter()
                    logger.warning(f"Might be in deadlock. Try running {to_run.label}")
                    await self.start_running(to_run)

                if (
                    self.running() == 0
                    and len(self.waiting) == 0
                    and len(self.runable) == 0
                    and len(self.blocked) == 0
                    and len(self.blocked_runable) == 0
                ):
                    return 0

                logger.debug(
                    f"Waiting for one of {len(self.started)} started actions and {len(self.connection_reader_tasks)}/{len(self.open_connections)} connections"
                )
                wakeup_task = asyncio.create_task(self.wakeup.get())

                done, pending = await asyncio.wait(
                    [t for t in self.started.keys()] + [wakeup_task],
                    timeout=1,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if len(done) == 0:
                    logger.debug("timeout")
                    continue

                for d in done:
                    logger.debug(f"Done d={d}")
                    if d == wakeup_task:
                        continue
                    await self.action_done(d)
        finally:
            logger.info("Closing unix socket")
            for client in self.open_connections:
                client.close()
            server.close()
            await server.wait_closed()
            logger.info("Unix socket closed")
            s_path.unlink(missing_ok=True)

            if self.connection_reader_tasks:
                logger.info(
                    f"Closing {len(self.connection_reader_tasks)} client connections"
                )
                tasks = list(self.connection_reader_tasks)
                for task in tasks:
                    task.cancel()

                logger.info("Gather {len(tasks)} connection reader tasks")
                await asyncio.gather(*tasks, return_exceptions=True)

            if self.open_connections:
                logger.warning(
                    f"{len(self.open_connections)} connections were not cleaned up properly."
                )
            logger.info("Done closing connections")


@dataclass(frozen=True)
class ScheduleAll:
    executer: ActionExecuter
    bf_label: ActionLabel
    categories: frozenset[str]

    async def __call__(self) -> None:
        actions = load_actions(self.bf_label.path)
        print(f"Loading all actions {self.bf_label}")
        for label, action in actions.items():
            if action.category in self.categories:
                await self.executer.schedule_action(BuildAction(label, needed=True))


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

    invocation_dir = Path(os.environ[FUSEBUILD_INVOCATION_DIR])
    logger.info(f"Using {invocation_dir} as invocation dir.")

    categories = frozenset(arg.category.split(","))

    invoker = DummyInvoker()
    max_running = arg.parallel
    if max_running <= 0:
        max_running = cpu_count()
    executer = ActionExecuterImpl(
        max_running=max_running, invocation_dir=invocation_dir
    )

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
                executer.schedule_action_nonasync(action)

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
                    executer.schedule_action_nonasync(
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
