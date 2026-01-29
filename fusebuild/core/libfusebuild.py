#
#    Copyright (C) 2025 Esben Nielsen <nielsen.esben@gmail.com>
#
# Based on LibFuse example:
#    Copyright (C) 2001  Jeff Epler  <jepler@unpythonic.dhs.org>
#    Copyright (C) 2006  Csaba Henk  <csaba.henk@creo.hu>
#
#    This program can be distributed under the terms of the GNU LGPL.
#    See the file COPYING.
#

from __future__ import print_function

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from errno import *
from pathlib import Path
from stat import *
from threading import Lock, Thread
from types import TracebackType
from typing import Any, Iterable, Optional, Protocol, Tuple, Union

import filelock
import marshmallow_dataclass2
import psutil
from marshmallow_dataclass2 import class_schema
from result import Err, Ok, Result
from watchdog.events import DirModifiedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

import fusebuild.core.logger as logger_module
from fusebuild.core.access_recorder import (
    AccessRecorder,
    check_accesses,
    merge_access_logs,
)
from fusebuild.core.action import (
    Action,
    ActionLabel,
    Mapping,
    MappingDefinition,
    RandomTmpDir,
    TmpDir,
    label_from_line,
)
from fusebuild.core.dependency import ActionSetupRecorder
from fusebuild.core.file_layout import (
    FUSEBUILD_INVOCATION_DIR,
    access_log_file,
    action_deps_file,
    action_dir,
    fusebuild_folder,
    has_invocation_dir,
    invocation_failed_file,
    invocation_ok_file,
    is_rule_output,
    last_definition_file,
    mountpoint_dir,
    new_access_log_file,
    output_folder_root,
    output_folder_root_str,
    status_file,
    status_lock_file,
    stderr_file,
    stdout_file,
    subbuild_failed_file,
    waiting_for_file,
)

from .action_invoker import ActionInvoker, WaitingFor
from .fuse_mount import BasicMount, unmount
from .utils import check_pid, kill_subprocess, os_environ, run_action

logger = logger_module.getLogger(__name__)


# Once a dependency is checked in an invocation, it stays ok
dependencies_ok: dict[ActionLabel, bool] = {}


def check_build_target(
    src_dir: Path, invoker: ActionInvoker
) -> tuple[bool, ActionLabel | None]:
    src_dir = src_dir.absolute()
    while True:
        logger.debug(f"{src_dir=}")

        target = src_dir.name
        src_dir = src_dir.parent
        build_file = src_dir / "FUSEBUILD.py"
        if build_file.exists():
            logger.debug(f"Found {build_file=}")
            break
        if src_dir == Path("/"):
            return True, None

    label = ActionLabel(src_dir.resolve(), target)

    if label in dependencies_ok:
        logger.debug(f"{label} was in dependencies_ok")
        return True, label

    if has_invocation_dir():
        if invocation_ok_file(label).exists():
            logger.debug(f"{label} was present in ok dir.")
            dependencies_ok[label] = True
            return True, label

        if invocation_failed_file(label).exists():
            logger.debug(f"{label} was already marked as failed.")
            return False, label

    action = get_action(src_dir, target, invoker)
    match action:
        case None:
            # clean_nonexisting_action(src_dir, target)
            return True, label
        case int(res):
            return False, label
        case _:  # Action
            pass

    logger.info(f"Dependency {src_dir}/{target}")
    res = run_action(src_dir, target, invoker)
    logger.info(f"Dependency {src_dir}/{target} returned {res}")
    if res == 0:
        dependencies_ok[label] = True
        return True, label

    return False, label


class StatusEnum(Enum):
    REQUIRING = 0
    CHECKING = 1
    RUNNING = 2
    DONE = 3
    DELETED = 4


@dataclass
class Status:
    status: StatusEnum
    running_pid: Optional[int] = os.getpid()


class ExecuterBase(ActionInvoker):
    def __init__(self, label: ActionLabel):
        self._waiting_for: dict[ActionLabel, int] = {}
        self.label = label
        self.full_path = self.directory / self.name
        logger.debug(f"{self.full_path=}")
        self.storage = action_dir(label)
        logger.debug(f"{self.storage=}")

    @property
    def directory(self) -> Path:
        return self.label.path

    @property
    def name(self) -> str:
        return self.label.name

    def runtarget_args(self) -> list[str]:
        return [str(self.label.path), self.label.name]

    @property
    def waiting_for_file(self) -> Path:
        return waiting_for_file(self.label)

    def _write_waiting_for(self) -> None:
        logger.debug(
            f"write_waiting_for {self.label} to {self.waiting_for_file}: {self._waiting_for}"
        )
        with self.waiting_for_file.open("w") as f:
            for l, c in self._waiting_for.items():
                json.dump({"label": (str(l.path), l.name), "count": c}, f)

    def _read_waiting_for(self) -> dict[ActionLabel, int]:
        with self.waiting_for_file.open("r") as f:
            for line in f.readlines():
                js = json.loads(line)
                label = ActionLabel(Path(js["label"][0]), js["label"][1])
                self._waiting_for[label] = int(js["count"])

        return self._waiting_for

    def get_status_lock_file(self) -> filelock.FileLock:
        return filelock.FileLock(str(status_lock_file(self.label)))

    def _read_status(self) -> Optional[Status]:
        status_file_path = status_file(self.label)
        if not status_file_path.exists():
            return None

        schema = marshmallow_dataclass2.class_schema(Status)()
        with status_file_path.open("r") as f:
            d = json.load(f)
            status: Status = schema.load(d)
            return status

    def waiting_for(self, label: ActionLabel) -> AbstractContextManager[WaitingFor]:
        class WaitingForImpl(WaitingFor):
            def __enter__(this) -> WaitingFor:
                with self.get_status_lock_file():
                    self._read_waiting_for()
                    if label in self._waiting_for:
                        self._waiting_for[label] += 1
                    else:
                        self._waiting_for[label] = 1

                    self._write_waiting_for()
                return this

            def __exit__(
                this,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> None:
                with self.get_status_lock_file():
                    self._read_waiting_for()
                    assert label in self._waiting_for
                    assert self._waiting_for[label] > 0
                    self._waiting_for[label] -= 1
                    if self._waiting_for[label] <= 0:
                        self._waiting_for.pop(label)
                    self._write_waiting_for()

        return WaitingForImpl()


class BasicExecuter(ExecuterBase):
    def __init__(self, label: ActionLabel, action: Action):
        super().__init__(label)
        self.action_setup_record = ActionSetupRecorder(action, os_environ())
        self.incremental = False
        self.building = True
        self.build_process: psutil.Popen | None = None
        self.fuse_mount: psutil.Popen | None = None

    def output_folder(self) -> Path:
        return Path(output_folder_root_str + str(self.full_path))

    @property
    def action(self) -> Action:
        return self.action_setup_record.definition

    @property
    def cmd(self) -> list[str]:
        return self.action.cmd

    def clean(self) -> None:
        output = self.output_folder()
        logger.debug(f"Cleaning {output}")
        if output.exists():
            shutil.rmtree(output)

    def mark_as_done(self, retcode: int) -> None:
        if retcode == 0:
            dependencies_ok[self.label] = True
        if has_invocation_dir():
            done_file = (
                invocation_ok_file(self.label)
                if retcode == 0
                else invocation_failed_file(self.label)
            )
            done_file.parent.mkdir(parents=True, exist_ok=True)
            with done_file.open("w") as f:
                f.write(f"{retcode}\n")

    def run(self) -> int | None:
        self.status = StatusEnum.RUNNING
        self.write_status()

        subbuild_failed_file(self.label).unlink(missing_ok=True)

        mountpoint = mountpoint_dir(self.label)
        logger.info(f"{mountpoint=}")
        try:
            unmount(mountpoint, quite=True)  # Expect unmount to fail
            mountpoint.mkdir(parents=True, exist_ok=True)
        except:
            unmount(mountpoint)
            mountpoint.mkdir(parents=True, exist_ok=True)

        writeable = self.output_folder()
        logger.debug(f"{writeable=}")
        writeable.mkdir(parents=True, exist_ok=True)
        cwd = self.directory.absolute()
        subbuild_failed_file(self.label).unlink(missing_ok=True)
        if len(list(mountpoint.iterdir())) > 0:
            logger.error(f"{mountpoint} not empty:  {list(mountpoint.iterdir())=}")
            assert False

        mount_cmd = [
            "python3",
            "-m",
            "fusebuild.core.mountview",
            str(self.directory.absolute()),
            self.name,
        ]
        logger.debug(f"Mount {mount_cmd=} {os.environ=}")
        self.fuse_mount = psutil.Popen(
            mount_cmd,
            env=os.environ,
        )
        logger.debug(f"Mountview for {self.label} is pid={self.fuse_mount.pid}")
        outer_cwd = Path(str(mountpoint) + "/" + str(cwd))
        logger.debug(f"Waiting for mount {mountpoint} at {cwd} by testing {outer_cwd}")
        while True:
            if outer_cwd.exists():
                break
            try:
                self.fuse_mount.wait(0.01)
                logger.error(
                    f"Mount on {mountpoint=} before build process started {self.fuse_mount.returncode=}"
                )
                return -1
            except psutil.TimeoutExpired:
                pass

        logger.debug("Mount complete")
        logger.debug(f"Start {self.cmd} in {cwd}")
        env = os.environ.copy()
        output_dir_full_path = str(self.output_folder().absolute())
        logger.debug(f"{output_dir_full_path=}")
        env["OUTPUT_DIR"] = output_dir_full_path
        match self.action.tmp:
            case TmpDir(p):
                env["TMPDIR"] = p
            case RandomTmpDir():
                env["TMPDIR"] = tempfile.mkdtemp()
            case None:
                if "TMPDIR" in env:
                    env.pop("TMPDIR")
            case _:
                assert False

        spawn_cmd, spawn_env = self.action.sandbox.generate_command(
            mountpoint, cwd, self.cmd, env
        )

        try:
            logger.info(f"{spawn_cmd=}  {spawn_env=}")
            with stderr_file(self.label).open("wb") as err_file:
                with stdout_file(self.label).open("wb") as out_file:
                    self.build_process = psutil.Popen(
                        spawn_cmd, env=spawn_env, stderr=err_file, stdout=out_file
                    )
                    logger.debug(
                        f"Build process for {self.label} is {self.build_process.pid=}"
                    )
                    assert self.build_process is not None
                    subbuild_failed = False
                    subbuild_failed_path = subbuild_failed_file(self.label)
                    while True:
                        logger.debug(
                            f"Waiting for cmd for {self.label} to finish {self.build_process.pid if self.build_process is not None else None}"
                        )
                        try:
                            assert self.build_process is not None
                            self.action_setup_record.return_code = (
                                self.build_process.wait(1.0)
                            )
                            self.build_process = None
                            break
                        except psutil.TimeoutExpired:
                            pass

                        if not self.fuse_mount.is_running():
                            logger.error("Fuse mount exited too early")
                            assert self.build_process is not None
                            self.action_setup_record.return_code = -1
                            break

                        subbuild_failed = subbuild_failed_path.exists()
                        if subbuild_failed:
                            logger.info(
                                f"Subbuild for {self.label} failed: {subbuild_failed_path.read_text()}, setting return code to -1"
                            )
                            self.action_setup_record.return_code = -1
            logger.debug(f"{self.cmd} done: {self.action_setup_record.return_code}")

            logger.debug(
                f"Checking {subbuild_failed_path=}: {subbuild_failed_path.exists()}"
            )
            subbuild_failed = subbuild_failed_path.exists()
            if subbuild_failed:
                logger.info("Return code == 0 but subbuild failed")
                self.action_setup_record.return_code = -1

            assert self.action_setup_record.return_code is not None
            self.mark_as_done(self.action_setup_record.return_code)
        except Exception as e:
            logger.error(
                f"Something went wrong when spawning {self.directory} / {self.name}: {e=}"
            )
            raise
        finally:
            logger.debug(f"Finally {mountpoint}")
            count = 0
            while (
                self.build_process is not None
                or self.fuse_mount is not None
                or os.path.ismount(mountpoint)
            ):
                count += 1
                quite = count < 10
                if self.build_process is not None:
                    try:
                        return_code = self.build_process.wait(1.0)
                        logger.debug(f"build process stopped: {return_code}")
                        # too late to use a good return code, something is wrong if it build process was running here
                        self.build_process = None
                    except psutil.TimeoutExpired:
                        logger.debug("Timeput while waiting for build process to stop")
                        assert self.fuse_mount is not None
                        if count > 60:
                            kill_subprocess(self.fuse_mount)
                    except Exception as e:
                        logger.error(f"Error while waiting for fuse mount to stop: {e}")

                if os.path.ismount(mountpoint):
                    if not unmount(mountpoint, quite=quite):
                        lc = logger.debug if quite else logger.error
                        lc(f"Failed to unmount {mountpoint} {count=}")
                    else:
                        logger.debug(f"Called unmount")

                if not os.path.ismount(mountpoint) and self.fuse_mount is not None:
                    try:
                        fuse_return_code = self.fuse_mount.wait(1.0)
                        logger.debug(f"Fuse mount stopped: {fuse_return_code}")
                        if fuse_return_code != 0:
                            logger.error(
                                f"Fuse mount didn't return ok: {fuse_return_code=}"
                            )
                            self.action_setup_record.return_code = -1
                        self.fuse_mount = None
                    except psutil.TimeoutExpired:
                        logger.debug("Timeput while waiting for fuse mount to stop")
                        assert self.fuse_mount is not None
                        if count > 60:
                            kill_subprocess(self.fuse_mount)
                    except Exception as e:
                        logger.error(f"Error while waiting for fuse mount to stop: {e}")

        logger.debug(f"Joined")
        assert len(list(mountpoint.iterdir())) == 0
        merge_access_logs(self.label)
        with self.last_definition().open("w") as f:
            first_line = self.action_setup_record
            schema = class_schema(ActionSetupRecorder)()
            first_dict = schema.dump(first_line)
            f.write(json.dumps(first_dict) + "\n")

        self.release_lock()
        return self.action_setup_record.return_code

    def release_lock(self) -> None:
        self.status = StatusEnum.DONE
        self.write_status()

    def require_for_build(self) -> tuple[bool, Status]:
        with self.get_status_lock_file() as lock:
            status = self._read_status()
            logger.debug(f"Status for {self.full_path} was {status}")
            if (
                not status
                or status.status == StatusEnum.DONE
                or status.running_pid is None
                or not check_pid(status.running_pid)
            ):
                status = Status(StatusEnum.REQUIRING)
                self._write_status(status)
                self._waiting_for.clear()
                self._write_waiting_for()
                return True, status
            else:
                return False, status

    def _write_status(self, status: Status) -> None:
        logger.debug(f"Writing status {status}")
        schema = marshmallow_dataclass2.class_schema(Status)()
        status_json = json.dumps(schema.dump(status))

        status_file_path = status_file(self.label)
        logger.debug(f"Writing status {status_json} to {status_file_path}")
        try:
            with status_file_path.open("w") as f:
                f.write(status_json)
        except Exception as e:
            logger.error(f"Can't write {status_file_path}: {e}")
            raise e

    def write_status(self) -> None:
        with self.get_status_lock_file() as lock:
            status = Status(
                self.status,
                running_pid=os.getpid(),
            )
            self._write_status(status)

    def handle_status_change(self) -> None:
        self.write_status()
        logger.debug(
            f"handle_status_change {self.fuse_mount=} and {self.build_process=}"
        )

    def last_definition(self) -> Path:
        return last_definition_file(self.label)

    def have_ok_file(self) -> bool:
        if has_invocation_dir():
            if invocation_ok_file(self.label).exists():
                logger.debug(f"({self.label} in invocation dir")
                dependencies_ok[self.label] = True
                return True

        return False

    def needs_rebuild(self, reason: str) -> Result[bool, bool]:
        """Check wether the action needs to be rebuild.
        In case the target was build ok it returns Ok(False), lock not held
        In case the target was build, but failed this build invocation, it returns Ok(True), lock is held
        In case it needs an rebuild (changed or failed last time): Ok(True), lock is held
        In case of deadlock Err(True), lock is not taken
        """

        if self.label in dependencies_ok:
            logger.debug(f"({self.directory}, {self.name}) in dependencies_ok")
            return Ok(False)

        if self.have_ok_file():
            return Ok(False)

        count = 0
        observer: Any = None
        semaphore = threading.Semaphore()
        while True:
            count += 1
            required, status = self.require_for_build()
            if required:
                logger.info(f"Got lock for {self.label}")
                break
            if observer is None:
                observer = Observer()

                class StatusChangedEvent(FileSystemEventHandler):
                    def on_modified(
                        this, event: DirModifiedEvent | FileModifiedEvent
                    ) -> None:
                        semaphore.release()

                observer.schedule(StatusChangedEvent(), status_file(self.label))
                observer.start()
                continue  # Run immediately again to avoid loosing an event

            if count > 2:
                logger.warning(
                    f"Couldn't require {self.label} for {reason} for build: {status=} deadlock?"
                )
                if check_deadlock(self.label, reason):
                    return Err(True)
                if count % 2 == 1:
                    print(f"Waiting for {self.label} while {reason}")
            else:
                logger.debug(
                    f"Couldn't require {self.label} for build: {status=} {count=}"
                )

            semaphore.acquire(timeout=1.0)
        if observer is not None:
            observer.stop()
            observer.join()

        if self.have_ok_file():
            self.release_lock()
            return Ok(False)
        self.status = StatusEnum.CHECKING
        self.write_status()
        merge_access_logs(self.label)
        dependency_log = self.last_definition()
        action_setup_record_old: ActionSetupRecorder | None = None
        if dependency_log.exists():
            fail = False
            action_setup_schema = class_schema(ActionSetupRecorder)()
            try:
                with dependency_log.open("r") as f:
                    line = f.readline()
                    logger.debug(f"{line=}")
                    action_setup_record_old = action_setup_schema.load(json.loads(line))
            except Exception as e:
                logger.error(f"Error while loading old access log: {e}")
                print("Failed to load previous dependencies")
                traceback.print_exc()
                fail = True
        else:
            print("Not build before")
            logger.info(f"{dependency_log} doesn't exist")

        if action_setup_record_old is None:
            logger.debug("Starting a new AccessRecorder")
            matches = False
        else:
            if (
                action_setup_record_old.definition
                != self.action_setup_record.definition
            ):
                print(f"{self.directory} / {self.name}: Action changed")
                matches = False
            else:
                matches = check_accesses(self.label, check_build_target, self)

        if action_setup_record_old is None:
            return Ok(True)
        elif matches and action_setup_record_old.return_code == 0:
            self.mark_as_done(0)
            return Ok(False)
        else:
            return Ok(True)

    def run_if_needed(self, invoker: ActionInvoker, reason: str) -> int | None:
        with invoker.waiting_for(self.label) as waiter:
            needs_rebuild = self.needs_rebuild(reason)
            logger.debug(f"{self.label} needs_rebuild: {needs_rebuild}")
            match needs_rebuild:
                case Ok(False):
                    return 0
                case Ok(True):
                    pass
                case Err(True):
                    return -1
                case _:
                    assert False
            if (
                has_invocation_dir()
                and (failed_file := invocation_failed_file(self.label)).exists()
            ):
                retcode = int(failed_file.read_text())
                logger.info(f"{self.label} failed with {retcode=}")
                return retcode
            elif self.incremental:
                logger.info(
                    f"Incremantal - run {self.full_path} again without cleaning"
                )
                return self.run()
            else:
                logger.info(f"Clean and run {self.full_path} again")
                self.last_definition().unlink(missing_ok=True)
                self.clean()
                access_log_file(self.label).unlink(missing_ok=True)
                new_access_log_file(self.label).unlink(missing_ok=True)
                action_deps_file(self.label).unlink(missing_ok=True)
            return self.run()


def check_deadlock_inner(
    seen: set[ActionLabel], label: ActionLabel
) -> list[ActionLabel]:
    executer = ExecuterBase(label)
    seen_next = seen.union([label])
    with executer.get_status_lock_file().acquire(blocking=False):
        status = executer._read_status()
        if status is None or status.status == StatusEnum.DONE:
            return []
        if not status.running_pid or not check_pid(status.running_pid):
            return []

        for l in executer._read_waiting_for():
            if l in seen:
                return [label, l]
            deadlock_list = check_deadlock_inner(seen_next, l)
            if len(deadlock_list) > 0:
                return [label] + deadlock_list

    return []


def check_deadlock(label: ActionLabel, reason: str) -> bool:
    try:
        deadlock_list = check_deadlock_inner(set([label]), label)
        if len(deadlock_list) > 0:
            logger.warning(f"Deadlock: {deadlock_list}")
            print(f"Deadlock when {reason}:")
            for l in deadlock_list:
                print(f"    {l} ->")
            return True
        return False
    except filelock.Timeout:
        # Couldn't optain all the status locks, declare no deadlock for now
        return False


class LoadBuildFileExecuter(BasicExecuter):
    def __init__(self, buildfile: Path) -> None:
        buildfile = buildfile.absolute()
        cmd = [
            "python",
            "-m",
            "fusebuild.core.load_build_file",
            str(buildfile),
        ]

        self.buildfile = buildfile
        super(LoadBuildFileExecuter, self).__init__(
            ActionLabel(buildfile.parent, "FUSEBUILD.py"),
            Action(cmd, category="", tmp=RandomTmpDir()),
        )

    def clean(self) -> None:
        output = self.output_folder()
        logger.debug(f"Cleaning action.json files from {output}")
        for action_file in output.glob("*.json"):
            action_file.unlink()

        schema = marshmallow_dataclass2.class_schema(Action)()
        output.mkdir(exist_ok=True, parents=True)
        action_def = output / (self.buildfile.name + ".json")
        logger.debug(f"Writing {action_def=}")
        with action_def.open("w") as f:
            json.dump(schema.dump(self.action), f)


class ActionExecuter(BasicExecuter):
    def __init__(self, label: ActionLabel, action: Action) -> None:
        super(ActionExecuter, self).__init__(label, action)


def clean_nonexisting_action(path: Path, name: str) -> None:
    logger.info(f"Clean non-exsisting target {path}/{name}")
    empty_action = Action(cmd=[], category="internal_clean")
    executer = ActionExecuter(ActionLabel(path, name), empty_action)
    executer.require_for_build()
    executer.clean()
    executer.status = StatusEnum.DELETED
    executer.write_status()


visited_directories: set[Path] = set([])
loaded_actions: dict[ActionLabel, Action] = {}


def all_actions() -> Iterable[ActionLabel]:
    return loaded_actions.keys()


action_schema = marshmallow_dataclass2.class_schema(Action)()


def load_action_file(label: ActionLabel, action_file: Path) -> Action:
    logger.debug(f"Loading action {label} from {action_file}")
    with action_file.open("r") as f:
        d = json.load(f)
        action: Action = action_schema.load(d)
        for provider in action.providers.values():
            provider.output_dir = (
                Path(output_folder_root_str + str(label.path)) / label.name
            )

    logger.debug(f"Got action {action_file}: {action}")
    return action


def load_actions(path: Path) -> dict[ActionLabel, Action]:
    logger.debug(f"load_actions({path})")
    actions_path = (
        Path(output_folder_root_str + str(path)) / "FUSEBUILD.py"
    ).absolute()
    actions = {}
    for action_file in actions_path.glob("*.json"):
        name = action_file.name.removesuffix(".json")
        label = ActionLabel(path, name)
        action = load_action_file(label, action_file)
        loaded_actions[label] = action
        actions[label] = action

    visited_directories.add(path)
    return actions


def check_build_file(
    buildfile: Path, invoker: ActionInvoker, reason: str
) -> Result[dict[ActionLabel, Action], int | None]:
    logger.debug(f"check_build_file({buildfile})")
    buildfile = buildfile.absolute()
    if not buildfile.exists():
        return Err(None)
    executer = LoadBuildFileExecuter(buildfile)
    ret = executer.run_if_needed(invoker, reason)
    logger.debug(f"run_if_needed for {buildfile}: {ret}")
    if ret is not None:
        executer.release_lock()
    logger.debug(f"LoadBuildFileAction({buildfile}) {ret=}")
    if ret != 0 and ret is not None:
        return Err(ret)

    return Ok(load_actions(buildfile.parent))


def get_action(
    path: Path | str, action: str, invoker: ActionInvoker
) -> Action | None | int:
    logger.debug(f"get_action({path}, {action})")
    if isinstance(path, str):
        path = Path(path)

    path = path.resolve()
    if path.is_file():
        path = path.parent

    label = ActionLabel(path, action)
    if label in loaded_actions:
        return loaded_actions[label]

    if path in visited_directories:
        # SBUILD file load and actions are up-to-date, but no such action
        return None

    res = check_build_file(path / "FUSEBUILD.py", invoker, str(label))
    match res:
        case Err(ret):
            logger.warning(f"Failed to load {path / 'FUSEBUILD.py'}: Returned {ret}")
            return ret
        case Ok(actions):
            pass

    if label not in loaded_actions:
        logger.info(f"{label} not defined {loaded_actions=}")
        return None

    return loaded_actions[label]


def get_action_executer(
    p: Path, name: str, invoker: ActionInvoker
) -> BasicExecuter | None:
    if name == "FUSEBUILD.py":
        return LoadBuildFileExecuter(p / name)

    action = get_action(p, name, invoker)
    logger.debug(f"get_action_executer({p}, {name}) = {action}")
    match action:
        case None:
            return None
        case int(res):
            return None
        case _:
            p = p.resolve()
            if p.is_file():
                p = p.parent
            return ActionExecuter(ActionLabel(p, name), action)


def get_action_from_path(
    t: Path, invoker: ActionInvoker
) -> tuple[ActionLabel, Action] | None:
    t = t.absolute()
    while True:
        name = t.name
        t = t.parent
        logger.debug(f"{t=} {name=}")
        build_file = t / "FUSEBUILD.py"
        if build_file.exists():
            action = get_action(t, name, invoker)
            logger.debug(f"get_action({t}, {name}) returned {action=}")
            if isinstance(action, Action):
                return (ActionLabel(t, name), action)
            else:
                return None

        if t == Path(".") or t == Path("/"):
            break

    return None
