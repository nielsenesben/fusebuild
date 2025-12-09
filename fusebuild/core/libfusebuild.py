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
import uuid
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

from fusebuild.core.access_recorder import (
    AccessRecorder,
    access_log_file,
    action_deps_file,
    check_accesses,
    merge_access_logs,
    new_access_log_file,
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
    action_dir,
    fusebuild_folder,
    is_rule_output,
    output_folder_root,
    output_folder_root_str,
)

from .action_invoker import ActionInvoker

# pull in some spaghetti to make this stuff work without fuse-py being installed
try:
    import _find_fuse_parts  # type: ignore
except ImportError:
    pass
import fuse  # type: ignore
from fuse import Fuse, FuseArgs, FuseError

import fusebuild.core.logger as logger_module

logger = logger_module.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

FUSEBUILD_INVOCATION_DIR = "FUSEBUILD_INVOCATION_DIR"


if not hasattr(fuse, "__version__"):
    raise RuntimeError(
        "your fuse-py doesn't know of fuse.__version__, probably it's too old."
    )

fuse.fuse_python_api = (0, 2)

fuse.feature_assert("stateful_files", "has_init")


def os_environ() -> dict[str, str]:
    return {k: os.environ[k] for k in os.environ}


def check_pid(pid: int) -> bool:
    """Check For the existence of a unix pid."""
    psutil.process_iter.cache_clear()
    return psutil.pid_exists(pid)


def kill_subprocess(process: subprocess.Popen | psutil.Popen) -> None:
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


def flag2mode(flags: int) -> str:
    md = {os.O_RDONLY: "rb", os.O_WRONLY: "wb", os.O_RDWR: "wb+"}
    m = md[flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR)]

    if flags | os.O_APPEND:
        m = m.replace("w", "a", 1)

    return m


def run_action(directory: Path, target: str, invoker: ActionInvoker) -> int:
    logger.debug(f"Run action {directory} / {target}")
    process = None
    try:
        process = psutil.Popen(
            [
                "python3",
                str(Path(__file__).parent / "runtarget.py"),
                str(directory.absolute() / "FUSEBUILD.py"),
                target,
            ]
            + invoker.runtarget_args(),
            env=os.environ,
        )
        res = process.wait()
    except:
        logger.error(f"Something went wrong when invoking {directory} / {target}")
        if process is not None:
            kill_subprocess(process)
        raise
    return res


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
        logger.debug(f"{build_file=}")
        if build_file.exists():
            break
        if src_dir == Path("/"):
            return True, None

    label = ActionLabel(src_dir, target)

    if label in dependencies_ok:
        logger.debug(f"{label} was in dependencies_ok")
        return True, label

    if FUSEBUILD_INVOCATION_DIR in os.environ:
        if Path(
            os.environ[FUSEBUILD_INVOCATION_DIR] + "/ok/" + str(src_dir) + "/" + target
        ).exists():
            logger.debug(f"{label} was present in ok dir.")
            dependencies_ok[label] = True
            return True, label

        if Path(
            os.environ[FUSEBUILD_INVOCATION_DIR]
            + "/failed/"
            + str(src_dir)
            + "/"
            + target
        ).exists():
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


class BasicMount(Fuse):  # type: ignore
    def __init__(
        self,
        label: ActionLabel,
        invoker: ActionInvoker,
        mountpoint: Path,
        access_recorder: AccessRecorder,
        writeable: str,
        mappings: list[MappingDefinition] = [],
        *args: Any,
        **kw: Any,
    ) -> None:
        self.label = label
        self.invoker = invoker
        self.fuse_args = FuseArgs()
        self.fuse_args.setmod("foreground")
        self.fuse_args.mountpoint = str(mountpoint.absolute())
        self.fuse_args.optlist = ["auto_unmount", "intr"]
        self.mountpoint = mountpoint
        self.subbuild_failed = False

        Fuse.__init__(self, *args, fuse_args=self.fuse_args, **kw)
        self.root = "/"
        self.writeable = writeable
        self.access_recorder = access_recorder
        self.mappings = [m.create(output_folder_root) for m in mappings]
        logger.debug(f"{self.mappings=}")

    def is_rule_output(self, path: Path | str) -> bool:
        return is_rule_output(path)

    def handle_other_rule_output(self, path: str) -> None:
        if not self.is_rule_output(path):
            return

        if self.is_writeable(path):
            return

        relative_to_output_root = Path(path).relative_to(output_folder_root)
        logger.debug(f"{relative_to_output_root=}")
        src_dir = Path("/") / relative_to_output_root

        if src_dir.is_dir():
            dirtocreate = Path(output_folder_root_str + str(src_dir))
            logger.info(
                f"Creating dir {dirtocreate} for some potential deper action  in {src_dir}"
            )
            dirtocreate.mkdir(exist_ok=True)
            self.access_recorder.record_dir_exists(src_dir, True)
        else:
            self.access_recorder.record_dir_exists(src_dir, False)
            success, label = check_build_target(src_dir, self.invoker)
            if label is not None:
                self.access_recorder.action_deps.add(label)
            if not success:
                with (action_dir(self.label) / "subbuild_failed").open("a") as f:
                    if label is None:
                        f.write("something odd")
                    else:
                        f.write(f"{label}\n")

            self.access_recorder.listener()

    def is_writeable(self, path: str) -> bool:
        res = os.path.commonprefix([self.writeable, path]) == self.writeable
        logger.debug(f"is_writeable {path}: {res}")
        if False:
            import traceback

            for line in traceback.format_stack():
                logger.debug(line.strip())
        return res

    def remap(self, path: Path) -> tuple[str, bool]:
        is_output = self.is_rule_output(path)
        for mapper in self.mappings:
            remapped = mapper.remap(path, is_output)
            if remapped is not None:
                logger.debug(f"Remapping {path=} -> {remapped=}")
                path_str = str(remapped)
                return path_str, self.is_rule_output(path_str)

        logger.debug(f"{path=} {is_output=}")
        return str(path), is_output

    def common_handle_path(self, path: Path) -> tuple[str, bool]:
        path_str, is_output = self.remap(path)
        if is_output:
            self.handle_other_rule_output(path_str)
        return path_str, is_output

    def getattr(self, path_in: Path) -> os.stat_result:
        path, is_output = self.common_handle_path(path_in)
        res = os.lstat("." + path)
        logger.debug(f"getattr {path} {res}")
        if not self.is_writeable(path):
            self.access_recorder.record_stat(path, res)
        if False and not self.is_writeable(path):
            res.st_mode &= 0o7666

        return res

    def readlink(self, path_in: Path) -> str:
        path: str
        path, is_output = self.common_handle_path(path_in)
        res = os.readlink("." + path)
        if not self.is_writeable(path):
            self.access_recorder.record_readlink(path, res)
        return res

    def readdir(self, path_in: Path, offset: int) -> Iterable[os.DirEntry]:
        path, is_output = self.common_handle_path(path_in)
        logger.debug(f"readdir {path} {offset}")
        to_record = []
        for e in os.listdir("." + path):
            to_record.append(e)
            res = fuse.Direntry(e)
            yield res
        if not self.is_writeable(path):
            self.access_recorder.record_readdir(path, to_record)

    def unlink(self, path_in: Path) -> None:
        path, is_output = self.remap(path_in)
        logger.debug(f"unlink {path}")
        if self.is_writeable(path):
            os.unlink("." + path)

    def rmdir(self, path_in: Path) -> None:
        path, is_output = self.remap(path_in)
        logger.debug(f"rmdir {path}")
        if self.is_writeable(path):
            os.rmdir("." + path)

    def symlink(self, path_in: Path, path1_in: Path) -> None:
        path, is_output = self.remap(path_in)
        path1, is_putput1 = self.remap(path1_in)
        if self.is_writeable(path1):
            os.symlink(path, "." + path1)

    def rename(self, path_in: Path, path1_in: Path) -> None:
        path, is_output = self.remap(path_in)
        path1, is_output1 = self.remap(path1_in)
        logger.debug(f"rename {path} {path1}")
        if self.is_writeable(path) and self.is_writeable(path1):
            os.rename("." + path, "." + path1)

    def link(self, path_in: Path, path1_in: Path) -> None:
        path, is_output = self.remap(path_in)
        path1, is_output1 = self.remap(path1_in)
        logger.debug(f"link {path} {path1}")
        if self.is_writeable(path1):
            os.link("." + path, "." + path1)

    def chmod(self, path_in: Path, mode: int) -> None:
        path, is_output = self.remap(path_in)
        logger.debug(f"rename {path} {mode}")
        if self.is_writeable(path):
            os.chmod("." + path, mode)

    def chown(self, path_in: Path, user: int, group: int) -> None:
        path, is_output = self.remap(path_in)
        logger.debug(f"chown {path} {user} {group}")
        if self.is_writeable(path):
            os.chown("." + path, user, group)

    def truncate(self, path_in: Path, len: int) -> None:
        path, is_output = self.remap(path_in)
        logger.debug(f"truncate {path} {len}")
        if self.is_writeable(path):
            f = open("." + path, "a")
            f.truncate(len)
            f.close()

    def mknod(self, path_in: Path, mode: int, dev: int) -> None:
        path, is_output = self.remap(path_in)
        logger.debug(f"mknod {path} {mode} {dev}")
        if self.is_writeable(path):
            os.mknod("." + path, mode, dev)

    def mkdir(self, path_in: Path, mode: int) -> None:
        path, is_output = self.remap(path_in)
        logger.debug(f"mkdir {path} {mode}")
        if self.is_writeable(path):
            os.mkdir("." + path, mode)

    def utime(
        self,
        path_in: Path,
        times: Optional[Union[Tuple[int, int], Tuple[float, float]]],
    ) -> None:
        path, is_output = self.remap(path_in)
        logger.debug(f"utime {path} {times}")
        if self.is_writeable(path):
            os.utime("." + path, times)

    #    The following utimens method would do the same as the above utime method.
    #    We can't make it better though as the Python stdlib doesn't know of
    #    sub-second preciseness in access/modify times.
    #
    #    def utimens(self, path, ts_acc, ts_mod):
    #      os.utime("." + path, (ts_acc.tv_sec, ts_mod.tv_sec))

    def access(self, path_in: Path, mode: int) -> int:
        path, is_output = self.common_handle_path(path_in)
        os_res = os.access("." + path, mode)
        if not self.is_writeable(path):
            self.access_recorder.record_access(path, mode, os_res)
        if not os_res or (mode == os.W_OK and not self.is_writeable(path)):
            res = -EACCES
        else:
            res = 0
        logger.debug(f"access {path} {mode}: {os_res} {res}")
        return res

    #    This is how we could add stub extended attribute handlers...
    #    (We can't have ones which aptly delegate requests to the underlying fs
    #    because Python lacks a standard xattr interface.)
    #
    #    def getxattr(self, path, name, size):
    #        val = name.swapcase() + '@' + path
    #        if size == 0:
    #            # We are asked for size of the value.
    #            return len(val)
    #        return val
    #
    #    def listxattr(self, path, size):
    #        # We use the "user" namespace to please XFS utils
    #        aa = ["user." + a for a in ("foo", "bar")]
    #        if size == 0:
    #            # We are asked for size of the attr list, i.e. joint size of attrs
    #            # plus null separators.
    #            return len("".join(aa)) + len(aa)
    #        return aa

    def statfs(self) -> os.statvfs_result:
        """
        Should return an object with statvfs attributes (f_bsize, f_frsize...).
        Eg., the return value of os.statvfs() is such a thing (since py 2.2).
        If you are not reusing an existing statvfs object, start with
        fuse.StatVFS(), and define the attributes.

        To provide usable information (i.e., you want sensible df(1)
        output, you are suggested to specify the following attributes:

            - f_bsize - preferred size of file blocks, in bytes
            - f_frsize - fundamental size of file blcoks, in bytes
                [if you have no idea, use the same as blocksize]
            - f_blocks - total number of blocks in the filesystem
            - f_bfree - number of free blocks
            - f_files - total number of file inodes
            - f_ffree - nunber of free file inodes
        """

        return os.statvfs(".")

    def fsinit(self) -> None:
        os.chdir(self.root)

    def main(self_outer) -> int:
        class BasicFile(object):
            def __init__(self, path_in: Path, flags: int, *mode: int) -> None:
                write_flags = (
                    os.O_WRONLY
                    | os.O_RDWR
                    | os.O_APPEND
                    | os.O_CREAT
                    | os.O_TRUNC
                    | os.O_FSYNC
                )

                self.path, is_output = self_outer.common_handle_path(path_in)
                self.writeable = self_outer.is_writeable(self.path)
                if not self.writeable and (flags & write_flags) != 0:
                    logger.warning(f"Refuse opening file for write at {self.path=}")
                    flags &= ~(write_flags)

                if not self.writeable:
                    self_outer.access_recorder.record_read(self.path)
                logger.debug(f"Open {self.path=} {flags=} {mode=} {flag2mode(flags)=}")
                self.file = os.fdopen(
                    os.open("." + self.path, flags, *mode), flag2mode(flags)
                )
                self.fd = self.file.fileno()
                if hasattr(os, "pread"):
                    self.iolock = None
                else:
                    self.iolock = Lock()

            def read(self, length: int, offset: int) -> bytes:
                if self.iolock:
                    self.iolock.acquire()
                    try:
                        self.file.seek(offset)
                        return self.file.read(length)
                    finally:
                        self.iolock.release()
                else:
                    return os.pread(self.fd, length, offset)

            def write(self, buf: bytes, offset: int) -> int:
                if not self_outer.is_writeable(self.path):
                    return -EACCES

                logger.debug(f"write {self.path} {offset=} {len(buf)=}")
                if self.iolock:
                    self.iolock.acquire()
                    try:
                        self.file.seek(offset)
                        self.file.write(buf)
                        return len(buf)
                    finally:
                        self.iolock.release()
                else:
                    return os.pwrite(self.fd, buf, offset)

            def release(self, flags: int) -> None:
                self.file.close()

            def _fflush(self) -> None:
                logger.debug(f"_fflush {self.path=} {self.file.mode=}")
                if (
                    "w" in self.file.mode or "a" in self.file.mode
                ) and self_outer.is_writeable(self.path):
                    self.file.flush()

            def fsync(self, isfsyncfile: bool) -> Any:
                if not self_outer.is_writeable(self.path):
                    return -EACCES

                self._fflush()
                if isfsyncfile and hasattr(os, "fdatasync"):
                    os.fdatasync(self.fd)
                else:
                    os.fsync(self.fd)

            def flush(self) -> None:
                self._fflush()
                # cf. xmp_flush() in fusexmp_fh.c
                os.close(os.dup(self.fd))

            def fgetattr(self) -> os.stat_result:
                return os.fstat(self.fd)

            def ftruncate(self, len: int) -> Any:
                logger.debug("ftruncate {self.path=} {len=}")
                if not self_outer.is_writeable(self.path):
                    return -EACCES

                self.file.truncate(len)

            def lock(self, cmd: int, owner: int, **kw: Any) -> Any:
                # The code here is much rather just a demonstration of the locking
                # API than something which actually was seen to be useful.

                # Advisory file locking is pretty messy in Unix, and the Python
                # interface to this doesn't make it better.
                # We can't do fcntl(2)/F_GETLK from Python in a platfrom independent
                # way. The following implementation *might* work under Linux.
                #
                # if cmd == fcntl.F_GETLK:
                #     import struct
                #
                #     lockdata = struct.pack('hhQQi', kw['l_type'], os.SEEK_SET,
                #                            kw['l_start'], kw['l_len'], kw['l_pid'])
                #     ld2 = fcntl.fcntl(self.fd, fcntl.F_GETLK, lockdata)
                #     flockfields = ('l_type', 'l_whence', 'l_start', 'l_len', 'l_pid')
                #     uld2 = struct.unpack('hhQQi', ld2)
                #     res = {}
                #     for i in xrange(len(uld2)):
                #          res[flockfields[i]] = uld2[i]
                #
                #     return fuse.Flock(**res)

                # Convert fcntl-ish lock parameters to Python's weird
                # lockf(3)/flock(2) medley locking API...
                op = {
                    fcntl.F_UNLCK: fcntl.LOCK_UN,
                    fcntl.F_RDLCK: fcntl.LOCK_SH,
                    fcntl.F_WRLCK: fcntl.LOCK_EX,
                }[kw["l_type"]]
                if cmd == fcntl.F_GETLK:
                    return -EOPNOTSUPP
                elif cmd == fcntl.F_SETLK:
                    if op != fcntl.LOCK_UN:
                        op |= fcntl.LOCK_NB
                    elif cmd == fcntl.F_SETLKW:
                        pass
                    else:
                        return -EINVAL

                fcntl.lockf(self.fd, op, kw["l_start"], kw["l_len"])

        self_outer.file_class = BasicFile
        assert not self_outer.mountpoint.is_mount()
        assert self_outer.mountpoint.is_dir()
        assert len(list(self_outer.mountpoint.iterdir())) == 0
        return Fuse.main(self_outer)


def unmount(mountpoint: Path, quite: bool = False) -> bool:
    cmd = ["fusermount", "-u", str(mountpoint.absolute())]
    if quite:
        with open(os.devnull, "w") as f:
            result = subprocess.run(cmd, stderr=f)
    else:
        result = subprocess.run(cmd)
    logger.info(f"{cmd}: {result=}")
    return result.returncode == 0


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


class ActionBase(ActionInvoker):
    def __init__(self, label: ActionLabel):
        self._waiting_for: dict[ActionLabel, int] = {}
        self.label = label
        self.full_path = self.directory / self.name
        logger.debug(f"{self.full_path=}")
        self.storage = fusebuild_folder / ("actions" + str(self.full_path))
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
        return self.storage / "waiting_for"

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
        return filelock.FileLock(str(self.storage / "status.lck"))

    def _read_status(self) -> Optional[Status]:
        status_file = self.storage / "status.json"
        if not status_file.exists():
            return None

        schema = marshmallow_dataclass2.class_schema(Status)()
        with (status_file).open("r") as f:
            d = json.load(f)
            return schema.load(d)

    def waiting_for(self, label: ActionLabel) -> AbstractContextManager:
        class WaitingFor:
            def __enter__(this) -> "WaitingFor":
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

        return WaitingFor()


class BasicAction(ActionBase):
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
        if output.exists():
            shutil.rmtree(output)

    def mark_as_done(self, retcode: int) -> None:
        if retcode == 0:
            dependencies_ok[self.label] = True
        if FUSEBUILD_INVOCATION_DIR in os.environ:
            done_file = Path(
                os.environ[FUSEBUILD_INVOCATION_DIR]
                + ("/ok/" if retcode == 0 else "/failed/")
                + str(self.full_path)
            )
            done_file.parent.mkdir(parents=True, exist_ok=True)
            with done_file.open("w") as f:
                f.write(f"{retcode}\n")

    def run(self) -> int | None:
        self.status = StatusEnum.RUNNING
        self.write_status()

        (action_dir(self.label) / "subbuild_failed").unlink(missing_ok=True)

        mountpoint = self.storage / "mountpoint"
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
        (action_dir(self.label) / "subbuild_failed").unlink(missing_ok=True)
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
        try:
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
                with (self.storage / "stderr").open("wb") as err_file:
                    with (self.storage / "stdout").open("wb") as out_file:
                        self.build_process = psutil.Popen(
                            spawn_cmd, env=spawn_env, stderr=err_file, stdout=out_file
                        )
                        logger.debug(
                            f"Build process for {self.label} is {self.build_process.pid=}"
                        )
                        assert self.build_process is not None
                        subbuild_failed = False
                        subbuild_failed_file = (
                            action_dir(self.label) / "subbuild_failed"
                        )
                        while True:
                            logger.debug(
                                f"Waiting for cmd for {self.label} to finish {self.build_process.pid=}"
                            )
                            procs = [
                                self.build_process,
                                self.fuse_mount,
                            ]
                            gone, alive = psutil.wait_procs(procs, timeout=1.0)
                            if self.build_process in gone:
                                self.action_setup_record.return_code = (
                                    self.build_process.returncode
                                )
                                logger.info(
                                    f"Running command for {self.label} finished {self.action_setup_record.return_code=}"
                                )
                                self.build_process = None
                                break
                            if self.fuse_mount in gone:
                                logger.error(
                                    f"Fuse mount exited too early with return code {self.fuse_mount.returncode}"
                                )
                                self.action_setup_record.return_code = -1
                                self.fuse_mount = None
                                kill_subprocess(self.build_process)
                                break
                            subbuild_failed = subbuild_failed_file.exists()
                            if subbuild_failed and self.build_process is not None:
                                logger.info(
                                    f"Subbuild for {self.label} failed: {subbuild_failed_file.read_text()}"
                                )
                                kill_subprocess(self.build_process)
                            if (
                                subbuild_failed
                                and self.action_setup_record.return_code is None
                            ):
                                logger.info(
                                    f"Subbuild for {self.label} failed: {subbuild_failed_file.read_text()}, setting return code to -1"
                                )

                                self.action_setup_record.return_code = -1
            except Exception as e:
                logger.error(
                    f"Something went wrong when spawning {self.directory} / {self.name}: {e=}"
                )
                if self.build_process is not None:
                    kill_subprocess(self.build_process)
                if self.fuse_mount is not None:
                    unmount(mountpoint)
                    kill_subprocess(self.fuse_mount)
                    assert len(list(mountpoint.iterdir())) == 0
                raise

            logger.debug(f"{self.cmd} done: {self.action_setup_record.return_code}")

            logger.debug(
                f"Checking {subbuild_failed_file=}: {subbuild_failed_file.exists()}"
            )
            subbuild_failed = subbuild_failed_file.exists()
            if subbuild_failed:
                logger.info("Return code == 0 but subbuild failed")
                self.action_setup_record.return_code = -1

            assert self.action_setup_record.return_code is not None
            self.mark_as_done(self.action_setup_record.return_code)
            return self.action_setup_record.return_code
        finally:
            logger.debug(f"Finally {mountpoint}")
            while self.fuse_mount is not None:
                if not unmount(mountpoint):
                    logger.error(f"Failed to unmount {mountpoint}")
                else:
                    logger.debug(f"Called unmount")
                    assert len(list(mountpoint.iterdir())) == 0

                try:
                    self.fuse_mount.wait(1.0)
                    logger.debug("Fuse mount stopped")
                    fuse_return_code = self.fuse_mount.returncode
                    if fuse_return_code != 0:
                        logger.error(
                            f"Fuse mount didn't return ok: {fuse_return_code=}"
                        )
                        self.action_setup_record.return_code = -1
                    self.fuse_mount = None
                except psutil.TimeoutExpired:
                    logger.debug("Timeput while waiting for fuse mount to stop")
                    pass

            logger.debug(f"Joined")
            assert len(list(mountpoint.iterdir())) == 0
            merge_access_logs(self.label)
            with self.last_definition().open("w") as f:
                first_line = self.action_setup_record
                schema = class_schema(ActionSetupRecorder)()
                first_dict = schema.dump(first_line)
                f.write(json.dumps(first_dict) + "\n")

            self.release_lock()

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

        status_file = self.storage / "status.json"
        logger.debug(f"Writing status {status_json} to {status_file}")
        try:
            with status_file.open("w") as f:
                f.write(status_json)
        except Exception as e:
            logger.error(f"Can't write {status_file}: {e}")
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
        return self.storage / "last_definition.json"

    def needs_rebuild(self, reason: str) -> Result[bool, bool]:
        if self.label in dependencies_ok:
            logger.debug(f"({self.directory}, {self.name}) in dependencies_ok")
            return Ok(False)
        if FUSEBUILD_INVOCATION_DIR in os.environ:
            if Path(
                os.environ[FUSEBUILD_INVOCATION_DIR]
                + "/ok/"
                + str(self.directory)
                + "/"
                + self.name
            ).exists():
                logger.debug(f"({self.directory}, {self.name}) in invocation dir")
                dependencies_ok[ActionLabel(self.directory, self.name)] = True
                return Ok(False)

        count = 0
        while True:
            count += 1
            required, status = self.require_for_build()
            if required:
                break
            if count > 1:
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
            time.sleep(1.0)

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
            match self.needs_rebuild(reason):
                case Err(b):
                    return -1
                case Ok(needs_rebuild):
                    if not needs_rebuild:
                        return 0
                case _:
                    assert False
            if (
                FUSEBUILD_INVOCATION_DIR in os.environ
                and (
                    failed_file := Path(
                        os.environ[FUSEBUILD_INVOCATION_DIR]
                        + "/failed/"
                        + str(self.full_path)
                    )
                ).exists()
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
    action = ActionBase(label)
    seen_next = seen.union([label])
    with action.get_status_lock_file().acquire(blocking=False):
        status = action._read_status()
        if status is None or status.status == StatusEnum.DONE:
            return []
        if not status.running_pid or not check_pid(status.running_pid):
            return []

        for l in action._read_waiting_for():
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


class LoadBuildFileAction(BasicAction):
    def __init__(self, buildfile: Path) -> None:
        buildfile = buildfile.absolute()
        cmd = [
            "python",
            "-m",
            "fusebuild.core.load_build_file",
            str(buildfile),
        ]

        self.buildfile = buildfile
        BasicAction.__init__(
            self,
            ActionLabel(buildfile.parent, "FUSEBUILD.py"),
            Action(cmd, category="", tmp=RandomTmpDir()),
        )

    def clean(self) -> None:
        output = self.output_folder()
        for action_file in output.glob("*.json"):
            action_file.unlink()

        schema = marshmallow_dataclass2.class_schema(Action)()
        output.mkdir(exist_ok=True, parents=True)
        action_def = output / (self.buildfile.name + ".json")
        logger.debug(f"Writing {action_def}")
        with action_def.open("w") as f:
            json.dump(schema.dump(self.action), f)


class RuleAction(BasicAction):
    def __init__(self, label: ActionLabel, action: Action) -> None:
        super(RuleAction, self).__init__(label, action)


def clean_nonexisting_action(path: Path, name: str) -> None:
    logger.info(f"Clean non-exsisting target {path}/{name}")
    empty_action = Action(cmd=[], category="internal_clean")
    action = RuleAction(ActionLabel(path, name), empty_action)
    action.require_for_build()
    action.clean()
    action.status = StatusEnum.DELETED
    action.write_status()


visited_directories: set[Path] = set([])
loaded_actions: dict[ActionLabel, Action] = {}


def all_actions() -> Iterable[ActionLabel]:
    return loaded_actions.keys()


action_schema = marshmallow_dataclass2.class_schema(Action)()


def load_action_file(label: ActionLabel, action_file: Path) -> Action:
    logger.debug(f"Loading action {label} from {action_file}")
    with action_file.open("r") as f:
        d = json.load(f)
        action = action_schema.load(d)
        for provider in action.providers.values():
            provider.output_dir = (
                Path(output_folder_root_str + str(label.path)) / label.name
            )

    logger.debug(f"Got action {action_file}: {action}")
    return action


def load_actions(path: Path) -> dict[ActionLabel, Action]:
    logger.debug(f"load_actions({path})")
    logger.debug(f"Looking for actions in {output_folder_root} / {str(path)}")
    actions = {}
    for action_file in (Path(output_folder_root_str + str(path)) / "FUSEBUILD.py").glob(
        "*.json"
    ):
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
    action = LoadBuildFileAction(buildfile)
    ret = action.run_if_needed(invoker, reason)
    if ret is not None:
        action.release_lock()
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
    path = path.absolute()
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


def get_rule_action(p: Path, name: str, invoker: ActionInvoker) -> RuleAction | None:
    action = get_action(p, name, invoker)
    logger.debug(f"get_rule_action({p}, {name}) = {action}")
    match action:
        case None:
            return None
        case int(res):
            return None
        case _:
            p = p.absolute()
            if p.is_file():
                p = p.parent
            return RuleAction(ActionLabel(p, name), action)


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


def find_all_actions(
    d: Path, invoker: ActionInvoker
) -> Result[dict[ActionLabel, Action], set[Path]]:
    fails: set[Path] = set([])
    actions: dict[ActionLabel, Action] = {}
    build_files = d.glob("**/FUSEBUILD.py")
    for bf in build_files:
        res = check_build_file(bf, invoker, "finding all actions in {d}")
        match res:
            case Err(ret):
                logger.error(f"Failed to load {bf}: Returned {ret}")
                fails.add(bf)
            case Ok(acts):
                actions.update(acts)

    if len(fails) > 0:
        return Err(fails)
    else:
        return Ok(actions)
