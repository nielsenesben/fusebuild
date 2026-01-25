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

import fcntl
import os
import subprocess
from errno import EACCES, EINVAL, EOPNOTSUPP
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Optional, Tuple, Union

from fusebuild.core.access_recorder import AccessRecorder
from fusebuild.core.action import ActionLabel, MappingDefinition
from fusebuild.core.action_invoker import ActionInvoker
from fusebuild.core.file_layout import (
    is_rule_output,
    output_folder_root,
    output_folder_root_str,
    subbuild_failed_file,
)

# pull in some spaghetti to make this stuff work without fuse-py being installed
try:
    import _find_fuse_parts  # type: ignore
except ImportError:
    pass
import fuse  # type: ignore
from fuse import Fuse, FuseArgs

if not hasattr(fuse, "__version__"):
    raise RuntimeError(
        "your fuse-py doesn't know of fuse.__version__, probably it's too old."
    )

fuse.fuse_python_api = (0, 2)

fuse.feature_assert("stateful_files", "has_init")


import fusebuild.core.logger as logger_module

logger = logger_module.getLogger(__name__)


def flag2mode(flags: int) -> str:
    md = {os.O_RDONLY: "rb", os.O_WRONLY: "wb", os.O_RDWR: "wb+"}
    m = md[flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR)]

    if flags | os.O_APPEND:
        m = m.replace("w", "a", 1)

    return m


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
        # Import here to avoid circular dependency
        from fusebuild.core.libfusebuild import check_build_target

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
                with subbuild_failed_file(self.label).open("a") as f:
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

    def readdir(self, path_in: Path, offset: int) -> Iterable[os.DirEntry[str]]:
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
