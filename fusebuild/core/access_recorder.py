import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from errno import *
from io import TextIOWrapper
from pathlib import Path
from stat import *
from threading import Lock
from typing import Any, Callable

from marshmallow_dataclass2 import class_schema

import fusebuild.core.logger as logger_module

from .action import ActionLabel, label_from_line
from .action_invoker import ActionInvoker
from .dependency import (
    AccessType,
    ActionSetupRecorder,
    DependencyIndex,
    DependencyRecord,
    DependencyValue,
    StatRecordDir,
    StatRecordFile,
    check_file_hash,
    get_file_hash,
)
from .file_layout import (
    access_log_file,
    action_deps_file,
    is_rule_output,
    new_access_log_file,
    output_folder_root_str,
    tmp_access_log_file,
)

logger = logger_module.getLogger(__name__)

dependency_record_schema = class_schema(DependencyRecord)()


def do_nothing() -> None:
    pass


@dataclass(frozen=False)
class AccessRecorder:
    central_dir: Path
    access_log: TextIOWrapper
    action_deps_file: Path
    accesses: dict[DependencyIndex, DependencyValue] = field(default_factory=dict)
    listener: Any = do_nothing
    waiting_for: set[tuple[Path, str]] = field(default_factory=set)
    action_deps: set[ActionLabel] = field(default_factory=set)
    mutex: Lock = Lock()

    def __post_init__(self) -> None:
        self.action_deps = set(load_action_deps(self.action_deps_file))
        self.mutex = Lock()

    def _report_change(self) -> None:
        self.listener()

    def _record_path(self, path: str | Path) -> tuple[bool, str]:
        if type(path) is not str:
            path = str(path)
        assert type(path) is str
        if is_rule_output(path):
            return (
                True,
                os.path.relpath(path, output_folder_root_str + str(self.central_dir)),
            )
        else:
            return (False, os.path.relpath(path, self.central_dir))

    @staticmethod
    def _stat_records(stat: os.stat_result) -> DependencyValue:
        if (stat.st_mode & S_IFDIR) != 0:
            return StatRecordDir(stat.st_mode)
        else:
            return StatRecordFile(stat.st_mode, stat.st_size)

    def write_entry(self, index: DependencyIndex, value: DependencyValue) -> None:
        with self.mutex:
            if index in self.accesses:
                if self.accesses[index] == value:
                    # No need to write same entry more than once
                    return
                if self.accesses[index] is None:
                    # Is already marked as changed-under-build:
                    return
                logger.error(
                    f"Dependency changed while building for {index}: {value} != {self.accesses[index]}"
                )
                # Inserted value which will not match real => rebuild next time
                value = None
            self.accesses[index] = value
            record = DependencyRecord(index, value)
            record_js = dependency_record_schema.dump(record)
            js = json.dumps(record_js, ensure_ascii=True)
            self.access_log.write(js + "\n")

    def record_access(self, path: str, mode: int, res: int) -> None:

        self.write_entry(
            DependencyIndex(self._record_path(path), AccessType.ACCESS, mode), res
        )

    def record_stat(self, path: str, res: os.stat_result) -> None:
        self.write_entry(
            DependencyIndex(self._record_path(path), AccessType.STAT),
            self._stat_records(res),
        )

    @staticmethod
    def _read_dir(entries: list[str]) -> str:
        entries.sort()
        sha = hashlib.sha256()
        for f in entries:
            sha.update(f.encode("UTF-8"))

        return sha.hexdigest()

    def record_readdir(self, path: str, to_record: list[str]) -> None:
        self.write_entry(
            DependencyIndex(self._record_path(path), AccessType.READDIR),
            self._read_dir(to_record),
        )

    @staticmethod
    def _read_file_check(path: Path, expected: str) -> bool:
        stat = os.lstat(path)

        if not check_file_hash(path, stat, expected):
            return False

        return True

    def record_read(self, path: str) -> None:
        is_output, record_path = self._record_path(path)
        if not is_output:
            stat = os.lstat(path)
            hash = get_file_hash(path, stat)
            self.write_entry(
                DependencyIndex((is_output, record_path), AccessType.READ), hash
            )

    def record_readlink(self, path: Path | str, res: str) -> None:
        is_output, record_path = self._record_path(path)
        if not is_output:
            self.write_entry(
                DependencyIndex((is_output, record_path), AccessType.READLINK), res
            )

    def record_dir_exists(self, path: Path, exists: bool) -> None:
        is_output, record_path = self._record_path(path)
        assert not is_output
        self.write_entry(
            DependencyIndex((is_output, record_path), AccessType.DIR_EXISTS), exists
        )

    @staticmethod
    def _read_link_check(path: Path, res: str) -> bool:
        rl = os.readlink(path)
        logger.debug(f"{rl=}")
        return str(rl) == res

    @staticmethod
    def _dir_exists_check(path: Path, expected: bool) -> bool:
        actual = path.is_dir()
        logger.debug(f"{expected=} {actual=}")
        return actual == expected

    def flush(self) -> None:
        with self.action_deps_file.open("w") as f:
            for ad in self.action_deps:
                f.write(f"{ad}\n")


def check_accesses_inner_loop(
    label: ActionLabel,
    check_build_target: Callable[
        [Path, ActionInvoker], tuple[bool, ActionLabel | None]
    ],
    invoker: ActionInvoker,
) -> bool:
    matches = True
    central_dir = label.path.absolute()
    access_log = access_log_file(label)
    with access_log.open("r", encoding="utf-8") as f:
        for line in f:
            if not matches:
                break
            d: DependencyRecord = dependency_record_schema.load(json.loads(line))
            key = d.index
            expected = d.value
            src_path = (central_dir / key.path[1]).absolute()
            if not key.path[0]:
                path = src_path
            else:
                path = Path(output_folder_root_str + str(src_path))
                if src_path.is_dir():
                    if not path.is_dir():
                        matches = False
                        break
                else:
                    check_build_target(src_path, invoker)

            logger.debug(f"Checking {path} for {key}")
            match key.access_type:
                case AccessType.ACCESS:
                    if os.access(path, key.access_mode) != expected:
                        logger.info(f"access({path}, {key}) != {expected}")
                        print(f"{path} access {key.access_mode} changed")
                        matches = False
                        break
                case AccessType.STAT:
                    if AccessRecorder._stat_records(os.lstat(path)) != expected:
                        logger.info(f"lstat({path}) {os.lstat(path)} != {expected}")
                        print(f"{path} stat changed")
                        matches = False
                        break
                case AccessType.READDIR:
                    files = list(os.listdir(path))
                    if AccessRecorder._read_dir(files) != expected:
                        logger.info(f"readdir({path}) != {expected}")
                        print(f"readdir of {path} changed")
                        matches = False
                        break
                case AccessType.READ:
                    assert isinstance(expected, str)
                    if not AccessRecorder._read_file_check(path, expected):
                        logger.info(f"read({path}) != {expected}")
                        print(f"{path} contents changed")
                        matches = False
                        break
                case AccessType.READLINK:
                    assert isinstance(expected, str)
                    if not AccessRecorder._read_link_check(path, expected):
                        logger.info(f"readlink({path}) != {expected}")
                        print(f"{path} readlink changed")
                        matches = False
                        break
                case AccessType.DIR_EXISTS:
                    expected = bool(expected)
                    assert isinstance(expected, bool)
                    if not AccessRecorder._dir_exists_check(path, expected):
                        logger.info(f"is_dir({path}) != {expected}")
                        if expected:
                            print(f"Directory {path} is no longer a directory.")
                        else:
                            print(f"New directory {path}")

                        matches = False
                        break
                case _:
                    logger.error("Unknown access: {key=} {expected=}")
                    matches = False
                    break

    return matches


def merge_access_logs(label: ActionLabel) -> None:
    access_log = access_log_file(label)
    new_access_log = new_access_log_file(label)
    if new_access_log.exists():
        if access_log.exists():
            tmp_access_log = tmp_access_log_file(label)
            ret = subprocess.run(
                f"cat {access_log} {new_access_log} | sort -u > {tmp_access_log}",
                shell=True,
            )
            assert ret.returncode == 0
            tmp_access_log.rename(access_log)
            new_access_log.unlink()
        else:
            new_access_log.rename(access_log)


def check_accesses(
    label: ActionLabel,
    check_build_target: Callable[
        [Path, ActionInvoker], tuple[bool, ActionLabel | None]
    ],
    invoker: ActionInvoker,
) -> bool:
    merge_access_logs(label)
    try:
        return check_accesses_inner_loop(label, check_build_target, invoker)
    except FileNotFoundError as e:
        logger.debug(f"Got {e=}")
        return False
    except json.decoder.JSONDecodeError as e:
        logger.warning(f"Got decodig exception {e}")
        return False
    except Exception as e:
        logger.error(f"Got exception {e=} while parsing access log")
        print(traceback.format_exc())
        sys.exit(1)
        return False


def load_action_deps(label: ActionLabel | Path) -> list[ActionLabel]:
    if isinstance(label, Path):
        txt_file = label
    else:
        txt_file = action_deps_file(label)

    if txt_file.exists():
        with txt_file.open("r") as f:
            return [label_from_line(line) for line in f.readlines()]
    else:
        return []
