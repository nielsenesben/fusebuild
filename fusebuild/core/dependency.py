import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .action import Action
from .logger import getLogger

logger = getLogger(__name__)


@dataclass
class ActionSetupRecorder:
    definition: Action
    environment: dict[str, str]
    return_code: int | None = None


class AccessType(Enum):
    READ = 1
    ACCESS = 2
    STAT = 3
    READDIR = 4
    READLINK = 5
    DIR_EXISTS = 6


@dataclass
class StatRecordDir:
    mode: int


@dataclass
class StatRecordFile:
    mode: int
    size: int


@dataclass(frozen=True)
class DependencyIndex:
    path: tuple[bool, str]
    access_type: AccessType
    access_mode: int = 0


# None mean changed under build - should force a rebuild next time
DependencyValue = str | StatRecordDir | StatRecordFile | int | bool | None


@dataclass
class DependencyRecord:
    index: DependencyIndex
    value: DependencyValue


StatChangedFieldsType = tuple[int, int, float, float]


def stat_result_to_record(stat: os.stat_result) -> StatChangedFieldsType:
    return (stat.st_ino, stat.st_size, stat.st_ctime, stat.st_mtime)


_hashed: dict[str, tuple[StatChangedFieldsType, str]] = {}


def _get_cached_hash(record_path: str, stat: os.stat_result) -> str | None:
    if record_path not in _hashed:
        return None

    old = _hashed[record_path]
    if old[0] == stat_result_to_record(stat):
        return old[1]
    else:
        return None


def store_hash(record_path: str, stat: os.stat_result, hash: str) -> None:
    _hashed[record_path] = (stat_result_to_record(stat), hash)


def _file_hash(path: str | Path, stat: os.stat_result) -> str:
    if isinstance(path, Path):
        path = str(path)

    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(path, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    hash = h.hexdigest()
    store_hash(path, stat, hash)
    return hash


def get_file_hash(path: str | Path, stat: os.stat_result) -> str:
    if isinstance(path, Path):
        path = str(path)

    hash = _get_cached_hash(path, stat)
    if hash is not None:
        return hash

    return _file_hash(path, stat)


def check_file_hash(path: str | Path, stat: os.stat_result, expected: str) -> bool:
    if isinstance(path, Path):
        path = str(path)

    hash = _get_cached_hash(path, stat)
    if hash is not None and hash == expected:
        return True

    hash = _file_hash(path, stat)
    logger.debug(f"File hash check for {path}: {hash}=?{expected}")
    return hash == expected
