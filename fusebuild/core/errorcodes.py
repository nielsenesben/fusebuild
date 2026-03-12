from enum import Enum


class ErrorCode(Enum):
    SUCCESS = 0
    INTERNAL_ERROR = 1
    INVALID_INPUT = 2
    ACTION_FAILED = 3
    DEADLOCK = 4
    BUILDFILE_FAILED = 5
