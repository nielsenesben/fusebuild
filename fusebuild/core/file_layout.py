import functools
import os
from pathlib import Path

import fusebuild.core.logger as logger_module

from .action import ActionLabel

logger = logger_module.getLogger(__name__)

FUSEBUILD_CACHE_DIR = "FUSEBUILD_CACHE_DIR"
FUSEBUILD_INVOCATION_DIR = "FUSEBUILD_INVOCATION_DIR"

if FUSEBUILD_CACHE_DIR in os.environ:
    fusebuild_folder = Path(os.environ[FUSEBUILD_CACHE_DIR])
else:
    fusebuild_folder = Path.home() / ".cache" / "fusebuild"

output_folder_root = fusebuild_folder / "output"
output_folder_root_str = str(output_folder_root)
action_folder_root = fusebuild_folder / "actions"
action_folder_root_str = str(action_folder_root)


def action_dir(label: ActionLabel) -> Path:
    return Path(action_folder_root_str + str(label.path)) / label.name


def output_dir(label: ActionLabel) -> Path:
    return Path(output_folder_root_str + str(label.path)) / label.name


def is_rule_output(path: str | Path) -> bool:
    res = os.path.commonprefix([output_folder_root_str, path]) == output_folder_root_str
    logger.debug(f"is_rule_output {path}: {res}")
    return res


# Action directory files
def status_lock_file(label: ActionLabel) -> Path:
    return action_dir(label) / "status.lck"


def status_file(label: ActionLabel) -> Path:
    return action_dir(label) / "status.json"


def waiting_for_file(label: ActionLabel) -> Path:
    return action_dir(label) / "waiting_for"


def stdout_file(label: ActionLabel) -> Path:
    return action_dir(label) / "stdout"


def stderr_file(label: ActionLabel) -> Path:
    return action_dir(label) / "stderr"


def subbuild_failed_file(label: ActionLabel) -> Path:
    return action_dir(label) / "subbuild_failed"


def mountpoint_dir(label: ActionLabel) -> Path:
    return action_dir(label) / "mountpoint"


def last_definition_file(label: ActionLabel) -> Path:
    return action_dir(label) / "last_definition.json"


def access_log_file(label: ActionLabel) -> Path:
    return action_dir(label) / "access_log.json"


def new_access_log_file(label: ActionLabel) -> Path:
    return action_dir(label) / "new_access_log.json"


def action_deps_file(label: ActionLabel) -> Path:
    return action_dir(label) / "action_deps.txt"


def tmp_access_log_file(label: ActionLabel) -> Path:
    return action_dir(label) / "tmp_action_log.json"


def action_json_file(label: ActionLabel) -> Path:
    """Path to the action definition JSON file in the output directory."""
    return (
        Path(output_folder_root_str + str(label.path))
        / "FUSEBUILD.py"
        / (label.name + ".json")
    )


def has_invocation_dir() -> bool:
    """Check if FUSEBUILD_INVOCATION_DIR is set."""
    return FUSEBUILD_INVOCATION_DIR in os.environ


# Invocation tracking files
def invocation_ok_file(label: ActionLabel) -> Path:
    """Path to the success marker file for an action in the current invocation."""
    if not has_invocation_dir():
        raise RuntimeError("FUSEBUILD_INVOCATION_DIR not set")
    return (
        Path(os.environ[FUSEBUILD_INVOCATION_DIR])
        / "ok"
        / str(label.path).lstrip("/")
        / label.name
    )


def invocation_failed_file(label: ActionLabel) -> Path:
    """Path to the failure marker file for an action in the current invocation."""
    if not has_invocation_dir():
        raise RuntimeError("FUSEBUILD_INVOCATION_DIR not set")
    return (
        Path(os.environ[FUSEBUILD_INVOCATION_DIR])
        / "failed"
        / str(label.path).lstrip("/")
        / label.name
    )
