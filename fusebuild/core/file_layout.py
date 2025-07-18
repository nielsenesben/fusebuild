import os
from pathlib import Path

import fusebuild.core.logger as logger_module

from .action import ActionLabel

logger = logger_module.getLogger(__name__)

FUSEBUILD_CACHE_DIR = "FUSEBUILD_CACHE_DIR"
if FUSEBUILD_CACHE_DIR in os.environ:
    fusebuild_folder = Path(os.environ[FUSEBUILD_CACHE_DIR])
else:
    fusebuild_folder = Path.home() / ".cache" / "fusebuild"

output_folder_root = fusebuild_folder / "output"
output_folder_root_str = str(output_folder_root)
action_folder_root = fusebuild_folder / "actions"
action_folder_root_str = str(action_folder_root)


def action_dir(label: ActionLabel):
    return Path(action_folder_root_str + str(label[0])) / label[1]


def output_dir(label: ActionLabel):
    return Path(output_folder_root_str + str(label[0])) / label[1]


def is_rule_output(path: str) -> bool:
    res = os.path.commonprefix([output_folder_root_str, path]) == output_folder_root_str
    logger.debug(f"is_rule_output {path}: {res}")
    return res
