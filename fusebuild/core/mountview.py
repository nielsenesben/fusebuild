import json
import sys
from pathlib import Path

import marshmallow_dataclass2

from .action import Action, ActionLabel
from .libfusebuild import (
    AccessRecorder,
    BasicMount,
    action_dir,
    action_folder_root_str,
    new_access_log_file,
    output_folder_root_str,
)
from .logger import getLogger

logger = getLogger(__name__)

# pull in some spaghetti to make this stuff work without fuse-py being installed
try:
    import _find_fuse_parts  # type: ignore
except ImportError:
    pass
import fuse  # type: ignore
from fuse import Fuse, FuseArgs, FuseError


def main(label: ActionLabel):
    schema = marshmallow_dataclass2.class_schema(Action)()
    action_dir_ = action_dir(label)
    action_file = (
        Path(str(output_folder_root_str + str(label[0])))
        / "FUSEBUILD.py"
        / (label[1] + ".json")
    )
    with action_file.open("r") as f:
        d = json.load(f)
        action = schema.load(d)

    logger.debug(f"{action=}")
    access_log = new_access_log_file(label)
    mountpoint = action_dir_ / "mountpoint"
    writeable = Path(output_folder_root_str + str(label[0])) / label[1]
    logger.debug(f"Mounting on {mountpoint=} with {writeable=}")
    with access_log.open("w", encoding="utf-8") as access_log_file:
        logger.debug(f"{access_log=}")
        access_recorder = AccessRecorder(
            central_dir=label[0],
            access_log=access_log_file,
            action_deps_file=action_dir_ / "action_deps.txt",
        )
        usage = (
            """
            """
            + Fuse.fusage
        )

        fuse_server = BasicMount(
            label,
            mountpoint,
            access_recorder,
            version="%prog " + fuse.__version__,
            usage=usage,
            dash_s_do="setsingle",
            writeable=str(writeable),
            mappings=action.mappings,
        )
        res = fuse_server.main()
        logger.info(f"Fuse main returned {res}")
        access_log_file.flush()
        access_log_file.close()
        access_recorder.flush()

    logger.info(f"mountview for {label} done")


if __name__ == "__main__":
    sys.exit(main((Path(sys.argv[1]), sys.argv[2])))
