# To be used from FUSEBUILD.py files to load actions
import inspect
import json
from pathlib import Path
from typing import Any

import marshmallow_dataclass2

from .action import Action, ActionLabel, BwrapSandbox, Provider, TmpDir, TmpStrategy
from .file_layout import output_dir, output_folder_root_str
from .libfusebuild import load_action_file, loaded_actions
from .logger import getLogger

logger = getLogger(__name__)


def _action(name: str, cmd: list[str], **kwargs: Any) -> Action:
    frame = inspect.currentframe()

    while True:
        assert frame is not None
        path = Path(frame.f_code.co_filename)
        if path.name == "FUSEBUILD.py":
            directory = path.parent
            break
        else:
            frame = frame.f_back

    label = ActionLabel(directory, name)

    if "tmp" in kwargs:
        tmp = kwargs["tmp"]
    else:
        tmp = TmpDir("/tmp")

    if "sandbox" in kwargs:
        sandbox = kwargs["sandbox"]
    else:
        sandbox = BwrapSandbox()

    if "mappings" in kwargs:
        mappings = kwargs["mappings"]
    else:
        mappings = []

    providers: dict[str, Provider] = kwargs.get("providers", {})
    for k, p in providers.items():
        p.output_dir = output_dir(label)
    category = kwargs.get("category", "build")

    action = Action(
        cmd,
        category=category,
        tmp=tmp,
        sandbox=sandbox,
        mappings=mappings,
        providers=providers,
    )
    loaded_actions[label] = action
    return action


def action(name: str, cmd: list[str], **kwargs: Any) -> Action:
    return _action(name, cmd, **kwargs)


def shell_action(name: str, cmd: str, **kwargs: Any) -> Action:
    return _action(name=name, cmd=["bash", "-e", "-c", cmd], **kwargs)


def write_actions() -> Any:
    schema = marshmallow_dataclass2.class_schema(Action)()
    for label, action in loaded_actions.items():
        action_path = Path(
            output_folder_root_str
            + str(label.path / "FUSEBUILD.py" / label.name)
            + ".json"
        )
        action_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"{action_path=}")
        with action_path.open("w") as f:
            json.dump(schema.dump(action), f)


def get_action(path: Path | str, name: str) -> Action:
    path = Path(path).resolve()
    label = ActionLabel(path, name)
    if label in loaded_actions:
        return loaded_actions[label]
    action = load_action_file(
        label, output_dir(ActionLabel(path, "FUSEBUILD.py")) / f"{name}.json"
    )
    return action
