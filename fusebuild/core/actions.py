# To be used from FUSEBUILD.py files to load actions
from .libfusebuild import (
    output_folder_root_str,
    load_action_file,
    loaded_actions,
    output_dir,
)
from .action import Action, TmpStrategy, TmpDir, BwrapSandbox
import marshmallow_dataclass2
from pathlib import Path
import inspect
import json
from .logger import getLogger

logger = getLogger(__name__)


def _action(name: str, cmd: list[str], **kwargs) -> Action:
    frame = inspect.currentframe()

    while True:
        assert frame is not None
        p = Path(frame.f_code.co_filename)
        if p.name == "FUSEBUILD.py":
            directory = p.parent
            break
        else:
            frame = frame.f_back

    label = (directory, name)

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

    providers = kwargs.get("providers", {})
    for k, p in providers.items():
        p.output_dir = output_dir(label)  # type: ignore
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


def action(name: str, cmd: list[str], **kwargs) -> Action:
    return _action(name, cmd, **kwargs)


def shell_action(name: str, cmd: str, **kwargs) -> Action:
    return _action(name=name, cmd=["bash", "-e", "-c", cmd], **kwargs)


def write_actions():
    schema = marshmallow_dataclass2.class_schema(Action)()
    for (path, name), action in loaded_actions.items():
        action_path = Path(
            output_folder_root_str + str(path / "FUSEBUILD.py" / name) + ".json"
        )
        action_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"{action_path=}")
        with action_path.open("w") as f:
            json.dump(schema.dump(action), f)


def get_action(path: Path | str, name: str) -> Action:
    path = Path(path).absolute()
    if (path, name) in loaded_actions:
        return loaded_actions[(path, name)]
    action = load_action_file(
        (path, name), output_dir((path, "FUSEBUILD.py")) / f"{name}.json"
    )
    return action
