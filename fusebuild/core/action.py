import re
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import marshmallow_dataclass2
from marshmallow import Schema, fields

from fusebuild.core.logger import getLogger

logger = getLogger(__name__)
# logger.setLevel(logging.DEBUG)


@dataclass(frozen=True)
class TmpDir:
    name: str


@dataclass(frozen=True)
class RandomTmpDir:
    pass


TmpStrategy: TypeAlias = TmpDir | RandomTmpDir | None

SandboxOptions = None | Any


class Sandbox(Protocol):
    def generate_command(
        self, mountpoint: Path, cwd: Path, orig_cmd: list[str], environ: dict[str, str]
    ) -> tuple[list[str], dict[str, str]]: ...


@dataclass
class NoSandbox(Sandbox):
    def generate_command(
        self, mountpoint: Path, cwd: Path, orig_cmd: list[str], environ: dict[str, str]
    ) -> tuple[list[str], dict[str, str]]:
        new_environ = deepcopy(environ)
        new_environ["OUTPUT_DIR"] = str(mountpoint) + "/" + environ["OUTPUT_DIR"]
        return [
            "python3",
            str(Path(__file__).parent / "spawn_jump.py"),
            str(mountpoint) + "/" + str(cwd),
        ] + orig_cmd, new_environ


@dataclass(frozen=True)
class BwrapSandbox(Sandbox):
    run_as_root: bool = False

    def generate_command(
        self, mountpoint: Path, cwd: Path, orig_cmd: list[str], environ: dict[str, str]
    ) -> tuple[list[str], dict[str, str]]:
        if "TMPDIR" in environ:
            tmp_dir = ["--tmpfs", environ["TMPDIR"]]
        else:
            tmp_dir = []

        if self.run_as_root:
            unshare_cmd = ["unshare", "-r"]
        else:
            unshare_cmd = []

        total_cmd = (
            unshare_cmd
            + [
                "bwrap",
                "--die-with-parent",
                "--bind",
                str(mountpoint),
                "/",
                "--dev-bind",
                "/dev",
                "/dev",
                "--proc",
                "/proc",
            ]
            + tmp_dir
            + ["--chdir", str(cwd)]
            + orig_cmd
        )

        return total_cmd, environ


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def get_definition(name: str):
    parts = name.rsplit(".", 1)
    return getattr(import_module(parts[0]), parts[1])


class ProtocolField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs) -> dict[str, Any] | None:
        if value is None:
            return None
        schema = marshmallow_dataclass2.class_schema(value.__class__)()
        return {"type": fullname(value), "value": schema.dump(value)}

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return None

        if isinstance(value, str):
            value_dict = json.loads(value)
        else:
            value_dict = value

        if "type" not in value_dict:
            raise ValueError(f"'type' is missing when trying to recover {value_dict}")

        definition = get_definition(value["type"])

        if "value" not in value_dict:
            raise ValueError(f"'value' is missing when trying to recover {value_dict}")
        schema = marshmallow_dataclass2.class_schema(definition)()
        return schema.load(value_dict["value"])


class ProtocolList(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        ret = []
        for v in value:
            schema = marshmallow_dataclass2.class_schema(v.__class__)()
            ret.append({"type": fullname(v), "value": schema.dump(v)})

        return ret

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return None

        if isinstance(value, str):
            value_list = json.loads(value)
        else:
            value_list = value

        res = []
        for v in value_list:
            if "type" not in v:
                raise ValueError(f"'type' is missing when trying to recover {v}")

            definition = get_definition(v["type"])

            if "value" not in v:
                raise ValueError(f"'value' is missing when trying to recover {v}")
            schema = marshmallow_dataclass2.class_schema(definition)()
            res.append(schema.load(v["value"]))

        return res


ActionLabel: TypeAlias = tuple[Path, str]


class Mapping(Protocol):
    def remap(self, virtual_path: Path, is_output: bool) -> Path | None: ...


class MappingDefinition(Protocol):
    def create(self, output_folder: Path) -> Mapping: ...


class PatternRemapOutput(Mapping):
    def __init__(self, regexp: str, output: str) -> None:
        self.pattern = re.compile(regexp)
        self.output = output

    def remap(self, path: Path, is_output: bool) -> Path | None:
        if is_output:
            return None
        str_path = str(path)
        str_res, count = re.subn(self.pattern, self.output, str_path)
        logger.debug(f"remap {self.pattern=} {str_path=} {count=}")
        if count == 0:
            return None
        else:
            logger.debug(f"pattern remap {path} -> {str_res}")
            return Path(str_res)


@dataclass
class PatternRemapToOutput(MappingDefinition):
    regexp: str
    output: str

    def create(self, output_folder: Path) -> Mapping:
        return PatternRemapOutput(self.regexp, str(output_folder) + self.output)


@dataclass
class Provider:
    # output_dir: str = field(metadata={"marshmallow": {"dump_by": lambda x: None}})
    pass


class ProtocolDict(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs) -> dict[str, Any] | None:
        if value is None:
            return None
        ret = {}
        for k, v in value.items():
            schema = marshmallow_dataclass2.class_schema(v.__class__)()
            ret[k] = {"type": fullname(v), "value": schema.dump(v)}

        return ret

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return {}

        if isinstance(value, str):
            value_dict = json.loads(value)
        else:
            value_dict = value

        res = {}
        for k, v in value_dict.items():
            if "type" not in v:
                raise ValueError(f"'type' is missing when trying to recover {v}")
            definition = get_definition(v["type"])

            if "value" not in v:
                raise ValueError(f"'value' is missing when trying to recover {v}")
            schema = marshmallow_dataclass2.class_schema(definition)()
            res[k] = schema.load(v["value"])

        return res


@dataclass
class Action:
    cmd: list[str]
    category: str
    tmp: TmpStrategy = TmpDir("/tmp")
    sandbox: Sandbox = field(
        metadata={"marshmallow_field": ProtocolField()}, default=BwrapSandbox()
    )
    mappings: list[MappingDefinition] = field(
        metadata={"marshmallow_field": ProtocolList()}, default_factory=list
    )
    # A dictionary for stuff of how other actions can interpret the output
    # See for instance containers/containerprovider.py
    # Paths are always relative to the output folder of the action
    providers: dict[str, Provider] = field(
        metadata={"marshmallow_field": ProtocolDict()}, default_factory=dict
    )


@dataclass
class ActionStatus:
    success: bool
    rerun: bool


def label_from_line(line: str):
    """Assumes action names doesn't contain / - might change"""
    last_slash = line.rfind("/")
    return (Path(line[0:last_slash]), line[last_slash + 1 :].rstrip())
