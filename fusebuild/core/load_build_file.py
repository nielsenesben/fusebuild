import fusebuild.core.actions as actions
from pathlib import Path
import importlib
import sys
from .logger import getLogger

logger = getLogger(__name__)


def load_build_file(buildfile: Path) -> bool:
    logger.debug(f"Loading {buildfile}")
    spec = importlib.util.spec_from_file_location("SBUILD", buildfile)
    assert spec is not None
    sbuild_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(sbuild_module)
    return True


def main(args: list[str]) -> int:
    build_file = Path(args[1])
    if load_build_file(build_file):
        actions.write_actions()
        logger.debug(f"All actions written successfully for {build_file=}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
