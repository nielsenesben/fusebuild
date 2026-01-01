import logging
import os
import sys

logging.basicConfig(format="%(process)d %(filename)s %(lineno)d: %(message)s")


def setLoggerPrefix(prefix: str) -> None:
    logging.basicConfig(
        format=f"%(process)d {prefix} %(filename)s %(lineno)d: %(message)s", force=True
    )


FUSEBUILD_LOG_LEVEL = "FUSEBUILD_LOG_LEVEL"


def getLogger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if FUSEBUILD_LOG_LEVEL in os.environ:
        log_level = int(os.environ[FUSEBUILD_LOG_LEVEL])
        logger.setLevel(log_level)
        logger.info(
            f"{os.environ[FUSEBUILD_LOG_LEVEL]} {logging.getLevelName(log_level)} {sys.argv}"
        )
    else:
        logger.setLevel(logging.WARN)
        assert "FUSEBUILD_INVOCATION_DIR" not in os.environ
    return logger
