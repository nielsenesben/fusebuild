import os
import subprocess
import sys

from fusebuild.core.logger import getLogger

logger = getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

# os.setpgrp()

cwd = sys.argv[1]

total_cmd = sys.argv[2:]

logger.info(f"{' '.join(total_cmd)=}")
res = subprocess.run(total_cmd, cwd=cwd)
sys.exit(res.returncode)
