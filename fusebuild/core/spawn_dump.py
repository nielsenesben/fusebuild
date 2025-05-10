import subprocess
import sys

cwd = sys.argv[1]
subprocess.run(sys.argv[2:], cwd=cwd)
