import argparse
import sys
import subprocess
from pathlib import Path

argp = argparse.ArgumentParser(description="Integration test driver")

argp.add_argument(
    "--test-dir",
    type=Path,
)

args = argp.parse_args()

assert args.test_dir.is_absolute()

for entry in args.test_dir.rglob("**/*.py"):
    if "execute.py" in Path(entry).name:
        continue

    cmd = ["python", str(entry)]

    print("**** Executing: ", " ".join(cmd))
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print("**** ERROR ****")
        sys.exit(1)
    else:
        print("**** PASS ****")
