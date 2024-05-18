import subprocess
import threading
from pathlib import Path


def execute_command(cmd: list[str], timeout: int | None = None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timerObject = None
    try:
        if timeout:

            def killProcess():
                p.kill()

            timerObject = threading.Timer(timeout, killProcess)
            timerObject.start()

        out, err = p.communicate()
        exitCode = p.wait()
    finally:
        if timerObject is not None:
            timerObject.cancel()

    return out, err, exitCode


cmd = ["python", "main.py", "wav"]

test_dir = Path(__file__).parent

cmd += ["--wav-file", str(test_dir / "test.wav")]

out, err, exitCode = execute_command(cmd, timeout=20)

err = err.decode("utf-8")
print(err)

assert "Turning light 6 on" in err
assert "Turning light 8 on" in err
