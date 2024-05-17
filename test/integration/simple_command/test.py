import subprocess
from pathlib import Path
import threading


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

assert (
    "** Inference result:  Can you turn the office light on? #command: turn office01 on#"
    in err.decode("utf-8")
)
