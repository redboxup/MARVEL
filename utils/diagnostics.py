import os
import signal
import psutil
import faulthandler
from pathlib import Path


def setup_faultlogger(log_path: Path):
    """Register SIGUSR1 to dump tracebacks to the given log file"""

    # ensure log file exists
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path} directory not found")

    # open file
    log_file = log_path / "traceback.log"
    f = open(log_file, "a")

    # enable faulthandler globally
    faulthandler.enable(f, all_threads=True)

    # register SIGUSR1 to write stack trace to this file
    faulthandler.register(signum=signal.SIGUSR1, file=f, all_threads=True)

    print(f"[faulthandler] SIGUSR1 handler registered; dumping to {log_file}")


def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)  # in MB
