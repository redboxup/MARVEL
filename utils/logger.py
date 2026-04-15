import logging
from typing import List, Any
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(
    log_dir: Path, verbose: bool = True, name: str = ""
) -> logging.Logger:
    """Set up logging with RichHandler and FileHandler."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        log_dir
        / f"{name or 'training'}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    handlers: List[Any] = [file_handler]
    if verbose:
        handlers.append(RichHandler(rich_tracebacks=True, markup=True))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger()
    return logger
