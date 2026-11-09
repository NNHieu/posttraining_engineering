from __future__ import annotations

import logging
from pathlib import Path


# Configure logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
# ch.setFormatter(formatter)
# logger.addHandler(ch)

_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
_HANDLER_MARKER = "_nano_rlvr_handler"


def setup_logging(
    *,
    rank: int = 0,
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure process logging. Rank 0 writes INFO logs to console and run.log;
    other ranks show warnings and errors on the console.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove only handlers installed by this function. This makes setup safe
    # to call again without duplicating output or disturbing other handlers.
    for handler in root.handlers[:]:
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler()
    console.setLevel(level if rank == 0 else logging.WARNING)
    console.setFormatter(logging.Formatter(_FORMAT))
    setattr(console, _HANDLER_MARKER, True)
    root.addHandler(console)

    if rank == 0 and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        setattr(file_handler, _HANDLER_MARKER, True)
        root.addHandler(file_handler)

    return logging.getLogger("nano_rlvr")