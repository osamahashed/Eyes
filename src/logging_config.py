import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(config):
    logging_cfg = config["logging"]
    log_level = getattr(logging, logging_cfg["level"].upper(), logging.INFO)
    log_dir = logging_cfg.get("directory", "logs")
    log_file = logging_cfg.get("filename", "eyemouse.log")
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
