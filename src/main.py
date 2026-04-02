import logging
import sys

from app_controller import AppController
from config import Config
from logging_config import setup_logging
from qt_compat import QApplication


def main():
    config = Config("config/default_config.json")
    setup_logging(config.data)
    logger = logging.getLogger(__name__)

    app = QApplication(sys.argv)
    controller = AppController(config)
    controller.start()

    try:
        exit_code = app.exec()
    except Exception as exc:
        logger.exception("Fatal application error: %s", exc)
        raise
    finally:
        controller.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
