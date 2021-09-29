import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from nlp_app_samples.constants import APP_NAME

LOGGING_LEVELS = {
    0: logging.NOTSET,
    1: logging.ERROR,
    2: logging.WARN,
    3: logging.INFO,
    4: logging.DEBUG,
}

FORMATTING_STRING = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
FORMATTER = logging.Formatter(FORMATTING_STRING)
LOG_FILE = f'{APP_NAME}_logs.log'


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def resolve_logging_level(logging_level:int):
    if logging_level > 0:
        level = LOGGING_LEVELS[logging_level] if logging_level in LOGGING_LEVELS else logging.DEBUG
    else:
        level = logging.NOTSET
    return level


def get_logger(logger_name, logging_level:int = 3):
    logger = logging.getLogger(logger_name)
    logger.setLevel(resolve_logging_level(logging_level))
    logger.addHandler(get_console_handler())
    logger.propagate = False
    return logger


def init_logging():
    logging.basicConfig(level=logging.INFO, format=FORMATTING_STRING)

