# -*- coding: utf-8 -*-

import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
import socket

backup_count = 10
formatter = None
ext_formatter = None
int_formatter = None

def init_formatter():
    """
    로그 포멧을 정의한다
    :param tp_id: Tp_Id.
    :return:
    """
    global formatter, ext_formatter, int_formatter
    hostname = socket.gethostname()
    formatter = logging.Formatter('[%(asctime)s | Decoder-Model | TP-{} | PRO | %(levelname)s | ' '%(filename)s:%(lineno)s] > %(message)s'.format(hostname))
    ext_formatter = logging.Formatter('[%(asctime)s | Decoder-Model | {} | EXT | %(levelname)s | ' '%(filename)s:%(lineno)s] > %(message)s'.format(hostname))
    int_formatter = logging.Formatter('[%(asctime)s | Decoder-Model | {} | INT | %(levelname)s | ' '%(filename)s:%(lineno)s] > %(message)s'.format(hostname))


def register_logger(level, path=None):
    """
    로그를 저장한다
    :param tp_id: 작업 명
    :param level: 로그 수준
    :param path: 로깅 위치
    :return: 로거
    """
    logger = logging.getLogger("Decoder-Model")
    logger.setLevel(level)
    init_formatter()

    if not logger.handlers:
        stream_handler = create_handler(level)
        file_handler = create_handler(level, path)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


def create_handler(level, path=None):
    """
    Create Handler.
    :param level: Level.
    :param path: Path.
    :return: File handler.
    """
    if path is None:
        stream_handler = StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        return stream_handler
    else:
        file_handler = TimedRotatingFileHandler(path, when='midnight', interval=1, backupCount=backup_count, encoding="UTF-8")
        file_handler.suffix = "%Y%m%d"
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        return file_handler