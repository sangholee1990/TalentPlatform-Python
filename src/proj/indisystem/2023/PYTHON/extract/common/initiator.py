import logging


def init_logger(log_path):
    """
    Init Logger.
    :param config_manager: Config Manager.
    :return:
    """
    from .logger import register_logger
    global logger
    logger = register_logger(logging.INFO, log_path)
    logger.info("start")