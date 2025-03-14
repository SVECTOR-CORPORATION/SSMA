import logging


def setup_logger(name, level=logging.INFO):
    """
    Set up a logger with the given name and level.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
