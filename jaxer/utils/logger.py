import os
import logging

DEFAULT_LOGGER_NAME = 'JAXER'
LOGGER_NAME = os.getenv('LOGGER_NAME', DEFAULT_LOGGER_NAME)


def get_logger():
    """Provides the LOGGER_NAME logger. If is not the default logger, it means that the user has already managed to
    configure the logger with a env variable: 'LOGGER_NAME'. If it is the default logger,
    we configure it with a StreamHandler and a default formatter. If the logger is already configured, we just return
    it (as a singleton).

    :return: logging.Logger
    """

    if LOGGER_NAME not in logging.Logger.manager.loggerDict.keys():
        logger = logging.getLogger(LOGGER_NAME)

        if LOGGER_NAME == DEFAULT_LOGGER_NAME:
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                formatter = logging.Formatter('[%(name)s - %(levelname)s]: %(message)s')
                ch.setFormatter(formatter)
                logger.addHandler(ch)
    else:
        logger = logging.getLogger(LOGGER_NAME)

    return logger
