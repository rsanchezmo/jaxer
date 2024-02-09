import os
import logging

DEFAULT_LOGGER_NAME = 'jaxer'
LOGGER_NAME = os.getenv('LOGGER_NAME', DEFAULT_LOGGER_NAME)


def get_logger():
    """Provide the LOGGER_NAME logger. If is not the default logger,
        it means that the user has already managed to configure the logger, so we don't do anything. If it is the
        default logger, we configure it with a StreamHandler and a default formatter. If the logger is already
        configured, we just return it.
    """

    if LOGGER_NAME not in logging.Logger.manager.loggerDict.keys():
        logger = logging.getLogger(LOGGER_NAME)

        if LOGGER_NAME == DEFAULT_LOGGER_NAME:
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
                ch.setFormatter(formatter)
                logger.addHandler(ch)
    else:
        logger = logging.getLogger(LOGGER_NAME)

    return logger
