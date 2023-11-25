import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[92m',  # green
        'DEBUG': '\033[94m',  # blue
        'WARNING': '\033[93m',  # yellow
        'ERROR': '\033[91m',  # red
        'CRITICAL': '\033[95m',  # purple
        'RESET': '\033[0m'  # reset
    }

    def format(self, record):
        log_message = super(ColoredFormatter, self).format(record)
        return f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}{log_message}{self.COLORS['RESET']}"


class Logger:
    def __init__(self, name="LOGGER"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = ColoredFormatter(f'[%(levelname)s - {name}]: %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    @staticmethod
    def _reset_color():
        print('\033[0m', end='')

    def set_status(self, status: bool = True):
        if status:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.CRITICAL + 1)

    def info(self, *args):
        self._reset_color()
        self.logger.info(" ".join(map(str, args)))

    def debug(self, *args):
        self._reset_color()
        self.logger.debug(" ".join(map(str, args)))

    def warning(self, *args):
        self._reset_color()
        self.logger.warning(" ".join(map(str, args)))

    def error(self, *args):
        self._reset_color()
        self.logger.error(" ".join(map(str, args)))

    def critical(self, *args):
        self._reset_color()
        self.logger.critical(" ".join(map(str, args)))
