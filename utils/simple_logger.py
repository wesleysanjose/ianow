import logging
import yaml

class Log:
    def __init__(self, name, level=logging.INFO):
        print(f'Initializing logger with {name} and level {level}')
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add the formatter to the console handler
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    @staticmethod
    def get_logger(self, name, level=logging.INFO):
        if self.logger is not None:
            return self.logger
        else:
            # Read the config file
            with open("logging_config.yaml", "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Get the logging level from the config file
            log_level_str = config.get("logging", {}).get("level", "INFO")
            log_level = getattr(logging, log_level_str.upper(), logging.INFO)

            return Log(name, level)