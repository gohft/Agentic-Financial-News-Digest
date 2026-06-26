import logging
import logging.config
import os

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"}
    },
    "handlers": {
        "function_node": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/function_node.log",
            "formatter": "standard",
            "maxBytes": 5_000_000,  # 5 MB per file
            "backupCount": 3,  # keep 3 old files
            "encoding": "utf-8",  # set encoding
        },
        "llm_node": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/llm_node.log",
            "formatter": "standard",
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
        },
        "graph": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/graph.log",
            "formatter": "standard",
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
        },
        "console": {
            "class": "logging.StreamHandler",  # print to console
            "formatter": "standard",
        },
    },
    "loggers": {
        "function_node": {
            "handlers": ["function_node", "console"],
            "level": "INFO",
            "propagate": False,
        },
        "llm_node": {
            "handlers": ["llm_node", "console"],
            "level": "INFO",
            "propagate": False,
        },
        "graph": {
            "handlers": ["graph", "console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def setup_logging():
    """Helper function to configure the logging config."""
    os.makedirs("logs", exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
