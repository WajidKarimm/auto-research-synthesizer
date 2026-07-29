"""Logging helpers for the Phase 5 application surface."""

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
