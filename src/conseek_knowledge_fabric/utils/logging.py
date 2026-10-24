"""Shared structured application logging."""

from __future__ import annotations

import functools
import inspect
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


class ThreadIdFilter(logging.Filter):
	def filter(self, record: logging.LogRecord) -> bool:
		# Get the current thread
		current_thread = threading.current_thread()
		# Check if we're in the main thread
		if current_thread is threading.main_thread():
			record.thread_id = "Main Thread"
		else:
			record.thread_id = str(record.thread)
		if not hasattr(record, "function_name"):
			record.function_name = record.funcName
		return True


class MicrosecondFormatter(logging.Formatter):
	"""Formatter that includes microseconds with 4-digit precision."""
	
	def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
		ct = self.converter(record.created)
		t = time.strftime("%Y-%m-%dT%H:%M:%S", ct)
		# Get microseconds and truncate to 4 digits
		microseconds = int((record.created - int(record.created)) * 1_000_000)
		microseconds_4d = microseconds // 100  # Truncate to 4 digits
		return f"{t}.{microseconds_4d:04d}"


def get_logger(name: str) -> logging.Logger:
	logger = logging.getLogger(name)
	if not logger.handlers:
		handler = logging.StreamHandler()
		handler.addFilter(ThreadIdFilter())
		handler.setFormatter(MicrosecondFormatter(
			"%(asctime)s:%(thread_id)s:%(levelname)s:%(function_name)s:%(message)s",
		))
		logger.addHandler(handler)
		logger.propagate = False
	logger.setLevel(os.getenv("CODEFABRIC_LOG_LEVEL", "INFO").upper())
	return logger


def log_call(
	logger: logging.Logger,
	level: int = logging.INFO,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
	"""Log call boundaries and exception types without arguments or values."""
	def decorate(function: Callable[P, R]) -> Callable[P, R]:
		if inspect.iscoroutinefunction(function):
			@functools.wraps(function)
			async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
				logger.log(level, "START", extra={"function_name": function.__name__})
				try:
					result = await function(*args, **kwargs)
				except Exception as error:
					logger.error("FAILED exception_type=%s", type(error).__name__, extra={"function_name": function.__name__})
					raise
				logger.log(level, "END", extra={"function_name": function.__name__})
				return result
			return async_wrapped  # type: ignore

		@functools.wraps(function)
		def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
			logger.log(level, "START", extra={"function_name": function.__name__})
			try:
				result = function(*args, **kwargs)
			except Exception as error:
				logger.error("FAILED exception_type=%s", type(error).__name__, extra={"function_name": function.__name__})
				raise
			logger.log(level, "END", extra={"function_name": function.__name__})
			return result
		return wrapped
	return decorate