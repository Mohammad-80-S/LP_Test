import logging
from typing import Optional
from pathlib import Path
import sys


class Logger:
    """Centralized logger with debug mode support."""
    
    _instance: Optional["Logger"] = None
    
    def __new__(cls, debug: bool = False, log_file: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, debug: bool = False, log_file: Optional[str] = None):
        if self._initialized:
            return
            
        self._debug = debug
        self._logger = logging.getLogger("LPR_Pipeline")
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self._logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Format
        if debug:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
        
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self._logger.addHandler(file_handler)
        
        self._initialized = True
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        cls._instance = None
    
    @property
    def debug_mode(self) -> bool:
        return self._debug
    
    def debug(self, msg: str):
        """Log debug message (only in debug mode)."""
        self._logger.debug(msg)
    
    def info(self, msg: str):
        """Log info message."""
        self._logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self._logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self._logger.error(msg)
    
    def stage_start(self, stage_name: str):
        """Log stage start."""
        self.info(f"{'='*50}")
        self.info(f"Starting: {stage_name}")
        self.info(f"{'='*50}")
    
    def stage_end(self, stage_name: str):
        """Log stage end."""
        self.info(f"Completed: {stage_name}")
        self.info("")


def get_logger() -> Logger:
    """Get the singleton logger instance."""
    if Logger._instance is None:
        return Logger(debug=False)
    return Logger._instance