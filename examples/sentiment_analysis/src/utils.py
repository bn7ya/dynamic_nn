"""
Utility functions for Sentiment Analysis project.
"""

import sys
import io
import time


def setup_unicode_output():
    """Fix Unicode output for Windows console."""
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace'
        )


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"{self.name} completed in {self.elapsed:.2f}s")

    def get_elapsed(self):
        if self.elapsed is not None:
            return self.elapsed
        return time.time() - self.start_time


def print_header(title, char="=", width=60):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def print_separator(char="-", width=60):
    """Print a separator line."""
    print(char * width)
