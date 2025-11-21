"""Timestamp Synchronization Module.

This module provides functionality to synchronize timestamps between log files
and peak profile files based on threshold-based detection.
"""

from .timestamp_synchronizer import synchronize_timestamps, TimestampSynchronizer

__all__ = [
    "synchronize_timestamps",
    "TimestampSynchronizer",
]

