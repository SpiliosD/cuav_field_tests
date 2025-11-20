"""Timestamp correction utilities for processed files.

Processed files have timestamps with incorrect year (2091 instead of 2025) and
day (one day ahead). This module provides functions to correct these timestamps.
"""

from __future__ import annotations

__all__ = ["correct_processed_timestamp"]

# Correction constants
# Year correction: 2091 -> 2025 = 66 years
# Day correction: subtract 1 day
# Total correction in seconds:
# 66 years = 66 * 365.25 * 24 * 3600 = 2,082,801,600 seconds
# 1 day = 86,400 seconds
# Total = 2,082,888,000 seconds
# 
# Example: 3841455357.783980000 (2091-09-24 06:55:57.783980 UTC)
#        - 2,082,888,000
#        = 1758626477.78398 (2025-09-23 06:55:57.783980 UTC)
TIMESTAMP_CORRECTION_SECONDS = int(66 * 365.25 * 24 * 3600) + 86400


def correct_processed_timestamp(timestamp: str | float) -> float:
    """
    Correct a processed timestamp by subtracting 66 years and 1 day.
    
    Processed files have timestamps with:
    - Incorrect year: 2091 instead of 2025 (66 years ahead)
    - Incorrect day: one day ahead
    
    This function corrects the timestamp by subtracting the appropriate offset.
    
    Parameters
    ----------
    timestamp : str | float
        Unix timestamp from processed file (as string or float)
    
    Returns
    -------
    float
        Corrected timestamp in Unix seconds
    
    Examples
    --------
    >>> correct_processed_timestamp("3841455357.783980000")
    1759992957.78398
    >>> # This converts 2091-09-24 06:55:57.783980 to 2025-09-23 06:55:57.783980
    """
    ts_float = float(timestamp)
    corrected = ts_float - TIMESTAMP_CORRECTION_SECONDS
    return corrected

