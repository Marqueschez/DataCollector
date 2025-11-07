# Create a new file: utils/timestamp_utils.py

import pandas as pd
from datetime import datetime, timezone
from typing import Union
import time

def normalize_timestamp(timestamp: Union[str, float, int, pd.Timestamp]) -> float:
    """
    Normalize various timestamp formats to a standard Unix timestamp (float).
    Rounds to millisecond precision to avoid nanosecond issues.
    """
    try:
        if isinstance(timestamp, str):
            # Handle ISO format strings from BitMEX
            dt = pd.to_datetime(timestamp, utc=True)
            return round(dt.timestamp(), 3)  # Round to milliseconds
        
        elif isinstance(timestamp, pd.Timestamp):
            return round(timestamp.timestamp(), 3)
        
        elif isinstance(timestamp, (int, float)):
            # Already a Unix timestamp
            return round(float(timestamp), 3)
        
        else:
            # Fallback to current time
            return round(time.time(), 3)
    
    except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
        # Return current time if parsing fails
        return round(time.time(), 3)

def timestamp_to_datetime(timestamp: Union[str, float, int, pd.Timestamp]) -> datetime:
    """
    Convert timestamp to datetime object safely.
    """
    normalized_ts = normalize_timestamp(timestamp)
    return datetime.fromtimestamp(normalized_ts, tz=timezone.utc)

def validate_timestamp_range(timestamp: float, max_age_hours: int = 24) -> bool:
    """
    Validate that timestamp is within reasonable range.
    """
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    
    # Check if timestamp is not too old or in the future
    return abs(timestamp - now) <= max_age_seconds

# Usage examples:
if __name__ == "__main__":
    # Test various timestamp formats
    test_cases = [
        "2025-08-14T11:55:17.123456Z",  # ISO with microseconds
        "2025-08-14T11:55:17.123Z",     # ISO with milliseconds
        1723636517.123456,              # Unix timestamp with microseconds
        pd.Timestamp.now(tz='UTC'),     # Pandas timestamp
    ]
    
    for ts in test_cases:
        normalized = normalize_timestamp(ts)
        dt = timestamp_to_datetime(ts)
        print(f"Input: {ts}")
        print(f"Normalized: {normalized}")
        print(f"DateTime: {dt}")
        print(f"Valid: {validate_timestamp_range(normalized)}")
        print("---")