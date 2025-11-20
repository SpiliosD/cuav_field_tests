#!/usr/bin/env python3
"""
Simple script to print raw directory paths from the database.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_reader.storage.database import DataDatabase, query_timestamp_range

def print_raw_dirs():
    """Print all unique raw directory paths from the database."""
    # Load configuration
    Config.load_from_file(silent=False)
    
    # Get database path
    db_path = Config.get_database_path()
    if db_path is None:
        print("✗ ERROR: Database path not configured in config.txt")
        return
    
    if not db_path.exists():
        print(f"✗ ERROR: Database does not exist: {db_path}")
        return
    
    print(f"Querying database: {db_path}")
    print("=" * 70)
    
    # Query all records
    records = query_timestamp_range(db_path)
    
    if not records:
        print("No records found in database.")
        return
    
    # Collect unique raw directory paths
    raw_dirs = set()
    for record in records:
        raw_dir = record.get("source_raw_dir")
        if raw_dir:
            raw_dirs.add(raw_dir)
    
    print(f"\nFound {len(records)} total records")
    print(f"Found {len(raw_dirs)} unique raw directory paths:\n")
    
    # Print all unique raw directory paths
    for i, raw_dir in enumerate(sorted(raw_dirs), 1):
        print(f"{i}. {raw_dir}")
    
    # Optionally, print for each timestamp
    print("\n" + "=" * 70)
    print("Raw directory paths by timestamp (first 10):")
    print("=" * 70)
    for i, record in enumerate(records[:10], 1):
        timestamp = record.get("timestamp")
        raw_dir = record.get("source_raw_dir")
        print(f"Timestamp {timestamp}: {raw_dir}")
    
    if len(records) > 10:
        print(f"\n... and {len(records) - 10} more records")

if __name__ == "__main__":
    print_raw_dirs()

