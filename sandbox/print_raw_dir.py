#!/usr/bin/env python3
"""
Simple script to print raw directory paths from the database.
"""

import sys
from pathlib import Path

# Add project root to path (parent of sandbox directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory to ensure relative paths work correctly
import os
os.chdir(project_root)

from config import Config
from data_reader.storage.database import DataDatabase, query_timestamp_range

def print_raw_dirs():
    """Print all unique raw directory paths from the database.
    
    Note: The database currently stores only the raw_root from config.txt,
    not the specific subdirectory paths for each timestamp. To get the
    actual file paths, you would need to reconstruct them from the
    matching process or enhance the database to store specific paths.
    """
    # Load configuration from project root
    config_file = project_root / "config.txt"
    Config.load_from_file(config_file, silent=False)
    
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
    
    # Collect unique raw directory paths and filenames
    raw_dirs = set()
    raw_files = set()
    for record in records:
        raw_dir = record.get("source_raw_dir")
        raw_file = record.get("source_raw_file")
        if raw_dir:
            raw_dirs.add(raw_dir)
        if raw_file:
            raw_files.add(raw_file)
    
    print(f"\nFound {len(records)} total records")
    print(f"Found {len(raw_dirs)} unique raw directory paths")
    print(f"Found {len(raw_files)} unique raw filenames\n")
    
    # Print all unique raw directory paths
    if raw_dirs:
        print("Unique raw directory paths:")
        for i, raw_dir in enumerate(sorted(raw_dirs), 1):
            print(f"  {i}. {raw_dir}")
    
    # Print all unique raw filenames
    if raw_files:
        print("\nUnique raw filenames:")
        for i, raw_file in enumerate(sorted(raw_files), 1):
            print(f"  {i}. {raw_file}")
    
    # Show what's stored for each timestamp
    print("\n" + "=" * 70)
    print("Raw directory paths and filenames by timestamp (first 20):")
    print("=" * 70)
    for i, record in enumerate(records[:20], 1):
        timestamp = record.get("timestamp")
        raw_dir = record.get("source_raw_dir")
        raw_file = record.get("source_raw_file")
        print(f"Timestamp {timestamp}:")
        print(f"  Directory: {raw_dir}")
        print(f"  File: {raw_file if raw_file else '(not stored)'}")
    
    if len(records) > 20:
        print(f"\n... and {len(records) - 20} more records")
    
    # Check if all paths are the same (old database) or different (new database)
    if len(raw_dirs) == 1:
        print("\n" + "=" * 70)
        print("NOTE: All timestamps have the same raw directory path.")
        print("This database may have been created before the enhancement.")
        print("Rebuild the database to store specific paths and filenames for each timestamp.")
        print("=" * 70)
    
    if len(raw_files) == 0:
        print("\n" + "=" * 70)
        print("NOTE: No raw filenames stored in database.")
        print("Rebuild the database to store filenames for each timestamp.")
        print("=" * 70)

if __name__ == "__main__":
    print_raw_dirs()

