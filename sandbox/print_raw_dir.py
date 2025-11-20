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

from pathlib import Path

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
    
    # Print all unique raw filenames (just the filename, not full path)
    if raw_files:
        print("\nUnique raw filenames:")
        unique_filenames = set()
        for raw_file in raw_files:
            if raw_file:
                filename = Path(raw_file).name
                unique_filenames.add(filename)
        for i, filename in enumerate(sorted(unique_filenames), 1):
            print(f"  {i}. {filename}")
    
    # Show raw filename for each timestamp
    print("\n" + "=" * 100)
    print("Timestamps and Raw Filenames:")
    print("=" * 100)
    print(f"{'Original TS':<25} {'Corrected TS':<25} {'Raw Filename':<50}")
    print("-" * 100)
    
    for record in records:
        corrected_ts = record.get("timestamp")
        original_ts = record.get("original_timestamp")
        raw_file = record.get("source_raw_file")
        
        # Format timestamps
        original_str = str(original_ts) if original_ts else "(not stored)"
        corrected_str = str(corrected_ts) if corrected_ts else "(not stored)"
        
        # Extract filename from full path
        if raw_file:
            filename = Path(raw_file).name
            print(f"{original_str:<25} {corrected_str:<25} {filename:<50}")
        else:
            print(f"{original_str:<25} {corrected_str:<25} {'(not stored)':<50}")
    
    # Summary statistics
    records_with_file = sum(1 for r in records if r.get("source_raw_file"))
    records_without_file = len(records) - records_with_file
    
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Total records: {len(records)}")
    print(f"  Records with raw filename: {records_with_file}")
    print(f"  Records without raw filename: {records_without_file}")
    
    if records_without_file > 0:
        print("\n" + "=" * 70)
        print("NOTE: Some records are missing raw filenames.")
        print("Rebuild the database to store filenames for all timestamps.")
        print("=" * 70)

if __name__ == "__main__":
    print_raw_dirs()

