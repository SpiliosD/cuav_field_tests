"""Database storage for aggregated CUAV field test data.

This module provides SQLite-based persistent storage for timestamp-indexed data
from multiple sources (peak, spectrum, wind, raw spectra, azimuth, elevation).

The database schema is designed to efficiently store and query time-series data
with support for:
- Fast timestamp-based lookups
- Range queries (e.g., all data between two timestamps)
- Metadata tracking (source files, import dates)
- Efficient storage of array data (using JSON or BLOB)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DataDatabase",
    "init_database",
    "save_timestamp_data",
    "query_timestamp",
    "query_timestamp_range",
]


class DataDatabase:
    """
    SQLite database for storing aggregated CUAV field test data.

    The database stores data keyed by processed timestamps, with support for
    efficient queries and data persistence across sessions.

    Attributes
    ----------
    db_path : Path
        Path to the SQLite database file
    connection : sqlite3.Connection
        Active database connection
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize database connection.

        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite database file. If it doesn't exist, it will be
            created when tables are initialized.
        """
        self.db_path = Path(db_path).expanduser().resolve()
        self.connection: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Establish connection to the database."""
        if self.connection is None:
            # Create parent directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(self.db_path))
            # Enable foreign keys
            self.connection.execute("PRAGMA foreign_keys = ON")
            # Use row factory for easier access
            self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close database connection."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.connection.commit()
        self.close()

    def create_tables(self) -> None:
        """
        Create database tables if they don't exist.

        Schema:
        - timestamps: Main table with timestamp and metadata
        - peak_data: Stores peak array data as JSON
        - spectrum_data: Stores spectrum array data as JSON
        - wind_data: Stores wind array data as JSON
        - power_density_spectrum: Stores raw spectra data as JSON
        """
        if self.connection is None:
            self.connect()

        cursor = self.connection.cursor()

        # Main timestamps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timestamps (
                timestamp REAL PRIMARY KEY,
                azimuth REAL,
                elevation REAL,
                source_processed_dir TEXT,
                source_raw_dir TEXT,
                source_log_file TEXT,
                imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Peak data table (stores array as JSON)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS peak_data (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)

        # Spectrum data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spectrum_data (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)

        # Wind data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wind_data (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)

        # Power density spectrum table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS power_density_spectrum (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)
        
        # SNR profile table (for Mode 3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snr_profile (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)
        
        # Wind profile table (for Mode 3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wind_profile (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)
        
        # Dominant frequency profile table (for Mode 3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dominant_frequency_profile (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)
        
        # SNR difference table (for sequential range pairs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snr_difference (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)
        
        # Wind difference table (for sequential range pairs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wind_difference (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)
        
        # FWHM table (for dominant frequency peaks)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fwhm_profile (
                timestamp REAL PRIMARY KEY,
                data_json TEXT NOT NULL,
                FOREIGN KEY (timestamp) REFERENCES timestamps(timestamp) ON DELETE CASCADE
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamps_imported 
            ON timestamps(imported_at)
        """)

        self.connection.commit()

    def insert_timestamp_data(
        self,
        timestamp: str | float,
        azimuth: float | None = None,
        elevation: float | None = None,
        peak: np.ndarray | None = None,
        spectrum: np.ndarray | None = None,
        wind: np.ndarray | None = None,
        power_density_spectrum: np.ndarray | None = None,
        source_processed_dir: str | None = None,
        source_raw_dir: str | None = None,
        source_log_file: str | None = None,
    ) -> None:
        """
        Insert or update data for a single timestamp.

        Parameters
        ----------
        timestamp : str | float
            Processed timestamp (will be converted to float)
        azimuth : float | None
            Azimuth angle from log file
        elevation : float | None
            Elevation angle from log file
        peak : np.ndarray | None
            Peak data array (columns 4 onwards from _Peak.txt)
        spectrum : np.ndarray | None
            Spectrum data array (columns 4 onwards from _Spectrum.txt)
        wind : np.ndarray | None
            Wind data array (columns 4 onwards from _Wind.txt)
        power_density_spectrum : np.ndarray | None
            Raw spectra data (after skipping first 13 lines)
        source_processed_dir : str | None
            Source processed directory path
        source_raw_dir : str | None
            Source raw directory path
        source_log_file : str | None
            Source log file path
        """
        if self.connection is None:
            self.connect()

        cursor = self.connection.cursor()
        ts_float = float(timestamp)

        # Insert or update main timestamp record
        cursor.execute("""
            INSERT INTO timestamps (
                timestamp, azimuth, elevation,
                source_processed_dir, source_raw_dir, source_log_file
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(timestamp) DO UPDATE SET
                azimuth = excluded.azimuth,
                elevation = excluded.elevation,
                source_processed_dir = excluded.source_processed_dir,
                source_raw_dir = excluded.source_raw_dir,
                source_log_file = excluded.source_log_file,
                updated_at = CURRENT_TIMESTAMP
        """, (
            ts_float,
            azimuth,
            elevation,
            source_processed_dir,
            source_raw_dir,
            source_log_file,
        ))

        # Insert array data (replace if exists)
        if peak is not None:
            peak_json = json.dumps(peak.tolist() if isinstance(peak, np.ndarray) else peak)
            cursor.execute("""
                INSERT INTO peak_data (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, peak_json))

        if spectrum is not None:
            spectrum_json = json.dumps(spectrum.tolist() if isinstance(spectrum, np.ndarray) else spectrum)
            cursor.execute("""
                INSERT INTO spectrum_data (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, spectrum_json))

        if wind is not None:
            wind_json = json.dumps(wind.tolist() if isinstance(wind, np.ndarray) else wind)
            cursor.execute("""
                INSERT INTO wind_data (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, wind_json))

        if power_density_spectrum is not None:
            pds_json = json.dumps(
                power_density_spectrum.tolist()
                if isinstance(power_density_spectrum, np.ndarray)
                else power_density_spectrum
            )
            cursor.execute("""
                INSERT INTO power_density_spectrum (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, pds_json))

        self.connection.commit()
    
    def insert_profile_data(
        self,
        timestamp: str | float,
        snr_profile: np.ndarray | None = None,
        wind_profile: np.ndarray | None = None,
        dominant_frequency_profile: np.ndarray | None = None,
    ) -> None:
        """
        Insert or update SNR, wind, and dominant frequency profile data for a single timestamp.
        
        This method is used by Mode 3 to store computed profiles. It only updates
        the profile tables, leaving all other data untouched.
        
        Parameters
        ----------
        timestamp : str | float
            Processed timestamp (will be converted to float)
        snr_profile : np.ndarray | None
            SNR profile array (one value per range)
        wind_profile : np.ndarray | None
            Wind profile array (one value per range)
        dominant_frequency_profile : np.ndarray | None
            Dominant frequency profile array (one value per range, in Hz)
        """
        if self.connection is None:
            self.connect()
        
        cursor = self.connection.cursor()
        ts_float = float(timestamp)
        
        # Insert or update SNR profile
        if snr_profile is not None:
            snr_json = json.dumps(snr_profile.tolist() if isinstance(snr_profile, np.ndarray) else snr_profile)
            cursor.execute("""
                INSERT INTO snr_profile (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, snr_json))
        
        # Insert or update wind profile
        if wind_profile is not None:
            wind_json = json.dumps(wind_profile.tolist() if isinstance(wind_profile, np.ndarray) else wind_profile)
            cursor.execute("""
                INSERT INTO wind_profile (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, wind_json))
        
        # Insert or update dominant frequency profile
        if dominant_frequency_profile is not None:
            freq_json = json.dumps(dominant_frequency_profile.tolist() if isinstance(dominant_frequency_profile, np.ndarray) else dominant_frequency_profile)
            cursor.execute("""
                INSERT INTO dominant_frequency_profile (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, freq_json))
        
        self.connection.commit()
    
    def insert_computed_analysis_data(
        self,
        timestamp: str | float,
        snr_difference: np.ndarray | None = None,
        wind_difference: np.ndarray | None = None,
        fwhm_profile: np.ndarray | None = None,
    ) -> None:
        """
        Insert or update computed analysis data (SNR differences, wind differences, FWHM).
        
        Parameters
        ----------
        timestamp : str | float
            Processed timestamp (will be converted to float)
        snr_difference : np.ndarray | None
            SNR difference array (differences between sequential range pairs)
        wind_difference : np.ndarray | None
            Wind speed difference array (differences between sequential range pairs)
        fwhm_profile : np.ndarray | None
            FWHM profile array (FWHM of dominant frequency peak for each range)
        """
        if self.connection is None:
            self.connect()
        
        cursor = self.connection.cursor()
        ts_float = float(timestamp)
        
        # Insert or update SNR difference
        if snr_difference is not None:
            snr_diff_json = json.dumps(snr_difference.tolist() if isinstance(snr_difference, np.ndarray) else snr_difference)
            cursor.execute("""
                INSERT INTO snr_difference (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, snr_diff_json))
        
        # Insert or update wind difference
        if wind_difference is not None:
            wind_diff_json = json.dumps(wind_difference.tolist() if isinstance(wind_difference, np.ndarray) else wind_difference)
            cursor.execute("""
                INSERT INTO wind_difference (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, wind_diff_json))
        
        # Insert or update FWHM profile
        if fwhm_profile is not None:
            fwhm_json = json.dumps(fwhm_profile.tolist() if isinstance(fwhm_profile, np.ndarray) else fwhm_profile)
            cursor.execute("""
                INSERT INTO fwhm_profile (timestamp, data_json)
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET data_json = excluded.data_json
            """, (ts_float, fwhm_json))
        
        self.connection.commit()

    def insert_from_dict(
        self,
        data_dict: dict[str, dict[str, Any]],
        source_processed_dir: str | None = None,
        source_raw_dir: str | None = None,
        source_log_file: str | None = None,
    ) -> int:
        """
        Insert data from dictionary (output of build_timestamp_data_dict).

        Parameters
        ----------
        data_dict : dict[str, dict[str, Any]]
            Dictionary keyed by timestamps, values are dicts with keys:
            azimuth, elevation, peak, spectrum, wind, power_density_spectrum
        source_processed_dir : str | None
            Source processed directory path
        source_raw_dir : str | None
            Source raw directory path
        source_log_file : str | None
            Source log file path

        Returns
        -------
        int
            Number of records inserted/updated
        """
        count = 0
        for timestamp, entry in data_dict.items():
            self.insert_timestamp_data(
                timestamp=timestamp,
                azimuth=entry.get("azimuth"),
                elevation=entry.get("elevation"),
                peak=entry.get("peak"),
                spectrum=entry.get("spectrum"),
                wind=entry.get("wind"),
                power_density_spectrum=entry.get("power_density_spectrum"),
                source_processed_dir=source_processed_dir,
                source_raw_dir=source_raw_dir,
                source_log_file=source_log_file,
            )
            count += 1
        return count

    def query_timestamp(self, timestamp: str | float) -> dict[str, Any] | None:
        """
        Query data for a specific timestamp.

        Parameters
        ----------
        timestamp : str | float
            Timestamp to query

        Returns
        -------
        dict[str, Any] | None
            Dictionary with all data for the timestamp, or None if not found
        """
        if self.connection is None:
            self.connect()

        cursor = self.connection.cursor()
        ts_float = float(timestamp)

        # Query main record
        cursor.execute("""
            SELECT * FROM timestamps WHERE timestamp = ?
        """, (ts_float,))
        row = cursor.fetchone()
        if row is None:
            return None

        result = {
            "timestamp": row["timestamp"],
            "azimuth": row["azimuth"],
            "elevation": row["elevation"],
            "source_processed_dir": row["source_processed_dir"],
            "source_raw_dir": row["source_raw_dir"],
            "source_log_file": row["source_log_file"],
            "imported_at": row["imported_at"],
            "updated_at": row["updated_at"],
        }

        # Query array data
        for table_name, key in [
            ("peak_data", "peak"),
            ("spectrum_data", "spectrum"),
            ("wind_data", "wind"),
            ("power_density_spectrum", "power_density_spectrum"),
            ("snr_profile", "snr_profile"),
            ("wind_profile", "wind_profile"),
            ("dominant_frequency_profile", "dominant_frequency_profile"),
            ("snr_difference", "snr_difference"),
            ("wind_difference", "wind_difference"),
            ("fwhm_profile", "fwhm_profile"),
        ]:
            cursor.execute(f"""
                SELECT data_json FROM {table_name} WHERE timestamp = ?
            """, (ts_float,))
            data_row = cursor.fetchone()
            if data_row is not None:
                result[key] = np.array(json.loads(data_row["data_json"]))
            else:
                result[key] = None

        return result

    def query_timestamp_range(
        self,
        start_timestamp: str | float | None = None,
        end_timestamp: str | float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query data for a range of timestamps.

        Parameters
        ----------
        start_timestamp : str | float | None
            Start timestamp (inclusive). If None, starts from beginning.
        end_timestamp : str | float | None
            End timestamp (inclusive). If None, goes to end.
        limit : int | None
            Maximum number of records to return

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries, one per timestamp
        """
        if self.connection is None:
            self.connect()

        cursor = self.connection.cursor()

        # Build query
        query = "SELECT timestamp FROM timestamps WHERE 1=1"
        params = []

        if start_timestamp is not None:
            query += " AND timestamp >= ?"
            params.append(float(start_timestamp))

        if end_timestamp is not None:
            query += " AND timestamp <= ?"
            params.append(float(end_timestamp))

        query += " ORDER BY timestamp"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        timestamps = [row["timestamp"] for row in cursor.fetchall()]

        # Fetch full data for each timestamp
        results = []
        for ts in timestamps:
            data = self.query_timestamp(ts)
            if data is not None:
                results.append(data)

        return results

    def get_statistics(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary with statistics about the database
        """
        if self.connection is None:
            self.connect()

        cursor = self.connection.cursor()

        stats = {}

        # Count total records
        cursor.execute("SELECT COUNT(*) as count FROM timestamps")
        stats["total_timestamps"] = cursor.fetchone()["count"]

        # Count records with each data type
        for table_name, key in [
            ("peak_data", "peak"),
            ("spectrum_data", "spectrum"),
            ("wind_data", "wind"),
            ("power_density_spectrum", "power_density_spectrum"),
            ("snr_profile", "snr_profile"),
            ("wind_profile", "wind_profile"),
            ("dominant_frequency_profile", "dominant_frequency_profile"),
            ("snr_difference", "snr_difference"),
            ("wind_difference", "wind_difference"),
            ("fwhm_profile", "fwhm_profile"),
        ]:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            stats[f"count_with_{key}"] = cursor.fetchone()["count"]

        # Timestamp range
        cursor.execute("SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM timestamps")
        row = cursor.fetchone()
        stats["timestamp_range"] = {
            "min": row["min_ts"],
            "max": row["max_ts"],
        }

        # Count records with azimuth/elevation
        cursor.execute("SELECT COUNT(*) as count FROM timestamps WHERE azimuth IS NOT NULL")
        stats["count_with_azimuth"] = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM timestamps WHERE elevation IS NOT NULL")
        stats["count_with_elevation"] = cursor.fetchone()["count"]

        return stats


# ============================================================================
# Convenience Functions
# ============================================================================

def init_database(db_path: str | Path) -> DataDatabase:
    """
    Initialize a new database with tables.

    Parameters
    ----------
    db_path : str | Path
        Path to the database file

    Returns
    -------
    DataDatabase
        Initialized database instance
    """
    db = DataDatabase(db_path)
    db.connect()
    db.create_tables()
    return db


def save_timestamp_data(
    data_dict: dict[str, dict[str, Any]],
    db_path: str | Path,
    source_processed_dir: str | None = None,
    source_raw_dir: str | None = None,
    source_log_file: str | None = None,
) -> int:
    """
    Save timestamp data dictionary to database.

    Parameters
    ----------
    data_dict : dict[str, dict[str, Any]]
        Dictionary from build_timestamp_data_dict
    db_path : str | Path
        Path to database file
    source_processed_dir : str | None
        Source processed directory
    source_raw_dir : str | None
        Source raw directory
    source_log_file : str | None
        Source log file

    Returns
    -------
    int
        Number of records saved
    """
    db = init_database(db_path)
    try:
        count = db.insert_from_dict(
            data_dict,
            source_processed_dir=source_processed_dir,
            source_raw_dir=source_raw_dir,
            source_log_file=source_log_file,
        )
        db.connection.commit()
        return count
    finally:
        db.close()


def query_timestamp(timestamp: str | float, db_path: str | Path) -> dict[str, Any] | None:
    """
    Query a single timestamp from database.

    Parameters
    ----------
    timestamp : str | float
        Timestamp to query
    db_path : str | Path
        Path to database file

    Returns
    -------
    dict[str, Any] | None
        Data for the timestamp, or None if not found
    """
    db = DataDatabase(db_path)
    try:
        db.connect()
        return db.query_timestamp(timestamp)
    finally:
        db.close()


def query_timestamp_range(
    db_path: str | Path,
    start_timestamp: str | float | None = None,
    end_timestamp: str | float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Query a range of timestamps from database.

    Parameters
    ----------
    db_path : str | Path
        Path to database file
    start_timestamp : str | float | None
        Start timestamp (inclusive)
    end_timestamp : str | float | None
        End timestamp (inclusive)
    limit : int | None
        Maximum number of records

    Returns
    -------
    list[dict[str, Any]]
        List of data dictionaries
    """
    db = DataDatabase(db_path)
    try:
        db.connect()
        return db.query_timestamp_range(start_timestamp, end_timestamp, limit)
    finally:
        db.close()

