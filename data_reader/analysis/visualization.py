"""Visualization and analysis tools for CUAV field test data.

This module provides utilities for creating heatmaps and other visualizations
from the aggregated data stored in the database.

The main functionality includes:
- Creating heatmaps with azimuth/elevation axes
- Extracting range-resolved profile data at specific ranges
- Aggregating data from multiple timestamps
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data_reader.storage.database import DataDatabase, query_timestamp_range

__all__ = [
    "create_heatmaps",
    "extract_range_values",
    "aggregate_azimuth_elevation_data",
]


def extract_range_values(
    profile_array: np.ndarray | None,
    range_step: float,
    starting_range: float,
    requested_ranges: list[float],
) -> dict[float, float | None]:
    """
    Extract values from a range-resolved profile at specific ranges.

    Parameters
    ----------
    profile_array : np.ndarray | None
        Range-resolved profile array. If None, returns None for all ranges.
    range_step : float
        Spacing between range bins in meters (e.g., 48 m).
    starting_range : float
        Starting range in meters (e.g., -1400 m). This is the range
        corresponding to the first element of the profile array.
    requested_ranges : list[float]
        List of specific ranges in meters to extract (e.g., [100, 200, 300]).

    Returns
    -------
    dict[float, float | None]
        Dictionary mapping requested range to extracted value.
        Value is None if profile_array is None or range is out of bounds.

    Why we need it
    --------------
    Range-resolved profiles are stored as arrays where each element
    corresponds to a specific range bin. This function maps requested
    ranges to their corresponding array indices and extracts the values.
    """
    if profile_array is None:
        return {rng: None for rng in requested_ranges}
    
    if len(profile_array) == 0:
        return {rng: None for rng in requested_ranges}
    
    result = {}
    for req_range in requested_ranges:
        # Calculate index: (requested_range - starting_range) / range_step
        index_float = (req_range - starting_range) / range_step
        index = int(np.round(index_float))
        
        # Check bounds
        if 0 <= index < len(profile_array):
            result[req_range] = float(profile_array[index])
        else:
            result[req_range] = None
    
    return result


def aggregate_azimuth_elevation_data(
    data_records: list[dict[str, Any]],
    parameter: str,
    range_step: float,
    starting_range: float,
    requested_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate data by azimuth and elevation for a specific range.

    Parameters
    ----------
    data_records : list[dict[str, Any]]
        List of data records from database query. Each record should have:
        - 'azimuth': float or None
        - 'elevation': float or None
        - 'peak', 'spectrum', or 'wind': np.ndarray or None
    parameter : str
        Parameter name to extract: 'wind', 'peak', or 'spectrum'.
    range_step : float
        Spacing between range bins in meters.
    starting_range : float
        Starting range in meters.
    requested_range : float
        Specific range in meters to extract.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (azimuth_values, elevation_values, parameter_values)
        Arrays of equal length containing the aggregated data.
        Records with None azimuth/elevation or missing parameter are excluded.

    Why we need it
    --------------
    Database records contain azimuth/elevation angles and range-resolved
    profiles. This function extracts values at a specific range and
    aggregates them by azimuth/elevation for heatmap visualization.
    """
    azimuth_list = []
    elevation_list = []
    parameter_list = []
    
    for record in data_records:
        azimuth = record.get("azimuth")
        elevation = record.get("elevation")
        
        # Skip if azimuth or elevation is missing
        if azimuth is None or elevation is None:
            continue
        
        # Get the parameter array
        profile_array = record.get(parameter)
        if profile_array is None:
            continue
        
        # Extract value at requested range
        range_values = extract_range_values(
            profile_array,
            range_step,
            starting_range,
            [requested_range],
        )
        
        value = range_values.get(requested_range)
        if value is None:
            continue
        
        azimuth_list.append(float(azimuth))
        elevation_list.append(float(elevation))
        parameter_list.append(value)
    
    return (
        np.array(azimuth_list),
        np.array(elevation_list),
        np.array(parameter_list),
    )


def create_heatmap_data(
    azimuth: np.ndarray,
    elevation: np.ndarray,
    values: np.ndarray,
    azimuth_bins: int | None = None,
    elevation_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create gridded heatmap data from scattered azimuth/elevation/values.

    Parameters
    ----------
    azimuth : np.ndarray
        Array of azimuth angles.
    elevation : np.ndarray
        Array of elevation angles.
    values : np.ndarray
        Array of parameter values corresponding to each (azimuth, elevation) pair.
    azimuth_bins : int | None
        Number of bins for azimuth axis. If None, auto-determine.
    elevation_bins : int | None
        Number of bins for elevation axis. If None, auto-determine.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (azimuth_grid, elevation_grid, value_grid)
        Gridded arrays for heatmap plotting. value_grid contains mean values
        for each grid cell, with NaN for empty cells.

    Why we need it
    --------------
    Scattered data points need to be binned into a regular grid for
    heatmap visualization. This function aggregates values within each
    grid cell (e.g., by taking the mean).
    """
    if len(azimuth) == 0:
        # Return empty grids
        if azimuth_bins is None:
            azimuth_bins = 10
        if elevation_bins is None:
            elevation_bins = 10
        return (
            np.linspace(0, 360, azimuth_bins),
            np.linspace(0, 90, elevation_bins),
            np.full((elevation_bins, azimuth_bins), np.nan),
        )
    
    # Determine bin edges
    azimuth_min, azimuth_max = azimuth.min(), azimuth.max()
    elevation_min, elevation_max = elevation.min(), elevation.max()
    
    if azimuth_bins is None:
        # Auto-determine based on data spread
        azimuth_bins = max(10, int((azimuth_max - azimuth_min) / 5) + 1) if azimuth_max > azimuth_min else 10
    
    if elevation_bins is None:
        # Auto-determine based on data spread
        elevation_bins = max(10, int((elevation_max - elevation_min) / 5) + 1) if elevation_max > elevation_min else 10
    
    # Create bins
    azimuth_edges = np.linspace(azimuth_min, azimuth_max, azimuth_bins + 1)
    elevation_edges = np.linspace(elevation_min, elevation_max, elevation_bins + 1)
    
    # Create grid centers
    azimuth_centers = (azimuth_edges[:-1] + azimuth_edges[1:]) / 2
    elevation_centers = (elevation_edges[:-1] + elevation_edges[1:]) / 2
    
    # Bin the data and compute mean for each bin
    value_grid = np.full((elevation_bins, azimuth_bins), np.nan)
    
    for i in range(elevation_bins):
        for j in range(azimuth_bins):
            # Find points in this bin
            in_elevation_bin = (elevation >= elevation_edges[i]) & (elevation < elevation_edges[i + 1])
            in_azimuth_bin = (azimuth >= azimuth_edges[j]) & (azimuth < azimuth_edges[j + 1])
            in_bin = in_elevation_bin & in_azimuth_bin
            
            if np.any(in_bin):
                # Use mean value for this bin
                value_grid[i, j] = np.mean(values[in_bin])
    
    return azimuth_centers, elevation_centers, value_grid


def create_heatmaps(
    db_path: str | Path,
    range_step: float,
    starting_range: float,
    requested_ranges: list[float],
    parameters: list[str] | None = None,
    output_dir: str | Path | None = None,
    azimuth_bins: int | None = None,
    elevation_bins: int | None = None,
    colormap: str = "viridis",
    save_format: str = "png",
) -> dict[str, dict[str, Any]]:
    """
    Create heatmaps for range-resolved parameters at specific ranges.

    Parameters
    ----------
    db_path : str | Path
        Path to the database file.
    range_step : float
        Spacing between range bins in meters (e.g., 48 m).
    starting_range : float
        Starting range in meters (e.g., -1400 m).
    requested_ranges : list[float]
        List of specific ranges in meters to visualize (e.g., [100, 200, 300]).
    parameters : list[str] | None
        List of parameters to visualize: ['wind', 'peak', 'spectrum'].
        If None, all three are used.
    output_dir : str | Path | None
        Directory to save heatmap images. If None, returns data without saving.
    azimuth_bins : int | None
        Number of bins for azimuth axis. If None, auto-determine.
    elevation_bins : int | None
        Number of bins for elevation axis. If None, auto-determine.
    colormap : str
        Matplotlib colormap name (default: 'viridis').
    save_format : str
        Image format to save ('png', 'pdf', 'svg', etc.).

    Returns
    -------
    dict[str, dict[str, Any]]
        Nested dictionary with structure:
        {
            'parameter_range': {
                'azimuth_grid': np.ndarray,
                'elevation_grid': np.ndarray,
                'value_grid': np.ndarray,
                'azimuth': np.ndarray (raw),
                'elevation': np.ndarray (raw),
                'values': np.ndarray (raw),
                'range': float,
                'parameter': str,
            }
        }
        Key format: '{parameter}_{range}' (e.g., 'wind_100.0').

    Why we need it
    --------------
    This function generates heatmaps that visualize how wind, peak, and
    spectrum parameters vary with pointing direction (azimuth/elevation)
    at specific ranges. This is essential for understanding the spatial
    distribution of atmospheric parameters.

    Notes
    -----
    - Requires matplotlib: `pip install matplotlib`
    - Heatmaps are saved as images if output_dir is provided
    - Gridded data uses mean values for each bin to handle multiple
      measurements at the same azimuth/elevation
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )
    
    if parameters is None:
        parameters = ["wind", "peak", "spectrum"]
    
    # Validate parameters
    valid_parameters = {"wind", "peak", "spectrum"}
    for param in parameters:
        if param not in valid_parameters:
            raise ValueError(
                f"Invalid parameter: {param}. Must be one of {valid_parameters}"
            )
    
    # Query all data from database
    db = DataDatabase(db_path)
    try:
        db.connect()
        records = db.query_timestamp_range()
    finally:
        db.close()
    
    results = {}
    
    # Create heatmap for each parameter and range combination
    for parameter in parameters:
        for requested_range in requested_ranges:
            key = f"{parameter}_{requested_range}"
            
            # Extract and aggregate data
            azimuth, elevation, values = aggregate_azimuth_elevation_data(
                records,
                parameter,
                range_step,
                starting_range,
                requested_range,
            )
            
            if len(azimuth) == 0:
                print(f"⚠ No data found for {parameter} at range {requested_range} m")
                continue
            
            # Create gridded heatmap data
            azimuth_grid, elevation_grid, value_grid = create_heatmap_data(
                azimuth,
                elevation,
                values,
                azimuth_bins=azimuth_bins,
                elevation_bins=elevation_bins,
            )
            
            # Store results
            results[key] = {
                "azimuth_grid": azimuth_grid,
                "elevation_grid": elevation_grid,
                "value_grid": value_grid,
                "azimuth": azimuth,
                "elevation": elevation,
                "values": values,
                "range": requested_range,
                "parameter": parameter,
            }
            
            # Create and save heatmap if output directory is provided
            if output_dir is not None:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create heatmap
                im = ax.contourf(
                    azimuth_grid,
                    elevation_grid,
                    value_grid,
                    levels=50,
                    cmap=colormap,
                )
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f"{parameter.capitalize()} Value", rotation=270, labelpad=20)
                
                # Labels and title
                ax.set_xlabel("Azimuth Angle (degrees)", fontsize=12)
                ax.set_ylabel("Elevation Angle (degrees)", fontsize=12)
                ax.set_title(
                    f"{parameter.capitalize()} Heatmap at Range {requested_range} m",
                    fontsize=14,
                    fontweight="bold",
                )
                
                # Save figure
                filename = output_path / f"{parameter}_range_{requested_range:.0f}m.{save_format}"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                plt.close()
                
                print(f"✓ Saved heatmap: {filename}")
    
    return results

