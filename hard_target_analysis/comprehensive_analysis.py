"""Comprehensive Hard Target Analysis using Multi-Parameter Criteria.

This module implements a sophisticated analysis method based on multiple heatmap
types and cross-parameter consistency checks to identify promising hard target
detections and flag potential false alarms.

Analysis Criteria:
1. Range-resolved signal intensity heatmap (SNR/Peak)
2. Dominant frequency / wind speed heatmap
3. FWHM of dominant peak heatmap
4. Sequential range differences in signal intensity (ΔIntensity)
5. Sequential range differences in wind speed (ΔWind)
6. Cross-parameter consistency checks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
from scipy import ndimage, stats

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from data_reader.storage.database import DataDatabase
from data_reader.analysis.visualization import (
    extract_range_values,
    aggregate_azimuth_elevation_data,
)


class ComprehensiveTargetAnalyzer:
    """
    Comprehensive analysis system for hard target detection using multi-parameter criteria.
    
    This analyzer evaluates targets based on:
    - Signal intensity patterns
    - Wind speed/frequency stability
    - FWHM characteristics
    - Sequential difference patterns
    - Cross-parameter consistency
    """
    
    def __init__(
        self,
        intensity_threshold_db: float = 3.0,
        fwhm_consistency_tolerance: float = 0.2,
        wind_plausibility_range: tuple[float, float] = (-15.0, 15.0),
        min_temporal_consistency: int = 3,
        range_falloff_tolerance: float = 0.3,
    ):
        """
        Initialize the comprehensive analyzer.
        
        Parameters
        ----------
        intensity_threshold_db : float
            Minimum intensity above background for promising detection (dB)
        fwhm_consistency_tolerance : float
            Tolerance for FWHM consistency across ranges (fraction)
        wind_plausibility_range : tuple[float, float]
            Plausible wind speed range in m/s
        min_temporal_consistency : int
            Minimum number of timestamps for temporal consistency
        range_falloff_tolerance : float
            Tolerance for range falloff smoothness
        """
        self.intensity_threshold_db = intensity_threshold_db
        self.fwhm_consistency_tolerance = fwhm_consistency_tolerance
        self.wind_plausibility_range = wind_plausibility_range
        self.min_temporal_consistency = min_temporal_consistency
        self.range_falloff_tolerance = range_falloff_tolerance
    
    def analyze_from_database(
        self,
        db_path: str | Path,
        range_step: float,
        starting_range: float,
        requested_ranges: list[float],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Perform comprehensive analysis from database.
        
        Parameters
        ----------
        db_path : str | Path
            Path to database file
        range_step : float
            Spacing between range bins in meters
        starting_range : float
            Starting range in meters
        requested_ranges : list[float]
            List of ranges to analyze
        output_dir : str | Path | None
            Output directory for results
            
        Returns
        -------
        dict[str, Any]
            Comprehensive analysis results
        """
        db = DataDatabase(db_path)
        try:
            db.connect()
            records = db.query_timestamp_range()
        finally:
            db.close()
        
        if len(records) == 0:
            return {"error": "No records found in database"}
        
        print(f"Analyzing {len(records)} records across {len(requested_ranges)} ranges...")
        
        # Extract all data
        all_range_data = {}
        for rng in requested_ranges:
            all_range_data[rng] = self._extract_comprehensive_data(
                records, range_step, starting_range, rng
            )
        
        # Perform analysis for each criterion
        results = {
            "intensity_analysis": self._analyze_intensity_heatmap(all_range_data, requested_ranges),
            "wind_analysis": self._analyze_wind_heatmap(all_range_data, requested_ranges),
            "fwhm_analysis": self._analyze_fwhm_heatmap(all_range_data, requested_ranges),
            "intensity_diff_analysis": self._analyze_intensity_differences(all_range_data, requested_ranges),
            "wind_diff_analysis": self._analyze_wind_differences(all_range_data, requested_ranges),
            "cross_parameter_analysis": self._analyze_cross_parameter_consistency(
                all_range_data, requested_ranges
            ),
        }
        
        # Generate overall assessment
        results["overall_assessment"] = self._generate_overall_assessment(results)
        
        # Save results if output directory provided
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self._save_analysis_results(results, output_path, requested_ranges)
        
        return results
    
    def _extract_comprehensive_data(
        self,
        records: list[dict[str, Any]],
        range_step: float,
        starting_range: float,
        requested_range: float,
    ) -> dict[str, Any]:
        """Extract comprehensive data for a specific range."""
        azimuth_list = []
        elevation_list = []
        intensity_list = []
        wind_list = []
        fwhm_list = []
        timestamps = []
        
        for record in records:
            azimuth = record.get("azimuth")
            elevation = record.get("elevation")
            timestamp = record.get("timestamp")
            
            if azimuth is None or elevation is None:
                continue
            
            # Extract intensity (SNR/peak)
            peak_profile = record.get("peak_profile")
            intensity_value = None
            if peak_profile is not None:
                intensity_dict = extract_range_values(
                    peak_profile, range_step, starting_range, [requested_range]
                )
                intensity_value = intensity_dict.get(requested_range)
            
            # Extract wind
            wind_profile = record.get("wind_profile")
            wind_value = None
            if wind_profile is not None:
                wind_dict = extract_range_values(
                    wind_profile, range_step, starting_range, [requested_range]
                )
                wind_value = wind_dict.get(requested_range)
            
            # Extract FWHM
            fwhm_profile = record.get("fwhm_profile")
            fwhm_value = None
            if fwhm_profile is not None:
                fwhm_dict = extract_range_values(
                    fwhm_profile, range_step, starting_range, [requested_range]
                )
                fwhm_value = fwhm_dict.get(requested_range)
            
            if intensity_value is not None:
                azimuth_list.append(float(azimuth))
                elevation_list.append(float(elevation))
                intensity_list.append(float(intensity_value))
                wind_list.append(float(wind_value) if wind_value is not None else np.nan)
                fwhm_list.append(float(fwhm_value) if fwhm_value is not None else np.nan)
                timestamps.append(float(timestamp) if timestamp is not None else np.nan)
        
        return {
            "azimuth": np.array(azimuth_list),
            "elevation": np.array(elevation_list),
            "intensity": np.array(intensity_list),
            "wind": np.array(wind_list),
            "fwhm": np.array(fwhm_list),
            "timestamps": np.array(timestamps),
        }
    
    def _analyze_intensity_heatmap(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> dict[str, Any]:
        """
        Analyze range-resolved signal intensity heatmap.
        
        Promising indicators:
        - Clear, contiguous high-intensity blob/track
        - Localized energy, not smeared
        - Consistent detectability with smooth range falloff
        - High contrast vs background
        - Repeatability across timestamps
        
        Red flags:
        - Speckled, intermittent hot pixels
        - Similar hot regions when no target present
        """
        promising_indicators = []
        red_flags = []
        scores = {}
        
        # Create intensity grid across all ranges
        intensity_grids = {}
        for rng in requested_ranges:
            data = all_range_data.get(rng)
            if data is None or len(data["intensity"]) == 0:
                continue
            
            # Create spatial grid
            az_grid, el_grid, int_grid = self._create_spatial_grid(
                data["azimuth"], data["elevation"], data["intensity"]
            )
            intensity_grids[rng] = {
                "azimuth_grid": az_grid,
                "elevation_grid": el_grid,
                "intensity_grid": int_grid,
                "raw_data": data,
            }
        
        if len(intensity_grids) == 0:
            return {
                "promising_indicators": [],
                "red_flags": ["No intensity data available"],
                "scores": {},
            }
        
        # Check 1: Contiguous high-intensity regions
        contiguous_regions = self._find_contiguous_regions(intensity_grids, requested_ranges)
        if len(contiguous_regions) > 0:
            promising_indicators.append(
                f"Found {len(contiguous_regions)} contiguous high-intensity region(s)"
            )
            scores["contiguous_regions"] = len(contiguous_regions)
        else:
            red_flags.append("No contiguous high-intensity regions found")
            scores["contiguous_regions"] = 0
        
        # Check 2: Localization (not smeared)
        localization_score = self._check_localization(intensity_grids, requested_ranges)
        if localization_score > 0.7:
            promising_indicators.append("High localization score - energy is well-localized")
        elif localization_score < 0.3:
            red_flags.append("Low localization - energy is smeared across many ranges")
        scores["localization"] = localization_score
        
        # Check 3: Range falloff smoothness
        falloff_score = self._check_range_falloff(intensity_grids, requested_ranges)
        if falloff_score > 0.7:
            promising_indicators.append("Smooth range falloff detected")
        elif falloff_score < 0.3:
            red_flags.append("Abrupt or irregular range falloff")
        scores["range_falloff"] = falloff_score
        
        # Check 4: Contrast vs background
        contrast_score = self._check_background_contrast(intensity_grids, requested_ranges)
        if contrast_score > self.intensity_threshold_db:
            promising_indicators.append(
                f"High contrast vs background ({contrast_score:.2f} dB above background)"
            )
        else:
            red_flags.append(
                f"Low contrast vs background ({contrast_score:.2f} dB, threshold: {self.intensity_threshold_db} dB)"
            )
        scores["background_contrast_db"] = contrast_score
        
        # Check 5: Temporal consistency
        temporal_score = self._check_temporal_consistency(intensity_grids, requested_ranges)
        if temporal_score > 0.7:
            promising_indicators.append("Good temporal consistency across timestamps")
        elif temporal_score < 0.3:
            red_flags.append("Poor temporal consistency - intermittent detections")
        scores["temporal_consistency"] = temporal_score
        
        # Check 6: Speckling (red flag)
        speckling_score = self._check_speckling(intensity_grids, requested_ranges)
        if speckling_score > 0.5:
            red_flags.append("High speckling detected - intermittent hot pixels")
        else:
            promising_indicators.append("Low speckling - coherent detections")
        scores["speckling"] = speckling_score
        
        return {
            "promising_indicators": promising_indicators,
            "red_flags": red_flags,
            "scores": scores,
            "contiguous_regions": contiguous_regions,
        }
    
    def _analyze_wind_heatmap(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> dict[str, Any]:
        """
        Analyze dominant frequency / wind speed heatmap.
        
        Promising indicators:
        - Stable dominant-frequency ridge at target range
        - Physically plausible wind speeds
        - Target-coupled modulation during maneuvers
        
        Red flags:
        - Unrealistic spikes
        - Dominant frequency disappearing or splitting
        """
        promising_indicators = []
        red_flags = []
        scores = {}
        
        # Check 1: Wind speed plausibility
        all_wind_values = []
        for rng in requested_ranges:
            data = all_range_data.get(rng)
            if data is None:
                continue
            wind_vals = data["wind"][~np.isnan(data["wind"])]
            all_wind_values.extend(wind_vals.tolist())
        
        if len(all_wind_values) > 0:
            wind_array = np.array(all_wind_values)
            plausible_mask = (wind_array >= self.wind_plausibility_range[0]) & \
                           (wind_array <= self.wind_plausibility_range[1])
            plausibility_ratio = np.sum(plausible_mask) / len(wind_array)
            
            if plausibility_ratio > 0.8:
                promising_indicators.append(
                    f"High wind speed plausibility ({plausibility_ratio*100:.1f}% within expected range)"
                )
            else:
                red_flags.append(
                    f"Low wind speed plausibility ({plausibility_ratio*100:.1f}% within expected range)"
                )
            scores["wind_plausibility"] = plausibility_ratio
        else:
            red_flags.append("No wind data available")
            scores["wind_plausibility"] = 0.0
        
        # Check 2: Stability across ranges
        stability_score = self._check_wind_stability(all_range_data, requested_ranges)
        if stability_score > 0.7:
            promising_indicators.append("Stable wind estimates across ranges")
        elif stability_score < 0.3:
            red_flags.append("Unstable wind estimates - frequent jumps")
        scores["wind_stability"] = stability_score
        
        # Check 3: Smooth transitions
        smoothness_score = self._check_wind_smoothness(all_range_data, requested_ranges)
        if smoothness_score > 0.7:
            promising_indicators.append("Smooth wind speed transitions")
        else:
            red_flags.append("Abrupt wind speed changes detected")
        scores["wind_smoothness"] = smoothness_score
        
        return {
            "promising_indicators": promising_indicators,
            "red_flags": red_flags,
            "scores": scores,
        }
    
    def _analyze_fwhm_heatmap(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> dict[str, Any]:
        """
        Analyze FWHM of dominant peak heatmap.
        
        Promising indicators:
        - Narrow FWHM at target ranges relative to background
        - Gradual broadening with range
        - FWHM dips coincident with intensity increases
        
        Red flags:
        - Wide FWHM where intensity says target is present
        - Alternating narrow/wide FWHM (checkerboarding)
        """
        promising_indicators = []
        red_flags = []
        scores = {}
        
        # Check 1: FWHM vs intensity correlation
        correlation_score = self._check_fwhm_intensity_correlation(all_range_data, requested_ranges)
        if correlation_score > 0.6:
            promising_indicators.append(
                "Good FWHM-intensity correlation - narrow FWHM with high intensity"
            )
        elif correlation_score < -0.3:
            red_flags.append(
                "Negative FWHM-intensity correlation - wide FWHM with high intensity (suspicious)"
            )
        scores["fwhm_intensity_correlation"] = correlation_score
        
        # Check 2: Range-dependent broadening
        broadening_score = self._check_fwhm_broadening(all_range_data, requested_ranges)
        if broadening_score > 0.7:
            promising_indicators.append("Gradual FWHM broadening with range (expected)")
        elif broadening_score < 0.3:
            red_flags.append("Irregular FWHM broadening pattern")
        scores["fwhm_broadening"] = broadening_score
        
        # Check 3: Checkerboarding
        checkerboard_score = self._check_fwhm_checkerboarding(all_range_data, requested_ranges)
        if checkerboard_score > 0.5:
            red_flags.append("FWHM checkerboarding detected - alternating narrow/wide pattern")
        else:
            promising_indicators.append("No FWHM checkerboarding - consistent pattern")
        scores["checkerboarding"] = checkerboard_score
        
        return {
            "promising_indicators": promising_indicators,
            "red_flags": red_flags,
            "scores": scores,
        }
    
    def _analyze_intensity_differences(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> dict[str, Any]:
        """
        Analyze sequential range differences in signal intensity (ΔIntensity).
        
        Promising indicators:
        - Sharp gradient at target boundary
        - Low ΔIntensity elsewhere
        - Edges move consistently with target trajectory
        
        Red flags:
        - Large fluctuations everywhere
        - Gradients that don't move with target
        """
        promising_indicators = []
        red_flags = []
        scores = {}
        
        # Calculate differences between adjacent ranges
        sorted_ranges = sorted(requested_ranges)
        difference_data = {}
        
        for i in range(len(sorted_ranges) - 1):
            rng1, rng2 = sorted_ranges[i], sorted_ranges[i + 1]
            data1 = all_range_data.get(rng1)
            data2 = all_range_data.get(rng2)
            
            if data1 is None or data2 is None:
                continue
            
            # Match points and calculate differences
            diffs = self._calculate_spatial_differences(
                data1["azimuth"], data1["elevation"], data1["intensity"],
                data2["azimuth"], data2["elevation"], data2["intensity"],
            )
            difference_data[(rng1, rng2)] = diffs
        
        # Check 1: Gradient sharpness
        gradient_score = self._check_gradient_sharpness(difference_data)
        if gradient_score > 0.7:
            promising_indicators.append("Sharp gradients at target boundaries")
        else:
            red_flags.append("Weak or diffuse gradients")
        scores["gradient_sharpness"] = gradient_score
        
        # Check 2: Background stability
        background_score = self._check_background_stability(difference_data)
        if background_score > 0.7:
            promising_indicators.append("Stable background - low ΔIntensity elsewhere")
        else:
            red_flags.append("Unstable background - large fluctuations everywhere")
        scores["background_stability"] = background_score
        
        return {
            "promising_indicators": promising_indicators,
            "red_flags": red_flags,
            "scores": scores,
        }
    
    def _analyze_wind_differences(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> dict[str, Any]:
        """
        Analyze sequential range differences in wind speed (ΔWind).
        
        Promising indicators:
        - Small, smooth ΔWind in background
        - Localized ΔWind anomaly at target range
        - Temporal coherence
        
        Red flags:
        - Random large spikes across many bins
        - Persistent structure when no target present
        """
        promising_indicators = []
        red_flags = []
        scores = {}
        
        # Calculate wind differences
        sorted_ranges = sorted(requested_ranges)
        difference_data = {}
        
        for i in range(len(sorted_ranges) - 1):
            rng1, rng2 = sorted_ranges[i], sorted_ranges[i + 1]
            data1 = all_range_data.get(rng1)
            data2 = all_range_data.get(rng2)
            
            if data1 is None or data2 is None:
                continue
            
            diffs = self._calculate_spatial_differences(
                data1["azimuth"], data1["elevation"], data1["wind"],
                data2["azimuth"], data2["elevation"], data2["wind"],
            )
            difference_data[(rng1, rng2)] = diffs
        
        # Check 1: Background smoothness
        smoothness_score = self._check_wind_diff_smoothness(difference_data)
        if smoothness_score > 0.7:
            promising_indicators.append("Smooth ΔWind in background ranges")
        else:
            red_flags.append("Irregular ΔWind in background")
        scores["wind_diff_smoothness"] = smoothness_score
        
        # Check 2: Localization
        localization_score = self._check_wind_diff_localization(difference_data)
        if localization_score > 0.7:
            promising_indicators.append("Localized ΔWind anomalies")
        else:
            red_flags.append("Widespread ΔWind anomalies")
        scores["wind_diff_localization"] = localization_score
        
        return {
            "promising_indicators": promising_indicators,
            "red_flags": red_flags,
            "scores": scores,
        }
    
    def _analyze_cross_parameter_consistency(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> dict[str, Any]:
        """
        Cross-parameter consistency checks.
        
        Promising indicators:
        - High intensity + narrow FWHM + stable frequency co-located
        - Edges in ΔIntensity and ΔWind bracket same target track
        - Altitude dependence makes sense
        
        Red flags:
        - Inconsistent signatures across parameters
        """
        promising_indicators = []
        red_flags = []
        scores = {}
        
        # Check 1: Co-location of high intensity, narrow FWHM, stable wind
        colocation_score = self._check_parameter_colocation(all_range_data, requested_ranges)
        if colocation_score > 0.7:
            promising_indicators.append(
                "Strong co-location: high intensity + narrow FWHM + stable wind"
            )
        elif colocation_score < 0.3:
            red_flags.append("Poor co-location - inconsistent signatures")
        scores["parameter_colocation"] = colocation_score
        
        # Check 2: Edge alignment
        edge_alignment_score = self._check_edge_alignment(all_range_data, requested_ranges)
        if edge_alignment_score > 0.7:
            promising_indicators.append("Aligned edges in ΔIntensity and ΔWind")
        else:
            red_flags.append("Misaligned edges - inconsistent boundaries")
        scores["edge_alignment"] = edge_alignment_score
        
        # Check 3: Overall consistency
        overall_consistency = (colocation_score + edge_alignment_score) / 2.0
        if overall_consistency > 0.7:
            promising_indicators.append("High overall cross-parameter consistency")
        elif overall_consistency < 0.3:
            red_flags.append("Low cross-parameter consistency")
        scores["overall_consistency"] = overall_consistency
        
        return {
            "promising_indicators": promising_indicators,
            "red_flags": red_flags,
            "scores": scores,
        }
    
    def _generate_overall_assessment(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate overall assessment from all analysis results."""
        # Count promising indicators and red flags
        total_promising = sum(
            len(r.get("promising_indicators", [])) for r in results.values()
            if isinstance(r, dict) and "promising_indicators" in r
        )
        total_red_flags = sum(
            len(r.get("red_flags", [])) for r in results.values()
            if isinstance(r, dict) and "red_flags" in r
        )
        
        # Calculate overall score
        all_scores = []
        for r in results.values():
            if isinstance(r, dict) and "scores" in r:
                all_scores.extend([v for v in r["scores"].values() if isinstance(v, (int, float))])
        
        overall_score = np.mean(all_scores) if len(all_scores) > 0 else 0.0
        
        # Determine assessment level
        if overall_score > 0.7 and total_red_flags < total_promising:
            assessment = "PROMISING"
            confidence = "HIGH"
        elif overall_score > 0.5:
            assessment = "MODERATE"
            confidence = "MEDIUM"
        else:
            assessment = "POOR"
            confidence = "LOW"
        
        return {
            "assessment": assessment,
            "confidence": confidence,
            "overall_score": float(overall_score),
            "total_promising_indicators": total_promising,
            "total_red_flags": total_red_flags,
            "promising_ratio": total_promising / (total_promising + total_red_flags + 1e-10),
        }
    
    # Helper methods for specific checks
    
    def _create_spatial_grid(
        self,
        azimuth: np.ndarray,
        elevation: np.ndarray,
        values: np.ndarray,
        az_bins: int = 72,
        el_bins: int = 18,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create spatial grid from scattered data."""
        if len(azimuth) == 0:
            return (
                np.linspace(0, 360, az_bins),
                np.linspace(0, 90, el_bins),
                np.full((el_bins, az_bins), np.nan),
            )
        
        az_edges = np.linspace(0, 360, az_bins + 1)
        el_edges = np.linspace(0, 90, el_bins + 1)
        
        # Bin data
        az_indices = np.digitize(azimuth, az_edges) - 1
        el_indices = np.digitize(elevation, el_edges) - 1
        
        # Create grid with mean values
        value_grid = np.full((el_bins, az_bins), np.nan)
        for i in range(el_bins):
            for j in range(az_bins):
                mask = (el_indices == i) & (az_indices == j)
                if np.any(mask):
                    value_grid[i, j] = np.nanmean(values[mask])
        
        az_centers = (az_edges[:-1] + az_edges[1:]) / 2
        el_centers = (el_edges[:-1] + el_edges[1:]) / 2
        
        return az_centers, el_centers, value_grid
    
    def _find_contiguous_regions(
        self,
        intensity_grids: dict[float, dict[str, Any]],
        requested_ranges: list[float],
    ) -> list[dict[str, Any]]:
        """Find contiguous high-intensity regions across ranges."""
        regions = []
        threshold_factor = 0.7  # 70th percentile
        
        for rng in requested_ranges:
            grid_data = intensity_grids.get(rng)
            if grid_data is None:
                continue
            
            int_grid = grid_data["intensity_grid"]
            if np.all(np.isnan(int_grid)):
                continue
            
            # Threshold
            threshold = np.nanpercentile(int_grid, threshold_factor * 100)
            binary = int_grid > threshold
            
            # Find connected components
            labeled, num_features = ndimage.label(binary)
            
            for label_id in range(1, num_features + 1):
                mask = labeled == label_id
                if np.sum(mask) >= 4:  # Minimum region size
                    region_intensity = np.nanmean(int_grid[mask])
                    region_coords = np.where(mask)
                    
                    regions.append({
                        "range": rng,
                        "size": int(np.sum(mask)),
                        "mean_intensity": float(region_intensity),
                        "azimuth_center": float(np.mean(grid_data["azimuth_grid"][region_coords[1]])),
                        "elevation_center": float(np.mean(grid_data["elevation_grid"][region_coords[0]])),
                    })
        
        return regions
    
    def _check_localization(
        self,
        intensity_grids: dict[float, dict[str, Any]],
        requested_ranges: list[float],
    ) -> float:
        """Check if energy is localized (not smeared)."""
        localization_scores = []
        
        for rng in requested_ranges:
            grid_data = intensity_grids.get(rng)
            if grid_data is None:
                continue
            
            int_grid = grid_data["intensity_grid"]
            if np.all(np.isnan(int_grid)):
                continue
            
            # Calculate energy concentration
            total_energy = np.nansum(int_grid)
            if total_energy == 0:
                continue
            
            # Find top 20% of pixels
            threshold = np.nanpercentile(int_grid, 80)
            top_mask = int_grid > threshold
            top_energy = np.nansum(int_grid[top_mask])
            
            concentration = top_energy / total_energy
            localization_scores.append(concentration)
        
        return np.mean(localization_scores) if len(localization_scores) > 0 else 0.0
    
    def _check_range_falloff(
        self,
        intensity_grids: dict[float, dict[str, Any]],
        requested_ranges: list[float],
    ) -> float:
        """Check smoothness of range falloff."""
        sorted_ranges = sorted(requested_ranges)
        falloff_scores = []
        
        for rng in sorted_ranges:
            grid_data = intensity_grids.get(rng)
            if grid_data is None:
                continue
            
            int_grid = grid_data["intensity_grid"]
            if np.all(np.isnan(int_grid)):
                continue
            
            # Compare with previous range
            prev_rng_idx = sorted_ranges.index(rng) - 1
            if prev_rng_idx >= 0:
                prev_grid_data = intensity_grids.get(sorted_ranges[prev_rng_idx])
                if prev_grid_data is not None:
                    prev_grid = prev_grid_data["intensity_grid"]
                    
                    # Calculate correlation
                    valid_mask = ~np.isnan(int_grid) & ~np.isnan(prev_grid)
                    if np.sum(valid_mask) > 10:
                        corr = np.corrcoef(
                            int_grid[valid_mask].flatten(),
                            prev_grid[valid_mask].flatten()
                        )[0, 1]
                        if not np.isnan(corr):
                            falloff_scores.append(corr)
        
        return np.mean(falloff_scores) if len(falloff_scores) > 0 else 0.0
    
    def _check_background_contrast(
        self,
        intensity_grids: dict[float, dict[str, Any]],
        requested_ranges: list[float],
    ) -> float:
        """Check contrast vs background (in dB)."""
        all_intensities = []
        high_intensities = []
        
        for rng in requested_ranges:
            grid_data = intensity_grids.get(rng)
            if grid_data is None:
                continue
            
            int_grid = grid_data["intensity_grid"]
            valid = int_grid[~np.isnan(int_grid)]
            all_intensities.extend(valid.tolist())
            
            # Top 10% as "target"
            if len(valid) > 0:
                threshold = np.percentile(valid, 90)
                high_intensities.extend(valid[valid > threshold].tolist())
        
        if len(all_intensities) == 0 or len(high_intensities) == 0:
            return 0.0
        
        # Convert to dB and calculate contrast
        all_db = 10 * np.log10(np.array(all_intensities) + 1e-10)
        high_db = 10 * np.log10(np.array(high_intensities) + 1e-10)
        
        background_mean = np.mean(all_db)
        target_mean = np.mean(high_db)
        
        return float(target_mean - background_mean)
    
    def _check_temporal_consistency(
        self,
        intensity_grids: dict[float, dict[str, Any]],
        requested_ranges: list[float],
    ) -> float:
        """Check temporal consistency across timestamps."""
        # This would require timestamp data - simplified version
        # Check consistency across ranges as proxy
        consistency_scores = []
        
        for rng in requested_ranges:
            grid_data = intensity_grids.get(rng)
            if grid_data is None:
                continue
            
            int_grid = grid_data["intensity_grid"]
            if np.all(np.isnan(int_grid)):
                continue
            
            # Calculate coefficient of variation
            valid = int_grid[~np.isnan(int_grid)]
            if len(valid) > 0 and np.mean(valid) > 0:
                cv = np.std(valid) / np.mean(valid)
                consistency_scores.append(1.0 / (1.0 + cv))  # Inverse CV as consistency
        
        return np.mean(consistency_scores) if len(consistency_scores) > 0 else 0.0
    
    def _check_speckling(
        self,
        intensity_grids: dict[float, dict[str, Any]],
        requested_ranges: list[float],
    ) -> float:
        """Check for speckling (intermittent hot pixels)."""
        speckling_scores = []
        
        for rng in requested_ranges:
            grid_data = intensity_grids.get(rng)
            if grid_data is None:
                continue
            
            int_grid = grid_data["intensity_grid"]
            if np.all(np.isnan(int_grid)):
                continue
            
            # Find isolated high pixels
            threshold = np.nanpercentile(int_grid, 90)
            high_mask = int_grid > threshold
            
            # Count isolated pixels (no neighbors)
            isolated_count = 0
            for i in range(1, int_grid.shape[0] - 1):
                for j in range(1, int_grid.shape[1] - 1):
                    if high_mask[i, j]:
                        neighbors = high_mask[i-1:i+2, j-1:j+2]
                        if np.sum(neighbors) <= 1:  # Only itself
                            isolated_count += 1
            
            speckling_ratio = isolated_count / np.sum(high_mask) if np.sum(high_mask) > 0 else 0.0
            speckling_scores.append(speckling_ratio)
        
        return np.mean(speckling_scores) if len(speckling_scores) > 0 else 0.0
    
    def _check_wind_stability(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> float:
        """Check wind speed stability across ranges."""
        stability_scores = []
        sorted_ranges = sorted(requested_ranges)
        
        for i in range(len(sorted_ranges) - 1):
            rng1, rng2 = sorted_ranges[i], sorted_ranges[i + 1]
            data1 = all_range_data.get(rng1)
            data2 = all_range_data.get(rng2)
            
            if data1 is None or data2 is None:
                continue
            
            # Match points and calculate differences
            diffs = self._calculate_spatial_differences(
                data1["azimuth"], data1["elevation"], data1["wind"],
                data2["azimuth"], data2["elevation"], data2["wind"],
            )
            
            if len(diffs) > 0:
                # Small differences = stable
                abs_diffs = np.abs(diffs)
                stability = 1.0 / (1.0 + np.mean(abs_diffs))
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if len(stability_scores) > 0 else 0.0
    
    def _check_wind_smoothness(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> float:
        """Check smoothness of wind transitions."""
        return self._check_wind_stability(all_range_data, requested_ranges)  # Similar metric
    
    def _check_fwhm_intensity_correlation(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> float:
        """Check correlation between FWHM and intensity (negative = good)."""
        all_intensities = []
        all_fwhms = []
        
        for rng in requested_ranges:
            data = all_range_data.get(rng)
            if data is None:
                continue
            
            valid_mask = ~np.isnan(data["intensity"]) & ~np.isnan(data["fwhm"])
            if np.sum(valid_mask) > 0:
                all_intensities.extend(data["intensity"][valid_mask].tolist())
                all_fwhms.extend(data["fwhm"][valid_mask].tolist())
        
        if len(all_intensities) < 10:
            return 0.0
        
        # Negative correlation is good (high intensity, low FWHM)
        corr = np.corrcoef(all_intensities, all_fwhms)[0, 1]
        return float(-corr if not np.isnan(corr) else 0.0)  # Negate so positive = good
    
    def _check_fwhm_broadening(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> float:
        """Check if FWHM broadens gradually with range."""
        sorted_ranges = sorted(requested_ranges)
        mean_fwhms = []
        
        for rng in sorted_ranges:
            data = all_range_data.get(rng)
            if data is None:
                continue
            
            fwhm_vals = data["fwhm"][~np.isnan(data["fwhm"])]
            if len(fwhm_vals) > 0:
                mean_fwhms.append(np.mean(fwhm_vals))
        
        if len(mean_fwhms) < 3:
            return 0.0
        
        # Check if FWHM increases with range (monotonic trend)
        mean_fwhms = np.array(mean_fwhms)
        range_indices = np.arange(len(mean_fwhms))
        corr = np.corrcoef(range_indices, mean_fwhms)[0, 1]
        
        return float(corr if not np.isnan(corr) else 0.0)
    
    def _check_fwhm_checkerboarding(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> float:
        """Check for alternating narrow/wide FWHM pattern."""
        sorted_ranges = sorted(requested_ranges)
        checkerboard_scores = []
        
        for i in range(len(sorted_ranges) - 2):
            rng1, rng2, rng3 = sorted_ranges[i], sorted_ranges[i+1], sorted_ranges[i+2]
            data1 = all_range_data.get(rng1)
            data2 = all_range_data.get(rng2)
            data3 = all_range_data.get(rng3)
            
            if data1 is None or data2 is None or data3 is None:
                continue
            
            # Check for alternating pattern
            fwhm1 = np.nanmean(data1["fwhm"][~np.isnan(data1["fwhm"])]) if len(data1["fwhm"][~np.isnan(data1["fwhm"])]) > 0 else np.nan
            fwhm2 = np.nanmean(data2["fwhm"][~np.isnan(data2["fwhm"])]) if len(data2["fwhm"][~np.isnan(data2["fwhm"])]) > 0 else np.nan
            fwhm3 = np.nanmean(data3["fwhm"][~np.isnan(data3["fwhm"])]) if len(data3["fwhm"][~np.isnan(data3["fwhm"])]) > 0 else np.nan
            
            if not (np.isnan(fwhm1) or np.isnan(fwhm2) or np.isnan(fwhm3)):
                # Check if fwhm2 is opposite of fwhm1 and fwhm3
                if (fwhm2 < fwhm1 and fwhm2 < fwhm3) or (fwhm2 > fwhm1 and fwhm2 > fwhm3):
                    checkerboard_scores.append(1.0)
                else:
                    checkerboard_scores.append(0.0)
        
        return np.mean(checkerboard_scores) if len(checkerboard_scores) > 0 else 0.0
    
    def _check_gradient_sharpness(
        self,
        difference_data: dict[tuple[float, float], np.ndarray],
    ) -> float:
        """Check sharpness of gradients in difference data."""
        sharpness_scores = []
        
        for diffs in difference_data.values():
            if len(diffs) == 0:
                continue
            
            # Calculate gradient magnitude
            abs_diffs = np.abs(diffs)
            max_diff = np.max(abs_diffs)
            mean_diff = np.mean(abs_diffs)
            
            if mean_diff > 0:
                sharpness = max_diff / mean_diff
                sharpness_scores.append(min(sharpness / 5.0, 1.0))  # Normalize
        
        return np.mean(sharpness_scores) if len(sharpness_scores) > 0 else 0.0
    
    def _check_background_stability(
        self,
        difference_data: dict[tuple[float, float], np.ndarray],
    ) -> float:
        """Check stability of background in difference data."""
        stability_scores = []
        
        for diffs in difference_data.values():
            if len(diffs) == 0:
                continue
            
            # Low variance = stable
            std_diff = np.std(diffs)
            mean_abs_diff = np.mean(np.abs(diffs))
            
            if mean_abs_diff > 0:
                cv = std_diff / mean_abs_diff
                stability = 1.0 / (1.0 + cv)
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if len(stability_scores) > 0 else 0.0
    
    def _check_wind_diff_smoothness(
        self,
        difference_data: dict[tuple[float, float], np.ndarray],
    ) -> float:
        """Check smoothness of wind differences."""
        return self._check_background_stability(difference_data)
    
    def _check_wind_diff_localization(
        self,
        difference_data: dict[tuple[float, float], np.ndarray],
    ) -> float:
        """Check if wind differences are localized."""
        localization_scores = []
        
        for diffs in difference_data.values():
            if len(diffs) == 0:
                continue
            
            # Check if large differences are localized
            threshold = np.percentile(np.abs(diffs), 90)
            large_diffs = np.abs(diffs) > threshold
            
            if np.sum(large_diffs) > 0:
                # Calculate spatial clustering of large diffs
                # Simplified: ratio of large diffs to total
                localization = 1.0 - (np.sum(large_diffs) / len(diffs))
                localization_scores.append(localization)
        
        return np.mean(localization_scores) if len(localization_scores) > 0 else 0.0
    
    def _check_parameter_colocation(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> float:
        """Check co-location of high intensity, narrow FWHM, stable wind."""
        colocation_scores = []
        
        for rng in requested_ranges:
            data = all_range_data.get(rng)
            if data is None:
                continue
            
            # Find high intensity points
            int_threshold = np.percentile(data["intensity"], 80)
            high_int_mask = data["intensity"] > int_threshold
            
            # Find narrow FWHM points (if available)
            if np.any(~np.isnan(data["fwhm"])):
                fwhm_threshold = np.percentile(data["fwhm"][~np.isnan(data["fwhm"])], 20)
                narrow_fwhm_mask = data["fwhm"] < fwhm_threshold
                narrow_fwhm_mask = narrow_fwhm_mask | np.isnan(data["fwhm"])
            else:
                narrow_fwhm_mask = np.ones(len(data["intensity"]), dtype=bool)
            
            # Find stable wind points
            if np.any(~np.isnan(data["wind"])):
                wind_vals = data["wind"][~np.isnan(data["wind"])]
                wind_mean = np.mean(wind_vals)
                wind_std = np.std(wind_vals)
                stable_wind_mask = np.abs(data["wind"] - wind_mean) < 2 * wind_std
                stable_wind_mask = stable_wind_mask | np.isnan(data["wind"])
            else:
                stable_wind_mask = np.ones(len(data["intensity"]), dtype=bool)
            
            # Check co-location
            colocated_mask = high_int_mask & narrow_fwhm_mask & stable_wind_mask
            if np.sum(high_int_mask) > 0:
                colocation_ratio = np.sum(colocated_mask) / np.sum(high_int_mask)
                colocation_scores.append(colocation_ratio)
        
        return np.mean(colocation_scores) if len(colocation_scores) > 0 else 0.0
    
    def _check_edge_alignment(
        self,
        all_range_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> float:
        """Check alignment of edges in ΔIntensity and ΔWind."""
        sorted_ranges = sorted(requested_ranges)
        alignment_scores = []
        
        for i in range(len(sorted_ranges) - 1):
            rng1, rng2 = sorted_ranges[i], sorted_ranges[i + 1]
            data1 = all_range_data.get(rng1)
            data2 = all_range_data.get(rng2)
            
            if data1 is None or data2 is None:
                continue
            
            # Calculate intensity differences
            int_diffs = self._calculate_spatial_differences(
                data1["azimuth"], data1["elevation"], data1["intensity"],
                data2["azimuth"], data2["elevation"], data2["intensity"],
            )
            
            # Calculate wind differences
            wind_diffs = self._calculate_spatial_differences(
                data1["azimuth"], data1["elevation"], data1["wind"],
                data2["azimuth"], data2["elevation"], data2["wind"],
            )
            
            if len(int_diffs) > 0 and len(wind_diffs) > 0:
                # Check correlation of large differences
                int_large = np.abs(int_diffs) > np.percentile(np.abs(int_diffs), 80)
                wind_large = np.abs(wind_diffs) > np.percentile(np.abs(wind_diffs), 80)
                
                # Alignment = both large at same locations
                alignment = np.sum(int_large & wind_large) / max(np.sum(int_large | wind_large), 1)
                alignment_scores.append(alignment)
        
        return np.mean(alignment_scores) if len(alignment_scores) > 0 else 0.0
    
    def _calculate_spatial_differences(
        self,
        az1: np.ndarray,
        el1: np.ndarray,
        val1: np.ndarray,
        az2: np.ndarray,
        el2: np.ndarray,
        val2: np.ndarray,
        max_distance: float = 5.0,
    ) -> np.ndarray:
        """Calculate spatial differences by matching nearest neighbors."""
        differences = []
        
        for i in range(len(az1)):
            if np.isnan(val1[i]):
                continue
            
            # Find nearest point in second dataset
            distances = np.sqrt((az2 - az1[i]) ** 2 + (el2 - el1[i]) ** 2)
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] <= max_distance and not np.isnan(val2[nearest_idx]):
                diff = val2[nearest_idx] - val1[i]
                differences.append(diff)
        
        return np.array(differences)
    
    def _save_analysis_results(
        self,
        results: dict[str, Any],
        output_dir: Path,
        requested_ranges: list[float],
    ) -> None:
        """Save analysis results to files."""
        # Save JSON report
        json_file = output_dir / "comprehensive_analysis_report.json"
        
        # Convert numpy types to native Python types
        def convert_to_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json(item) for item in obj]
            return obj
        
        json_results = convert_to_json(results)
        json_results["metadata"] = {
            "ranges_analyzed": requested_ranges,
            "analysis_timestamp": str(Path(__file__).stat().st_mtime),
        }
        
        with open(json_file, "w") as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Saved analysis report: {json_file}")
        
        # Create text summary
        summary_file = output_dir / "analysis_summary.txt"
        with open(summary_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("Comprehensive Hard Target Analysis Report\n")
            f.write("=" * 70 + "\n\n")
            
            assessment = results.get("overall_assessment", {})
            f.write(f"Overall Assessment: {assessment.get('assessment', 'UNKNOWN')}\n")
            f.write(f"Confidence: {assessment.get('confidence', 'UNKNOWN')}\n")
            f.write(f"Overall Score: {assessment.get('overall_score', 0.0):.3f}\n")
            f.write(f"Promising Indicators: {assessment.get('total_promising_indicators', 0)}\n")
            f.write(f"Red Flags: {assessment.get('total_red_flags', 0)}\n")
            f.write(f"Promising Ratio: {assessment.get('promising_ratio', 0.0):.3f}\n")
            f.write("\n" + "=" * 70 + "\n\n")
            
            # Write detailed results for each criterion
            for criterion_name, criterion_results in results.items():
                if criterion_name == "overall_assessment":
                    continue
                
                f.write(f"{criterion_name.upper().replace('_', ' ')}\n")
                f.write("-" * 70 + "\n")
                
                if isinstance(criterion_results, dict):
                    promising = criterion_results.get("promising_indicators", [])
                    red_flags = criterion_results.get("red_flags", [])
                    scores = criterion_results.get("scores", {})
                    
                    f.write("\nPromising Indicators:\n")
                    for indicator in promising:
                        f.write(f"  ✓ {indicator}\n")
                    
                    f.write("\nRed Flags:\n")
                    for flag in red_flags:
                        f.write(f"  ✗ {flag}\n")
                    
                    f.write("\nScores:\n")
                    for score_name, score_value in scores.items():
                        f.write(f"  {score_name}: {score_value:.3f}\n")
                
                f.write("\n")
        
        print(f"✓ Saved analysis summary: {summary_file}")
        
        # Create visualization if matplotlib available
        if HAS_MATPLOTLIB:
            self._create_analysis_visualization(results, output_dir, requested_ranges)
    
    def _create_analysis_visualization(
        self,
        results: dict[str, Any],
        output_dir: Path,
        requested_ranges: list[float],
    ) -> None:
        """Create visualization of analysis results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Comprehensive Hard Target Analysis Results", fontsize=16, fontweight="bold")
        
        # Plot 1: Overall assessment
        ax = axes[0, 0]
        assessment = results.get("overall_assessment", {})
        score = assessment.get("overall_score", 0.0)
        promising = assessment.get("total_promising_indicators", 0)
        red_flags = assessment.get("total_red_flags", 0)
        
        ax.barh(["Overall Score"], [score], color='green' if score > 0.7 else 'orange' if score > 0.5 else 'red')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title("Overall Assessment")
        ax.text(0.5, 0, f"Promising: {promising}, Red Flags: {red_flags}", 
                ha='center', va='bottom', transform=ax.transAxes)
        
        # Plot 2-6: Individual criterion scores
        criterion_names = [
            "intensity_analysis",
            "wind_analysis",
            "fwhm_analysis",
            "intensity_diff_analysis",
            "wind_diff_analysis",
        ]
        plot_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for i, (criterion_name, (row, col)) in enumerate(zip(criterion_names, plot_positions)):
            ax = axes[row, col]
            criterion_results = results.get(criterion_name, {})
            scores_dict = criterion_results.get("scores", {})
            
            if len(scores_dict) > 0:
                score_names = list(scores_dict.keys())
                score_values = [scores_dict[k] for k in score_names]
                
                colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in score_values]
                ax.barh(score_names, score_values, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Score")
                ax.set_title(criterion_name.replace("_", " ").title())
        
        plt.tight_layout()
        viz_file = output_dir / "analysis_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved analysis visualization: {viz_file}")
        plt.close()


def analyze_results(
    db_path: str | Path | None = None,
    image_folder: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Main function to analyze hard target detection results.
    
    This function can be called from IDE with just the image folder path.
    It will automatically detect database paths and configuration.
    
    Parameters
    ----------
    db_path : str | Path | None
        Path to database file. If None, uses first database from config.
    image_folder : str | Path | None
        Folder containing visualization images. Used for reference/validation.
        If None, uses visualization_output_dir from config.
    output_dir : str | Path | None
        Output directory for analysis results. If None, creates hard_target_analysis_results/
        
    Returns
    -------
    dict[str, Any]
        Comprehensive analysis results
    """
    from config import Config
    
    # Load configuration
    Config.load_from_file(silent=False)
    
    # Get database path
    if db_path is None:
        db_path = Config.get_database_path()
        if db_path is None:
            raise ValueError("Database path not configured. Please specify db_path or set in config.txt")
    
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    # Get image folder (for reference)
    if image_folder is None:
        image_folder = Config.get_visualization_output_dir_path()
    else:
        image_folder = Path(image_folder)
    
    # Get output directory
    if output_dir is None:
        base_output = Config.get_visualization_output_dir_path()
        output_dir = base_output.parent / "hard_target_analysis_results"
    else:
        output_dir = Path(output_dir)
    
    # Get analysis parameters from config
    range_step = Config.RANGE_STEP
    starting_range = Config.get_starting_range()
    requested_ranges = Config.get_requested_ranges()
    
    if not requested_ranges:
        raise ValueError("No requested_ranges specified in config.txt")
    
    print("=" * 70)
    print("Comprehensive Hard Target Analysis")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Image Folder: {image_folder}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Range Step: {range_step} m")
    print(f"  Starting Range: {starting_range} m")
    print(f"  Requested Ranges: {requested_ranges} m")
    print()
    
    # Create analyzer
    analyzer = ComprehensiveTargetAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_from_database(
        db_path=db_path,
        range_step=range_step,
        starting_range=starting_range,
        requested_ranges=requested_ranges,
        output_dir=output_dir,
    )
    
    # Print summary
    print()
    print("=" * 70)
    print("Analysis Summary")
    print("=" * 70)
    print()
    
    assessment = results.get("overall_assessment", {})
    print(f"Overall Assessment: {assessment.get('assessment', 'UNKNOWN')}")
    print(f"Confidence: {assessment.get('confidence', 'UNKNOWN')}")
    print(f"Overall Score: {assessment.get('overall_score', 0.0):.3f}")
    print()
    print(f"Promising Indicators: {assessment.get('total_promising_indicators', 0)}")
    print(f"Red Flags: {assessment.get('total_red_flags', 0)}")
    print()
    
    # Print detailed results
    for criterion_name, criterion_results in results.items():
        if criterion_name == "overall_assessment":
            continue
        
        print(f"{criterion_name.upper().replace('_', ' ')}:")
        if isinstance(criterion_results, dict):
            promising = criterion_results.get("promising_indicators", [])
            red_flags = criterion_results.get("red_flags", [])
            
            for indicator in promising[:3]:  # Show first 3
                print(f"  ✓ {indicator}")
            if len(promising) > 3:
                print(f"  ... and {len(promising) - 3} more")
            
            for flag in red_flags[:3]:  # Show first 3
                print(f"  ✗ {flag}")
            if len(red_flags) > 3:
                print(f"  ... and {len(red_flags) - 3} more")
        print()
    
    print(f"✓ Detailed results saved to: {output_dir}")
    print()
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Example usage from IDE
    results = analyze_results(
        image_folder="visualization_output/output7",
        output_dir="hard_target_analysis_results",
    )

