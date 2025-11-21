"""Hard Target Detection Algorithm.

This module implements a sophisticated multi-criteria detection system for identifying
hard targets (solid objects) that could be hit by the lidar beam.

Detection Criteria:
1. SNR Anomaly Detection: High SNR values indicate strong reflections from solid objects
2. Difference Gradient Analysis: Abrupt changes in sequential differences suggest hard boundaries
3. Spatial Consistency: Targets should appear consistently across multiple ranges
4. FWHM Analysis: Narrow FWHM values indicate coherent reflections from solid surfaces
5. Statistical Outlier Detection: Values significantly above background levels
6. Spatial Clustering: Grouped detections in azimuth/elevation space
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage, stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from data_reader.storage.database import DataDatabase


class HardTargetDetector:
    """
    Sophisticated hard target detection system using multiple analysis criteria.
    
    This detector analyzes lidar data to identify solid objects that could be hit
    by the lidar beam. It uses a combination of signal processing, statistical analysis,
    and pattern recognition techniques.
    """
    
    def __init__(
        self,
        snr_threshold_percentile: float = 95.0,
        difference_threshold_percentile: float = 90.0,
        spatial_consistency_threshold: int = 3,
        fwhm_threshold_percentile: float = 20.0,  # Lower FWHM = more coherent = hard target
        outlier_z_score: float = 2.5,
        min_cluster_size: int = 3,
        range_consistency_window: float = 200.0,  # meters
    ):
        """
        Initialize the hard target detector with configurable thresholds.
        
        Parameters
        ----------
        snr_threshold_percentile : float
            Percentile threshold for SNR values (higher = stronger reflection)
        difference_threshold_percentile : float
            Percentile threshold for sequential differences (abrupt changes)
        spatial_consistency_threshold : int
            Minimum number of ranges where target must appear
        fwhm_threshold_percentile : float
            Percentile threshold for FWHM (lower = more coherent reflection)
        outlier_z_score : float
            Z-score threshold for statistical outlier detection
        min_cluster_size : int
            Minimum number of points in a spatial cluster
        range_consistency_window : float
            Range window (meters) for checking spatial consistency
        """
        self.snr_threshold_percentile = snr_threshold_percentile
        self.difference_threshold_percentile = difference_threshold_percentile
        self.spatial_consistency_threshold = spatial_consistency_threshold
        self.fwhm_threshold_percentile = fwhm_threshold_percentile
        self.outlier_z_score = outlier_z_score
        self.min_cluster_size = min_cluster_size
        self.range_consistency_window = range_consistency_window
    
    def detect_from_database(
        self,
        db_path: str | Path,
        range_step: float,
        starting_range: float,
        requested_ranges: list[float],
    ) -> dict[str, Any]:
        """
        Detect hard targets by analyzing database data directly.
        
        This is more accurate than image analysis as it uses raw data values.
        
        Parameters
        ----------
        db_path : str | Path
            Path to the database file
        range_step : float
            Spacing between range bins in meters
        starting_range : float
            Starting range in meters
        requested_ranges : list[float]
            List of ranges to analyze
            
        Returns
        -------
        dict[str, Any]
            Detection results with targets, scores, and metadata
        """
        db = DataDatabase(db_path)
        try:
            db.connect()
            records = db.query_timestamp_range()
        finally:
            db.close()
        
        if len(records) == 0:
            return {
                "targets": [],
                "scores": {},
                "metadata": {"total_records": 0, "ranges_analyzed": len(requested_ranges)},
            }
        
        # Extract data for all ranges
        all_data = {}
        for rng in requested_ranges:
            all_data[rng] = self._extract_range_data(records, range_step, starting_range, rng)
        
        # Apply detection criteria
        detection_results = self._apply_detection_criteria(all_data, requested_ranges)
        
        return detection_results
    
    def _extract_range_data(
        self,
        records: list[dict[str, Any]],
        range_step: float,
        starting_range: float,
        requested_range: float,
    ) -> dict[str, np.ndarray]:
        """Extract data for a specific range from records."""
        try:
            from data_reader.analysis.visualization import extract_range_values
        except ImportError:
            # Fallback: direct calculation
            def extract_range_values(profile_array, range_step, starting_range, requested_ranges):
                if profile_array is None or len(profile_array) == 0:
                    return {rng: None for rng in requested_ranges}
                result = {}
                for req_range in requested_ranges:
                    index_float = (req_range - starting_range) / range_step
                    index = int(np.round(index_float))
                    if 0 <= index < len(profile_array):
                        result[req_range] = float(profile_array[index])
                    else:
                        result[req_range] = None
                return result
        
        azimuth_list = []
        elevation_list = []
        snr_list = []
        wind_list = []
        fwhm_list = []
        
        for record in records:
            azimuth = record.get("azimuth")
            elevation = record.get("elevation")
            
            if azimuth is None or elevation is None:
                continue
            
            # Extract SNR (peak) value
            peak_profile = record.get("peak_profile")
            snr_value = None
            if peak_profile is not None:
                snr_dict = extract_range_values(peak_profile, range_step, starting_range, [requested_range])
                snr_value = snr_dict.get(requested_range)
            
            # Extract wind value
            wind_profile = record.get("wind_profile")
            wind_value = None
            if wind_profile is not None:
                wind_dict = extract_range_values(wind_profile, range_step, starting_range, [requested_range])
                wind_value = wind_dict.get(requested_range)
            
            # Extract FWHM value
            fwhm_profile = record.get("fwhm_profile")
            fwhm_value = None
            if fwhm_profile is not None:
                fwhm_dict = extract_range_values(fwhm_profile, range_step, starting_range, [requested_range])
                fwhm_value = fwhm_dict.get(requested_range)
            
            if snr_value is not None:
                azimuth_list.append(float(azimuth))
                elevation_list.append(float(elevation))
                snr_list.append(float(snr_value))
                wind_list.append(float(wind_value) if wind_value is not None else np.nan)
                fwhm_list.append(float(fwhm_value) if fwhm_value is not None else np.nan)
        
        return {
            "azimuth": np.array(azimuth_list),
            "elevation": np.array(elevation_list),
            "snr": np.array(snr_list),
            "wind": np.array(wind_list),
            "fwhm": np.array(fwhm_list),
        }
    
    def _apply_detection_criteria(
        self,
        all_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> dict[str, Any]:
        """Apply all detection criteria and combine results."""
        
        # Criterion 1: SNR Anomaly Detection
        snr_detections = self._detect_snr_anomalies(all_data, requested_ranges)
        
        # Criterion 2: Difference Gradient Analysis
        difference_detections = self._detect_difference_gradients(all_data, requested_ranges)
        
        # Criterion 3: FWHM Analysis
        fwhm_detections = self._detect_fwhm_patterns(all_data, requested_ranges)
        
        # Criterion 4: Statistical Outlier Detection
        outlier_detections = self._detect_statistical_outliers(all_data, requested_ranges)
        
        # Criterion 5: Spatial Consistency Analysis
        consistency_detections = self._detect_spatial_consistency(
            snr_detections, difference_detections, fwhm_detections, requested_ranges
        )
        
        # Criterion 6: Spatial Clustering
        clustered_targets = self._cluster_detections(consistency_detections)
        
        # Combine all criteria with weighted scoring
        final_targets = self._combine_detection_scores(
            snr_detections,
            difference_detections,
            fwhm_detections,
            outlier_detections,
            clustered_targets,
            requested_ranges,
        )
        
        return {
            "targets": final_targets,
            "scores": {
                "snr_detections": len(snr_detections),
                "difference_detections": len(difference_detections),
                "fwhm_detections": len(fwhm_detections),
                "outlier_detections": len(outlier_detections),
                "final_targets": len(final_targets),
            },
            "metadata": {
                "ranges_analyzed": len(requested_ranges),
                "total_data_points": sum(len(data["azimuth"]) for data in all_data.values()),
            },
        }
    
    def _detect_snr_anomalies(
        self,
        all_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> list[dict[str, Any]]:
        """Detect high SNR values indicating strong reflections."""
        detections = []
        
        for rng in requested_ranges:
            data = all_data.get(rng)
            if data is None or len(data["snr"]) == 0:
                continue
            
            snr_values = data["snr"]
            threshold = np.nanpercentile(snr_values, self.snr_threshold_percentile)
            
            # Find points above threshold
            high_snr_mask = snr_values > threshold
            
            for i in np.where(high_snr_mask)[0]:
                detections.append({
                    "range": rng,
                    "azimuth": data["azimuth"][i],
                    "elevation": data["elevation"][i],
                    "snr": snr_values[i],
                    "criterion": "high_snr",
                    "score": (snr_values[i] - threshold) / (np.nanmax(snr_values) - threshold + 1e-10),
                })
        
        return detections
    
    def _detect_difference_gradients(
        self,
        all_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> list[dict[str, Any]]:
        """Detect abrupt changes in sequential differences (hard boundaries)."""
        detections = []
        
        # Sort ranges
        sorted_ranges = sorted(requested_ranges)
        
        for i in range(len(sorted_ranges) - 1):
            rng1, rng2 = sorted_ranges[i], sorted_ranges[i + 1]
            data1 = all_data.get(rng1)
            data2 = all_data.get(rng2)
            
            if data1 is None or data2 is None:
                continue
            
            # Match points by azimuth/elevation (simplified - use nearest neighbor)
            # For each point in range1, find corresponding point in range2
            for j, (az1, el1, snr1) in enumerate(zip(data1["azimuth"], data1["elevation"], data1["snr"])):
                # Find nearest point in range2
                distances = np.sqrt(
                    (data2["azimuth"] - az1) ** 2 + (data2["elevation"] - el1) ** 2
                )
                nearest_idx = np.argmin(distances)
                
                if distances[nearest_idx] > 5.0:  # Too far, skip
                    continue
                
                snr2 = data2["snr"][nearest_idx]
                difference = abs(snr2 - snr1)
                
                # Calculate threshold based on all differences
                all_diffs = []
                for k in range(len(data1["snr"])):
                    dists = np.sqrt(
                        (data2["azimuth"] - data1["azimuth"][k]) ** 2 +
                        (data2["elevation"] - data1["elevation"][k]) ** 2
                    )
                    if np.min(dists) <= 5.0:
                        all_diffs.append(abs(data2["snr"][np.argmin(dists)] - data1["snr"][k]))
                
                if len(all_diffs) > 0:
                    threshold = np.percentile(all_diffs, self.difference_threshold_percentile)
                    
                    if difference > threshold:
                        detections.append({
                            "range": (rng1 + rng2) / 2,
                            "azimuth": az1,
                            "elevation": el1,
                            "difference": difference,
                            "criterion": "abrupt_change",
                            "score": (difference - threshold) / (np.max(all_diffs) - threshold + 1e-10),
                        })
        
        return detections
    
    def _detect_fwhm_patterns(
        self,
        all_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> list[dict[str, Any]]:
        """Detect narrow FWHM values indicating coherent reflections."""
        detections = []
        
        for rng in requested_ranges:
            data = all_data.get(rng)
            if data is None or len(data["fwhm"]) == 0 or np.all(np.isnan(data["fwhm"])):
                continue
            
            fwhm_values = data["fwhm"][~np.isnan(data["fwhm"])]
            if len(fwhm_values) == 0:
                continue
            
            # Lower FWHM = more coherent = hard target
            threshold = np.nanpercentile(fwhm_values, self.fwhm_threshold_percentile)
            
            low_fwhm_mask = data["fwhm"] < threshold
            low_fwhm_mask = low_fwhm_mask & ~np.isnan(data["fwhm"])
            
            for i in np.where(low_fwhm_mask)[0]:
                detections.append({
                    "range": rng,
                    "azimuth": data["azimuth"][i],
                    "elevation": data["elevation"][i],
                    "fwhm": data["fwhm"][i],
                    "criterion": "low_fwhm",
                    "score": (threshold - data["fwhm"][i]) / (threshold - np.nanmin(fwhm_values) + 1e-10),
                })
        
        return detections
    
    def _detect_statistical_outliers(
        self,
        all_data: dict[float, dict[str, np.ndarray]],
        requested_ranges: list[float],
    ) -> list[dict[str, Any]]:
        """Detect statistical outliers using Z-score analysis."""
        detections = []
        
        for rng in requested_ranges:
            data = all_data.get(rng)
            if data is None or len(data["snr"]) == 0:
                continue
            
            snr_values = data["snr"]
            
            # Calculate Z-scores
            mean_snr = np.nanmean(snr_values)
            std_snr = np.nanstd(snr_values)
            
            if std_snr == 0:
                continue
            
            z_scores = (snr_values - mean_snr) / std_snr
            
            outlier_mask = z_scores > self.outlier_z_score
            
            for i in np.where(outlier_mask)[0]:
                detections.append({
                    "range": rng,
                    "azimuth": data["azimuth"][i],
                    "elevation": data["elevation"][i],
                    "snr": snr_values[i],
                    "z_score": z_scores[i],
                    "criterion": "statistical_outlier",
                    "score": min(z_scores[i] / (self.outlier_z_score * 2), 1.0),
                })
        
        return detections
    
    def _detect_spatial_consistency(
        self,
        snr_detections: list[dict[str, Any]],
        difference_detections: list[dict[str, Any]],
        fwhm_detections: list[dict[str, Any]],
        requested_ranges: list[float],
    ) -> list[dict[str, Any]]:
        """Detect targets that appear consistently across multiple ranges."""
        # Combine all detections
        all_detections = snr_detections + difference_detections + fwhm_detections
        
        if len(all_detections) == 0:
            return []
        
        # Group detections by spatial location (azimuth/elevation)
        spatial_groups = {}
        
        for det in all_detections:
            # Round to nearest 5 degrees for grouping
            az_key = round(det["azimuth"] / 5.0) * 5.0
            el_key = round(det["elevation"] / 5.0) * 5.0
            key = (az_key, el_key)
            
            if key not in spatial_groups:
                spatial_groups[key] = []
            spatial_groups[key].append(det)
        
        # Find groups that appear in multiple ranges
        consistent_targets = []
        for key, group in spatial_groups.items():
            ranges_in_group = set(d["range"] for d in group)
            
            if len(ranges_in_group) >= self.spatial_consistency_threshold:
                # Calculate average properties
                avg_az = np.mean([d["azimuth"] for d in group])
                avg_el = np.mean([d["elevation"] for d in group])
                avg_range = np.mean([d["range"] for d in group])
                avg_score = np.mean([d.get("score", 0.5) for d in group])
                
                consistent_targets.append({
                    "azimuth": avg_az,
                    "elevation": avg_el,
                    "range": avg_range,
                    "range_span": (max(ranges_in_group) - min(ranges_in_group)),
                    "detection_count": len(group),
                    "ranges_detected": sorted(ranges_in_group),
                    "criterion": "spatial_consistency",
                    "score": avg_score * (len(ranges_in_group) / len(requested_ranges)),
                })
        
        return consistent_targets
    
    def _cluster_detections(
        self,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Cluster detections spatially to identify target groups."""
        if len(detections) < self.min_cluster_size:
            return detections
        
        # Extract spatial coordinates
        coords = np.array([[d["azimuth"], d["elevation"]] for d in detections])
        
        # Calculate pairwise distances
        distances = pdist(coords)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')
        
        # Cluster with distance threshold (10 degrees)
        cluster_labels = fcluster(linkage_matrix, t=10.0, criterion='distance')
        
        # Group by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(detections[i])
        
        # Process clusters
        clustered_targets = []
        for label, cluster in clusters.items():
            if len(cluster) >= self.min_cluster_size:
                # Calculate cluster center and properties
                avg_az = np.mean([d["azimuth"] for d in cluster])
                avg_el = np.mean([d["elevation"] for d in cluster])
                avg_range = np.mean([d["range"] for d in cluster])
                avg_score = np.mean([d.get("score", 0.5) for d in cluster])
                
                clustered_targets.append({
                    "azimuth": avg_az,
                    "elevation": avg_el,
                    "range": avg_range,
                    "cluster_size": len(cluster),
                    "criterion": "spatial_clustering",
                    "score": avg_score * min(len(cluster) / 10.0, 1.0),
                })
        
        return clustered_targets if clustered_targets else detections
    
    def _combine_detection_scores(
        self,
        snr_detections: list[dict[str, Any]],
        difference_detections: list[dict[str, Any]],
        fwhm_detections: list[dict[str, Any]],
        outlier_detections: list[dict[str, Any]],
        clustered_targets: list[dict[str, Any]],
        requested_ranges: list[float],
    ) -> list[dict[str, Any]]:
        """Combine all detection criteria with weighted scoring."""
        
        # Weight factors for each criterion
        weights = {
            "high_snr": 0.25,
            "abrupt_change": 0.20,
            "low_fwhm": 0.20,
            "statistical_outlier": 0.15,
            "spatial_consistency": 0.15,
            "spatial_clustering": 0.05,
        }
        
        # Create a spatial grid to aggregate scores
        az_bins = np.linspace(0, 360, 73)  # 5-degree bins
        el_bins = np.linspace(0, 90, 19)  # 5-degree bins
        
        score_grid = np.zeros((len(el_bins) - 1, len(az_bins) - 1))
        count_grid = np.zeros((len(el_bins) - 1, len(az_bins) - 1))
        
        # Aggregate SNR detections
        for det in snr_detections:
            az_idx = np.digitize(det["azimuth"], az_bins) - 1
            el_idx = np.digitize(det["elevation"], el_bins) - 1
            if 0 <= az_idx < score_grid.shape[1] and 0 <= el_idx < score_grid.shape[0]:
                score_grid[el_idx, az_idx] += det.get("score", 0.5) * weights["high_snr"]
                count_grid[el_idx, az_idx] += 1
        
        # Aggregate difference detections
        for det in difference_detections:
            az_idx = np.digitize(det["azimuth"], az_bins) - 1
            el_idx = np.digitize(det["elevation"], el_bins) - 1
            if 0 <= az_idx < score_grid.shape[1] and 0 <= el_idx < score_grid.shape[0]:
                score_grid[el_idx, az_idx] += det.get("score", 0.5) * weights["abrupt_change"]
                count_grid[el_idx, az_idx] += 1
        
        # Aggregate FWHM detections
        for det in fwhm_detections:
            az_idx = np.digitize(det["azimuth"], az_bins) - 1
            el_idx = np.digitize(det["elevation"], el_bins) - 1
            if 0 <= az_idx < score_grid.shape[1] and 0 <= el_idx < score_grid.shape[0]:
                score_grid[el_idx, az_idx] += det.get("score", 0.5) * weights["low_fwhm"]
                count_grid[el_idx, az_idx] += 1
        
        # Aggregate outlier detections
        for det in outlier_detections:
            az_idx = np.digitize(det["azimuth"], az_bins) - 1
            el_idx = np.digitize(det["elevation"], el_bins) - 1
            if 0 <= az_idx < score_grid.shape[1] and 0 <= el_idx < score_grid.shape[0]:
                score_grid[el_idx, az_idx] += det.get("score", 0.5) * weights["statistical_outlier"]
                count_grid[el_idx, az_idx] += 1
        
        # Aggregate consistent targets (higher weight)
        for det in clustered_targets:
            az_idx = np.digitize(det["azimuth"], az_bins) - 1
            el_idx = np.digitize(det["elevation"], el_bins) - 1
            if 0 <= az_idx < score_grid.shape[1] and 0 <= el_idx < score_grid.shape[0]:
                score_grid[el_idx, az_idx] += det.get("score", 0.5) * weights["spatial_consistency"]
                count_grid[el_idx, az_idx] += det.get("detection_count", 1)
        
        # Find peaks in score grid
        final_targets = []
        threshold = np.percentile(score_grid[score_grid > 0], 75) if np.any(score_grid > 0) else 0.0
        
        for el_idx in range(score_grid.shape[0]):
            for az_idx in range(score_grid.shape[1]):
                if score_grid[el_idx, az_idx] > threshold:
                    az_center = (az_bins[az_idx] + az_bins[az_idx + 1]) / 2
                    el_center = (el_bins[el_idx] + el_bins[el_idx + 1]) / 2
                    
                    final_targets.append({
                        "azimuth": az_center,
                        "elevation": el_center,
                        "confidence_score": float(score_grid[el_idx, az_idx]),
                        "detection_count": int(count_grid[el_idx, az_idx]),
                        "range": np.mean(requested_ranges),  # Average range
                    })
        
        # Sort by confidence score
        final_targets.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        return final_targets


def detect_hard_targets(
    db_path: str | Path,
    range_step: float,
    starting_range: float,
    requested_ranges: list[float],
    output_dir: str | Path | None = None,
    **detector_kwargs,
) -> dict[str, Any]:
    """
    Main function to detect hard targets from database analysis.
    
    Parameters
    ----------
    db_path : str | Path
        Path to the database file
    range_step : float
        Spacing between range bins in meters
    starting_range : float
        Starting range in meters
    requested_ranges : list[float]
        List of ranges to analyze
    output_dir : str | Path | None
        Directory to save detection results and visualizations
    **detector_kwargs
        Additional arguments passed to HardTargetDetector
        
    Returns
    -------
    dict[str, Any]
        Detection results with targets, scores, and metadata
    """
    detector = HardTargetDetector(**detector_kwargs)
    results = detector.detect_from_database(db_path, range_step, starting_range, requested_ranges)
    
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        import json
        results_file = output_path / "hard_target_detections.json"
        with open(results_file, "w") as f:
            # Convert numpy types to native Python types for JSON
            json_results = {
                "targets": [
                    {
                        "azimuth": float(t["azimuth"]),
                        "elevation": float(t["elevation"]),
                        "range": float(t["range"]),
                        "confidence_score": float(t["confidence_score"]),
                        "detection_count": int(t["detection_count"]),
                    }
                    for t in results["targets"]
                ],
                "scores": {k: int(v) for k, v in results["scores"].items()},
                "metadata": {
                    "ranges_analyzed": int(results["metadata"]["ranges_analyzed"]),
                    "total_data_points": int(results["metadata"]["total_data_points"]),
                },
            }
            json.dump(json_results, f, indent=2)
        
        # Create visualization if matplotlib is available
        if HAS_MATPLOTLIB:
            _create_detection_visualization(results, output_path, requested_ranges)
    
    return results


def _create_detection_visualization(
    results: dict[str, Any],
    output_dir: Path,
    requested_ranges: list[float],
) -> None:
    """Create visualization of detected hard targets."""
    targets = results["targets"]
    
    if len(targets) == 0:
        print("No targets detected for visualization")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
    
    # Extract coordinates
    azimuths = np.deg2rad([t["azimuth"] for t in targets])
    elevations = [t["elevation"] for t in targets]
    scores = [t["confidence_score"] for t in targets]
    
    # Create scatter plot with color-coded confidence scores
    scatter = ax.scatter(azimuths, elevations, c=scores, s=100, cmap='hot', alpha=0.7, edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Confidence Score')
    
    # Label top targets
    top_targets = sorted(targets, key=lambda x: x["confidence_score"], reverse=True)[:10]
    for i, target in enumerate(top_targets):
        az_rad = np.deg2rad(target["azimuth"])
        el = target["elevation"]
        ax.annotate(
            f"#{i+1}\n{target['confidence_score']:.2f}",
            (az_rad, el),
            fontsize=8,
            ha='center',
        )
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    ax.set_ylim(0, 90)
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Elevation (degrees)', labelpad=20)
    ax.set_title(f'Hard Target Detections\n{len(targets)} targets detected across {len(requested_ranges)} ranges', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    output_file = output_dir / "hard_target_detections.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved detection visualization: {output_file}")
    plt.close()

