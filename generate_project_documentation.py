"""Generate PDF documentation for CUAV Field Tests Data Reader project.

This script creates a comprehensive PDF document describing the project,
including dataflow, analysis methods, and system architecture.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        Table,
        TableStyle,
        Image,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
except ImportError:
    print("ERROR: reportlab is required to generate PDF.")
    print("Install with: pip install reportlab")
    import sys
    sys.exit(1)


def create_pdf(output_path: str | Path = "project_documentation.pdf"):
    """Create PDF documentation for the project."""
    output_path = Path(output_path)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    
    # Container for PDF elements
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    
    heading1_style = ParagraphStyle(
        "CustomHeading1",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
        spaceBefore=20,
        fontName="Helvetica-Bold",
    )
    
    heading2_style = ParagraphStyle(
        "CustomHeading2",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=10,
        spaceBefore=15,
        fontName="Helvetica-Bold",
    )
    
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
    )
    
    code_style = ParagraphStyle(
        "CustomCode",
        parent=styles["Code"],
        fontSize=9,
        leading=11,
        leftIndent=20,
        rightIndent=20,
        backColor=colors.HexColor("#f5f5f5"),
        borderColor=colors.HexColor("#cccccc"),
        borderWidth=1,
        borderPadding=8,
    )
    
    # Title page
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("CUAV Field Tests", title_style))
    story.append(Paragraph("Data Reader Project", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("High-Level Documentation", styles["Heading2"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", heading1_style))
    toc_items = [
        "1. Project Overview",
        "2. System Architecture",
        "3. Data Flow",
        "4. Core Methods",
        "5. Data Processing Pipeline",
        "6. Database Storage",
        "7. Analysis and Visualization",
        "8. Configuration",
        "9. Usage Examples",
    ]
    for item in toc_items:
        story.append(Paragraph(item, body_style))
        story.append(Spacer(1, 0.1 * inch))
    story.append(PageBreak())
    
    # 1. Project Overview
    story.append(Paragraph("1. Project Overview", heading1_style))
    story.append(Paragraph(
        "The CUAV Field Tests Data Reader is a Python package designed for processing, "
        "analyzing, and visualizing atmospheric measurement data from Raymetrics CUAV "
        "(Coherent Doppler Lidar) field tests. The system handles data from multiple sources "
        "including processed measurement files, raw spectra files, and log files containing "
        "pointing angles and timestamps.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Key Features:</b>",
        body_style
    ))
    features = [
        "Automated matching of processed and raw data files across mirrored directory structures",
        "Timestamp-based filtering and synchronization with log files",
        "Data aggregation from multiple sources (peak, spectrum, wind, raw spectra)",
        "SQLite database for persistent storage and efficient querying",
        "Range-resolved heatmap visualization for spatial analysis",
        "Configurable processing parameters via simple text file",
    ]
    for feature in features:
        story.append(Paragraph(f"• {feature}", body_style))
    
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "<b>Main Use Cases:</b>",
        body_style
    ))
    use_cases = [
        "Match processed timestamps with corresponding raw spectra files",
        "Filter data to include only measurements present in log files",
        "Aggregate data from multiple file types into unified structures",
        "Store aggregated data in a database for persistent access",
        "Generate heatmaps showing parameter distributions across azimuth/elevation angles",
        "Analyze range-resolved profiles at specific altitudes",
    ]
    for use_case in use_cases:
        story.append(Paragraph(f"• {use_case}", body_style))
    story.append(PageBreak())
    
    # 2. System Architecture
    story.append(Paragraph("2. System Architecture", heading1_style))
    story.append(Paragraph(
        "The system is organized into modular packages, each responsible for a specific "
        "aspect of data processing:",
        body_style
    ))
    
    # Architecture table
    arch_data = [
        ["Module", "Purpose", "Key Functions"],
        [
            "<b>parsing</b>",
            "Extract metadata from filenames and log files",
            "timestamp_from_spectra_filename(), read_log_files()"
        ],
        [
            "<b>matching</b>",
            "Align processed and raw data by traversing directory trees",
            "match_processed_and_raw(), filter_matches_by_log_timestamps()"
        ],
        [
            "<b>reading</b>",
            "Load data files into NumPy arrays",
            "read_processed_data_file(), read_raw_spectra_file()"
        ],
        [
            "<b>processing</b>",
            "Aggregate, filter, and integrate data from multiple sources",
            "build_timestamp_data_dict(), build_and_save_to_database()"
        ],
        [
            "<b>storage</b>",
            "SQLite database for persistent data storage",
            "DataDatabase, query_timestamp(), query_timestamp_range()"
        ],
        [
            "<b>analysis</b>",
            "Generate visualizations and extract range-resolved data",
            "create_heatmaps(), extract_range_values()"
        ],
    ]
    
    arch_table = Table(arch_data, colWidths=[1.5 * inch, 2.5 * inch, 3 * inch])
    arch_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 0.3 * inch))
    
    story.append(Paragraph(
        "<b>Module Dependencies:</b>",
        heading2_style
    ))
    story.append(Paragraph(
        "The architecture follows a layered dependency structure: parsing → matching → reading → "
        "processing → storage → analysis. Lower-level modules (parsing, matching, reading) provide "
        "foundational data access, while higher-level modules (processing, storage, analysis) build "
        "upon these foundations to provide advanced functionality.",
        body_style
    ))
    story.append(PageBreak())
    
    # 3. Data Flow
    story.append(Paragraph("3. Data Flow", heading1_style))
    story.append(Paragraph(
        "The system processes data through several stages, from raw file discovery to final "
        "visualization. The following diagram illustrates the data flow:",
        body_style
    ))
    
    # Data flow diagram (text representation)
    flow_diagram = """
    ┌─────────────────┐
    │ Processed Files │
    │ (_Peak.txt,     │
    │ _Spectrum.txt,  │
    │ _Wind.txt)      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐      ┌──────────────┐
    │ Extract         │◄─────│ Log File     │
    │ Timestamps      │      │ (output.txt) │
    └────────┬────────┘      └──────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Match with Raw  │
    │ Spectra Files   │
    │ (spectra_*.txt) │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Filter by Log   │
    │ Timestamps      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Aggregate Data  │
    │ (peak, spectrum,│
    │ wind, azimuth,  │
    │ elevation, raw) │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Store in        │
    │ SQLite Database │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Generate        │
    │ Heatmaps        │
    └─────────────────┘
    """
    
    story.append(Paragraph("<pre>" + flow_diagram + "</pre>", code_style))
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph(
        "<b>Data Flow Stages:</b>",
        heading2_style
    ))
    
    stages = [
        ("<b>1. File Discovery:</b>", 
         "Traverse processed and raw directory trees. Processed trees contain files like "
         "_Peak.txt, _Spectrum.txt, _Wind.txt. Raw trees contain spectra_*.txt files."),
        ("<b>2. Timestamp Extraction:</b>", 
         "Extract timestamps from processed files (third column) and from raw filenames "
         "(parsed from datetime format: spectra_YYYY-MM-DD_HH-MM-SS.ff.txt)."),
        ("<b>3. Matching:</b>", 
         "Pair processed timestamps with raw files based on index order within each "
         "directory pair. Each processed timestamp corresponds to one raw file."),
        ("<b>4. Filtering:</b>", 
         "Keep only matches where the processed timestamp exists in the log file "
         "(within tolerance). This ensures data consistency."),
        ("<b>5. Aggregation:</b>", 
         "For each filtered timestamp, collect data from: processed files (peak, spectrum, "
         "wind), log file (azimuth, elevation), and raw spectra files (power density spectrum)."),
        ("<b>6. Storage:</b>", 
         "Save aggregated data to SQLite database with timestamp as primary key. Arrays "
         "are stored as JSON strings."),
        ("<b>7. Visualization:</b>", 
         "Query database, extract range-resolved values at specific ranges, aggregate by "
         "azimuth/elevation, and generate heatmaps."),
    ]
    
    for stage_title, stage_desc in stages:
        story.append(Paragraph(stage_title, body_style))
        story.append(Paragraph(stage_desc, body_style))
        story.append(Spacer(1, 0.15 * inch))
    
    story.append(PageBreak())
    
    # 4. Core Methods
    story.append(Paragraph("4. Core Methods", heading1_style))
    
    methods_data = [
        ["Category", "Function", "Description"],
        [
            "<b>Parsing</b>",
            "timestamp_from_spectra_filename()",
            "Extract datetime from spectra filename format"
        ],
        [
            "",
            "read_log_files()",
            "Load log file (columns: azimuth, elevation, timestamp)"
        ],
        [
            "<b>Matching</b>",
            "match_processed_and_raw()",
            "Walk directory trees and pair processed/raw files"
        ],
        [
            "",
            "filter_matches_by_log_timestamps()",
            "Keep only matches present in log file"
        ],
        [
            "<b>Reading</b>",
            "read_processed_data_file()",
            "Load processed files into NumPy arrays"
        ],
        [
            "",
            "read_raw_spectra_file()",
            "Load raw spectra (skip first 13 lines for ASCII)"
        ],
        [
            "<b>Processing</b>",
            "build_timestamp_data_dict()",
            "Aggregate data from all sources per timestamp"
        ],
        [
            "",
            "build_and_save_to_database()",
            "Build and save aggregated data to database"
        ],
        [
            "<b>Storage</b>",
            "query_timestamp()",
            "Query single timestamp from database"
        ],
        [
            "",
            "query_timestamp_range()",
            "Query range of timestamps from database"
        ],
        [
            "<b>Analysis</b>",
            "extract_range_values()",
            "Extract values from range-resolved profile"
        ],
        [
            "",
            "create_heatmaps()",
            "Generate heatmaps for parameters at specific ranges"
        ],
    ]
    
    methods_table = Table(methods_data, colWidths=[1.2 * inch, 2.8 * inch, 4 * inch])
    methods_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ]))
    story.append(methods_table)
    story.append(PageBreak())
    
    # 5. Data Processing Pipeline
    story.append(Paragraph("5. Data Processing Pipeline", heading1_style))
    
    story.append(Paragraph(
        "<b>5.1 Matching Process</b>",
        heading2_style
    ))
    story.append(Paragraph(
        "The matching process pairs processed data files with raw spectra files by traversing "
        "mirrored directory structures. Both trees follow the pattern: "
        "<i>Wind/YYYY-MM-DD/MM-DD_HHh/MM-DD_HH-##/</i>",
        body_style
    ))
    story.append(Paragraph(
        "Within each matched directory pair, files are sorted by timestamp and paired by index. "
        "Each processed timestamp (from _Peak.txt, third column) is matched with the corresponding "
        "raw file (spectra_*.txt) at the same index position.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>5.2 Timestamp Filtering</b>",
        heading2_style
    ))
    story.append(Paragraph(
        "Filtering ensures data consistency by keeping only timestamps that exist in the log file. "
        "The process uses tolerance-based matching (default: 0.0001 seconds) to account for "
        "floating-point precision differences. Timestamps are normalized to 6 decimal places before "
        "comparison.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>5.3 Data Aggregation</b>",
        heading2_style
    ))
    story.append(Paragraph(
        "For each filtered timestamp, the system aggregates data from multiple sources:",
        body_style
    ))
    aggregation_items = [
        "<b>Azimuth & Elevation:</b> Retrieved from log file by matching timestamp",
        "<b>Peak Data:</b> Read from _Peak.txt file, starting from column 4 (range-resolved SNR)",
        "<b>Spectrum Data:</b> Read from _Spectrum.txt file, starting from column 4 (range-resolved spectrum)",
        "<b>Wind Data:</b> Read from _Wind.txt file, starting from column 4 (range-resolved wind velocity)",
        "<b>Power Density Spectrum:</b> Read from raw spectra file (ASCII .txt), skipping first 13 header lines",
    ]
    for item in aggregation_items:
        story.append(Paragraph(f"• {item}", body_style))
    
    story.append(Paragraph(
        "The aggregated data is organized into a nested dictionary structure, with timestamps as "
        "keys and dictionaries containing all parameters as values.",
        body_style
    ))
    story.append(PageBreak())
    
    # 6. Database Storage
    story.append(Paragraph("6. Database Storage", heading1_style))
    
    story.append(Paragraph(
        "The system uses SQLite for persistent storage, enabling efficient querying and long-term "
        "data retention. The database schema is normalized with separate tables for different data types.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>6.1 Database Schema</b>",
        heading2_style
    ))
    
    schema_data = [
        ["Table", "Columns", "Purpose"],
        [
            "<b>timestamps</b>",
            "timestamp (PK), azimuth, elevation,<br/>source_processed_dir, source_raw_dir,<br/>source_log_file, imported_at, updated_at",
            "Main table storing timestamp metadata"
        ],
        [
            "<b>peak_data</b>",
            "timestamp (FK), data (JSON)",
            "Range-resolved peak/SNR data"
        ],
        [
            "<b>spectrum_data</b>",
            "timestamp (FK), data (JSON)",
            "Range-resolved spectrum data"
        ],
        [
            "<b>wind_data</b>",
            "timestamp (FK), data (JSON)",
            "Range-resolved wind velocity data"
        ],
        [
            "<b>power_density_spectrum</b>",
            "timestamp (FK), data (JSON)",
            "Raw power density spectrum data"
        ],
    ]
    
    schema_table = Table(schema_data, colWidths=[1.8 * inch, 3 * inch, 3.2 * inch])
    schema_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
    ]))
    story.append(schema_table)
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph(
        "<b>6.2 Key Features</b>",
        heading2_style
    ))
    db_features = [
        "Fast timestamp-based lookups using primary key indexing",
        "Range queries for time-series analysis",
        "Automatic database creation and schema initialization",
        "Support for updating existing entries without duplicates",
        "Foreign key constraints ensure data integrity",
        "Array data stored as JSON for efficient storage and retrieval",
        "Metadata tracking (source files, import/update timestamps)",
    ]
    for feature in db_features:
        story.append(Paragraph(f"• {feature}", body_style))
    
    story.append(PageBreak())
    
    # 7. Analysis and Visualization
    story.append(Paragraph("7. Analysis and Visualization", heading1_style))
    
    story.append(Paragraph(
        "<b>7.1 Range-Resolved Profile Analysis</b>",
        heading2_style
    ))
    story.append(Paragraph(
        "The system processes range-resolved profiles, where each parameter (wind, peak, spectrum) "
        "is a 1D array representing measurements at different altitudes. The range resolution is "
        "defined by:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Range Step:</b> Spacing between range bins (default: 48 m)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Starting Range:</b> Range corresponding to first bin (default: -1400 m)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Requested Ranges:</b> Specific ranges to extract (e.g., [100, 200, 300] m)",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>7.2 Heatmap Generation</b>",
        heading2_style
    ))
    story.append(Paragraph(
        "Heatmaps visualize parameter distributions across azimuth and elevation angles at specific "
        "ranges. The generation process:",
        body_style
    ))
    heatmap_steps = [
        "Query all data from database for the specified parameter",
        "Extract values at the requested range for each timestamp",
        "Group data by azimuth and elevation angles",
        "Create a 2D grid by binning azimuth/elevation values",
        "Compute mean values for each grid cell",
        "Generate heatmap visualization with color-coded values",
        "Save plot as image file (PNG, PDF, SVG, etc.)",
        "Display plot interactively",
    ]
    for i, step in enumerate(heatmap_steps, 1):
        story.append(Paragraph(f"{i}. {step}", body_style))
    
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "<b>7.3 Supported Parameters</b>",
        heading2_style
    ))
    param_data = [
        ["Parameter", "Source File", "Description"],
        ["wind", "_Wind.txt", "Wind velocity profile (range-resolved)"],
        ["peak (SNR)", "_Peak.txt", "Signal-to-noise ratio profile (range-resolved)"],
        ["spectrum", "_Spectrum.txt", "Spectral power profile (range-resolved)"],
    ]
    param_table = Table(param_data, colWidths=[1.5 * inch, 2 * inch, 4.5 * inch])
    param_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
    ]))
    story.append(param_table)
    story.append(PageBreak())
    
    # 8. Configuration
    story.append(Paragraph("8. Configuration", heading1_style))
    
    story.append(Paragraph(
        "The system uses a simple text-based configuration file (config.txt) with key=value pairs. "
        "This allows easy customization without modifying code.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>8.1 Key Configuration Parameters</b>",
        heading2_style
    ))
    
    config_params = [
        ("<b>Directory Paths:</b>", [
            "processed_root: Root directory for processed data files",
            "raw_root: Root directory for raw spectra files",
            "log_file: Path to log file (output.txt)",
            "database_path: SQLite database file path",
            "visualization_output_dir: Output directory for heatmaps",
        ]),
        ("<b>Processing Parameters:</b>", [
            "timestamp_tolerance: Tolerance for timestamp matching (default: 0.0001 s)",
            "timestamp_precision: Decimal places for normalization (default: 6)",
            "processed_suffix: Suffix for processed files (default: _Peak.txt)",
            "raw_file_pattern: Pattern for raw files (default: spectra_*.txt)",
            "raw_spectra_skip_rows: Header lines to skip (default: 13)",
        ]),
        ("<b>Visualization Parameters:</b>", [
            "range_step: Spacing between range bins (default: 48.0 m)",
            "starting_range: Starting range for profiles (default: -1400.0 m)",
            "requested_ranges: Comma-separated ranges to visualize (e.g., 100,200,300)",
            "heatmap_colormap: Matplotlib colormap name (default: viridis)",
            "heatmap_format: Image format (default: png)",
        ]),
        ("<b>Execution Mode:</b>", [
            "run_mode: Main script mode (test or heatmaps)",
            "heatmap_parameters: Comma-separated parameters (e.g., wind,snr)",
        ]),
    ]
    
    for category, params in config_params:
        story.append(Paragraph(category, body_style))
        for param in params:
            story.append(Paragraph(f"  • {param}", body_style))
        story.append(Spacer(1, 0.15 * inch))
    
    story.append(PageBreak())
    
    # 9. Usage Examples
    story.append(Paragraph("9. Usage Examples", heading1_style))
    
    story.append(Paragraph(
        "<b>9.1 Running Tests</b>",
        heading2_style
    ))
    story.append(Paragraph(
        "To run the complete test suite (includes database creation):",
        body_style
    ))
    story.append(Paragraph(
        "<pre>python main.py --test</pre>",
        code_style
    ))
    
    story.append(Paragraph(
        "<b>9.2 Generating Heatmaps</b>",
        heading2_style
    ))
    story.append(Paragraph(
        "Generate heatmaps for wind and SNR at specific ranges:",
        body_style
    ))
    story.append(Paragraph(
        "<pre>python main.py --heatmaps --parameters wind snr --ranges 100 200 300</pre>",
        code_style
    ))
    
    story.append(Paragraph(
        "Use default parameters from config.txt:",
        body_style
    ))
    story.append(Paragraph(
        "<pre>python main.py --heatmaps</pre>",
        code_style
    ))
    
    story.append(Paragraph(
        "<b>9.3 Programmatic Usage</b>",
        heading2_style
    ))
    story.append(Paragraph(
        "Example Python code for matching and filtering:",
        body_style
    ))
    story.append(Paragraph(
        "<pre>from data_reader import match_processed_and_raw, filter_matches_by_log_timestamps\n"
        "\nmatches = match_processed_and_raw(processed_root, raw_root)\n"
        "filtered = filter_matches_by_log_timestamps(matches, log_file)\n"
        "print(f'Filtered matches: {len(filtered)}')</pre>",
        code_style
    ))
    
    story.append(Paragraph(
        "Example for generating heatmaps:",
        body_style
    ))
    story.append(Paragraph(
        "<pre>from data_reader import create_heatmaps\n"
        "\nresults = create_heatmaps(\n"
        "    db_path='data/cuav_data.db',\n"
        "    range_step=48.0,\n"
        "    starting_range=-1400.0,\n"
        "    requested_ranges=[100, 200, 300],\n"
        "    parameters=['wind', 'peak'],\n"
        "    output_dir='visualization_output',\n"
        "    colormap='viridis',\n"
        "    save_format='png'\n"
        ")</pre>",
        code_style
    ))
    
    story.append(PageBreak())
    
    # Summary
    story.append(Paragraph("Summary", heading1_style))
    story.append(Paragraph(
        "The CUAV Field Tests Data Reader provides a comprehensive solution for processing, "
        "storing, and analyzing atmospheric measurement data from lidar field tests. The system "
        "automates the complex task of aligning data from multiple sources, provides efficient "
        "persistent storage through SQLite, and enables sophisticated spatial analysis through "
        "range-resolved heatmaps.",
        body_style
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Key advantages of this system include:",
        body_style
    ))
    advantages = [
        "Automated data alignment across multiple file types and directory structures",
        "Robust timestamp matching with tolerance-based filtering",
        "Efficient database storage enabling fast queries and long-term data retention",
        "Flexible visualization capabilities for spatial analysis",
        "Simple configuration management through text-based config files",
        "Modular architecture enabling easy extension and maintenance",
    ]
    for advantage in advantages:
        story.append(Paragraph(f"• {advantage}", body_style))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF documentation generated: {output_path}")


if __name__ == "__main__":
    import sys
    
    output_file = "project_documentation.pdf"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    try:
        create_pdf(output_file)
    except Exception as e:
        print(f"ERROR: Failed to generate PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

