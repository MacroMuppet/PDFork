# Improved Chart and Table Detection

This document describes the enhanced chart and table detection capabilities implemented in the PDF processing pipeline.

## Overview

The improved detection system uses a multi-feature confidence scoring approach to accurately classify visual elements in PDF documents as charts, tables, or regular images. This replaces the previous basic detection that relied primarily on line detection and aspect ratio constraints.

## Key Improvements

1. **Multi-feature Analysis**: Instead of relying on a single feature, the system now analyzes multiple characteristics of each visual element:
   - For charts: line patterns, color distribution, text regions, and histogram features
   - For tables: grid structure, text alignment, and row uniformity

2. **Confidence Scoring**: Each feature contributes a weighted score to the overall confidence, allowing for more nuanced classification.

3. **Removed Aspect Ratio Limitation**: Charts and tables are no longer required to be nearly square (0.8-1.2 aspect ratio).

4. **Small Image Filtering**: Very small images (< 100x100 pixels) are automatically filtered out as they're likely icons or decorations.

5. **Enhanced Preprocessing**: Improved image preprocessing with adaptive thresholding and noise reduction for better feature detection.

6. **Detailed Debugging**: The system provides detailed confidence scores to help troubleshoot classification issues.

## Feature Details

### Chart Detection Features

1. **Line Detection** (30% weight)
   - Identifies axes and grid lines common in charts
   - Analyzes line angles to detect horizontal and vertical axes

2. **Color Distribution** (20% weight)
   - Analyzes color histograms to identify distinct color groups
   - Charts typically have 3-7 distinct color groups

3. **Text Region Analysis** (20% weight)
   - Detects text regions that might represent labels, legends, or titles
   - Analyzes text alignment patterns characteristic of charts

4. **Histogram Features** (30% weight)
   - Identifies rectangular shapes that might represent bars or columns
   - Analyzes uniformity of shapes to detect bar/column charts

### Table Detection Features

1. **Grid Structure** (40% weight)
   - Detects grid-like patterns of intersecting lines
   - Counts line intersections to identify cell structures

2. **Text Alignment** (30% weight)
   - Analyzes alignment of text regions into rows and columns
   - Groups text by similar x and y coordinates

3. **Row Uniformity** (30% weight)
   - Measures consistency of row heights
   - Uses statistical analysis to detect uniform spacing

## Testing the Improved Detection

A test script `test_visual_detection.py` is provided to evaluate the detection performance on individual PDF files.

### Usage

```bash
python test_visual_detection.py path/to/your/document.pdf [--output OUTPUT_DIR] [--no-save]
```

Options:
- `--output`, `-o`: Specify output directory for saving annotated images (default: "detection_results")
- `--no-save`: Don't save annotated images, just print analysis results

### Output

The script provides:
1. Detailed analysis of each image with confidence scores for all features
2. Annotated images showing the classification result
3. Summary statistics of charts, tables, and regular images detected

### Example

```bash
python test_visual_detection.py sample_report.pdf --output results
```

## Integration

The improved detection is fully integrated into the main PDF processing pipeline in `PDFLCEmbed_semantchunk_mapper.py`. The system maintains backward compatibility through legacy functions that use the new confidence scoring internally.

## Future Improvements

Potential future enhancements:
1. Machine learning-based classification using the confidence scores as features
2. Specialized detectors for pie charts, scatter plots, and other chart types
3. OCR integration to analyze text content within tables
4. Improved handling of low-quality or scanned documents 