#!/usr/bin/env python3
"""
Test script for improved chart and table detection in PDFLCEmbed_semantchunk_mapper.py
This script allows testing the detection on individual PDF files and outputs detailed results.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import fitz  # PyMuPDF

# Import the detection functions from the main script
from PDFLCEmbed_semantchunk_mapper import (
    classify_visual_element,
    calculate_chart_confidence,
    calculate_table_confidence,
    _detect_chart_lines,
    _analyze_color_distribution,
    _analyze_text_regions,
    _detect_histogram_features,
    _detect_table_grid,
    _detect_text_alignment,
    _detect_uniform_rows
)

def extract_and_analyze_images(pdf_path, output_dir=None, save_images=True):
    """Extract images from PDF and analyze them with the improved detection algorithms"""
    if output_dir and save_images:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Analyzing PDF: {pdf_path}")
    print(f"{'='*80}")
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Statistics
        total_images = 0
        charts_detected = 0
        tables_detected = 0
        regular_images = 0
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            print(f"\nProcessing page {page_num+1}/{len(doc)}")
            
            # Extract images
            image_list = page.get_images()
            if not image_list:
                print(f"  No images found on page {page_num+1}")
                continue
                
            print(f"  Found {len(image_list)} image(s) on page {page_num+1}")
            
            # Process each image
            for img_index, img in enumerate(image_list):
                total_images += 1
                
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to numpy array for analysis
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None or image.size == 0:
                        print(f"  Warning: Could not decode image {img_index+1} on page {page_num+1}")
                        continue
                    
                    # Get image dimensions
                    height, width = image.shape[:2]
                    
                    # Skip very small images
                    if width < 100 or height < 100:
                        print(f"  Skipping small image {img_index+1} ({width}x{height})")
                        continue
                    
                    print(f"\n  Image {img_index+1}: {width}x{height}")
                    
                    # Calculate confidence scores
                    chart_score = calculate_chart_confidence(image)
                    table_score = calculate_table_confidence(image)
                    
                    # Get detailed feature scores
                    line_score = _detect_chart_lines(image)
                    color_score = _analyze_color_distribution(image)
                    text_score = _analyze_text_regions(image)
                    histogram_score = _detect_histogram_features(image)
                    
                    grid_score = _detect_table_grid(image)
                    alignment_score = _detect_text_alignment(image)
                    row_score = _detect_uniform_rows(image)
                    
                    # Classify the image
                    element_type = classify_visual_element(image)
                    
                    # Update statistics
                    if element_type == "chart":
                        charts_detected += 1
                    elif element_type == "table":
                        tables_detected += 1
                    else:
                        regular_images += 1
                    
                    # Print detailed results
                    print(f"  Classification: {element_type.upper()}")
                    print(f"  Chart confidence: {chart_score:.2f}, Table confidence: {table_score:.2f}")
                    print(f"  Chart features: Lines={line_score:.2f}, Color={color_score:.2f}, Text={text_score:.2f}, Histogram={histogram_score:.2f}")
                    print(f"  Table features: Grid={grid_score:.2f}, Alignment={alignment_score:.2f}, Rows={row_score:.2f}")
                    
                    # Save the image with classification info if requested
                    if save_images and output_dir:
                        # Create a copy of the image for visualization
                        vis_image = image.copy()
                        
                        # Add classification text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = f"{element_type.upper()} (C:{chart_score:.2f}, T:{table_score:.2f})"
                        cv2.putText(vis_image, text, (10, 30), font, 0.7, (0, 0, 255), 2)
                        
                        # Save the image
                        img_filename = f"page{page_num+1}_img{img_index+1}_{element_type}.png"
                        img_path = output_path / img_filename
                        cv2.imwrite(str(img_path), vis_image)
                        print(f"  Saved annotated image to: {img_path}")
                
                except Exception as e:
                    print(f"  Error processing image {img_index+1} on page {page_num+1}: {str(e)}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {pdf_path}")
        print(f"{'='*80}")
        print(f"Total images processed: {total_images}")
        print(f"Charts detected: {charts_detected}")
        print(f"Tables detected: {tables_detected}")
        print(f"Regular images: {regular_images}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
    finally:
        if 'doc' in locals():
            doc.close()

def main():
    parser = argparse.ArgumentParser(description="Test improved chart and table detection")
    parser.add_argument("pdf_path", help="Path to the PDF file to analyze")
    parser.add_argument("--output", "-o", help="Output directory for saving annotated images", default="detection_results")
    parser.add_argument("--no-save", action="store_true", help="Don't save annotated images")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return 1
    
    extract_and_analyze_images(args.pdf_path, args.output, not args.no_save)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 