# Intelligent Pharma-Context Engine

An end-to-end AI pipeline for extracting, verifying, and enriching pharmaceutical information from medicine label images.

## Overview

The Intelligent Pharma-Context Engine processes images of medicine bottles and strips to extract critical information including drug names, manufacturers, composition, dosage, and barcodes. The system then verifies this data against authoritative sources (openFDA and RxNorm) and enriches it with supplemental clinical information not present on physical labels.

## Key Features

### 🔍 Advanced Detection & Extraction
- **Object Detection**: YOLO-based medicine bottle and label localization
- **Multi-Engine OCR**: Dual-mode text extraction (EasyOCR + Tesseract)
- **Barcode Decoding**: Automatic barcode and QR code extraction
- **Layout-Agnostic Recognition**: Field extraction without fixed coordinates

### 🛡️ Robust Preprocessing
- **Glare Removal**: Handles specular reflections on blister packs
- **Perspective Correction**: Rectifies text warping on curved bottles
- **Adaptive Enhancement**: CLAHE-based contrast improvement
- **Noise Reduction**: Preserves text edges while denoising

### ✅ Intelligent Verification
- **Multi-Source Validation**: Cross-references openFDA and RxNorm databases
- **Fuzzy Entity Resolution**: Corrects OCR errors using Levenshtein distance
- **Barcode Validation**: Uses NDC codes to validate/correct text extraction
- **Discrepancy Detection**: Identifies conflicts between data sources

### 📊 Clinical Enrichment
- **Drug Interactions**: Retrieved from FDA label data
- **Safety Warnings**: Contraindications, precautions, black box warnings
- **Storage Requirements**: Proper handling and storage conditions
- **Dosage Guidelines**: Administration instructions

## Architecture

Input Image
↓
[Preprocessing]

Glare removal

Perspective correction

Contrast enhancement

Denoising & sharpening
↓
[Detection]

YOLO object detection

ROI extraction
↓
[Extraction]

OCR (EasyOCR/Tesseract)

Barcode decoding

Entity recognition
↓
[Verification]

openFDA query

RxNorm validation

Fuzzy matching

Barcode cross-check
↓
[Enrichment]

Clinical data retrieval

Safety information

Storage requirements

Drug interactions
↓
Enriched JSON Output

## Installation

### Prerequisites
- **OS**: Windows 10/11
- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU (optional, for faster processing)

### Step 1: Clone or Extract Project

Extract the project to `D:\Pharma Project` or clone if using git.

### Step 2: Create Virtual Environment

Open PowerShell or Command Prompt:

```cmd
cd "D:\Pharma Project"
python -m venv venv