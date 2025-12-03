# Automated Cell Counting

## Overview
This project utilizes **OpenCV** and Python to detect and count cells in microscopic images. It was developed to solve the problem of overlapping cells and image noise using morphological operations.

## Methodology
1. **Preprocessing:** Gaussian Blur and Negative transformation.
2. **Segmentation:** Binary Thresholding.
3. **Refinement:** FloodFill algorithm to fill holes, followed by Morphological Opening and Erosion to separate connected cells.
4. **Counting:** Contour detection.

## Result
* **Count:** 236 Cells detected successfully.

## Usage
```bash
pip install -r requirements.txt
python main.py
