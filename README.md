NumPy Analyzer

A menu-driven command-line toolkit for creating and analyzing one-, two-, and three-dimensional NumPy arrays. It demonstrates OOP, array indexing and slicing, element-wise math, dot and matrix products, combining/splitting, search/sort/filter, aggregate statistics, percentiles, and correlation.

## Requirements

- Python 3.10+
- NumPy (installed from `requirements.txt`)

## Run (Windows PowerShell)

From this project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\numpy_analyzer.py
```

If activation is restricted, invoke `.venv\Scripts\python.exe` directly for installation and running.

macOS/Linux: activate with `source .venv/bin/activate`, install requirements, then run `python src/numpy_analyzer.py`.

## Menu overview

1. Create a 1D, 2D, or 3D numeric array, then inspect it with indexing/slicing.
2. Perform element-wise addition, subtraction, multiplication, division, dot product, or matrix multiplication.
3. Concatenate compatible arrays or split an array into chunks.
4. Search for a value, sort ascending/descending, or filter using a comparison.
5. Calculate sum, mean, median, standard deviation, variance, min/max, percentiles, and correlation between equal-length arrays.
6. Exit.

Arrays created in the session stay in the analyzer's in-memory collection. Choose them by their displayed index. The program validates dimensions and shapes and reports invalid operations without crashing.

## Assumptions

- Array entries are real numbers entered as whitespace-separated values.
- All arrays must contain at least one value; rows, columns, and depth must be positive integers.
- Element-wise arithmetic requires identical shapes. Division by zero is rejected.
- Dot product and matrix multiplication require compatible dimensions.
- Correlation uses flattened arrays of equal length; constant arrays do not have a defined correlation coefficient.
- Percentile input is between 0 and 100 inclusive.
