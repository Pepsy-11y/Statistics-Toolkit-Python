# Python Statistics Toolkit

## 1. Project Overview
This project implements a Python-based Statistics Toolkit for analysing numerical data from CSV files. 
It calculates descriptive statistics, identifies outliers, validates data, and generates statistical reports.

The project also uses NumPy to verify the accuracy of calculations performed using standard Python.

## 2. Project Structure

```text
day3-stats-toolkit/
│
├── notebooks/
│   ├── stats_toolkit.py
|   └── test_stats_toolkit.py
│
├── data/
│   └── agent_deposits.csv
│
├── output/
│   └── statistics_report.txt
│
│
└── README.md
```

## 3. Key Features

- CSV data loading and validation.
- Handling missing and invalid values.
- Calculation of mean, median and mode.
- Sample and population variance.
- Sample and population standard deviation.
- Quartile calculations and Interquartile Range (IQR).
- Detection and removal of outliers.
- Statistical summary and report generation.
- NumPy verification of statistical calculations.
- Unit testing to validate functionality.

## 4. Code Implementation

### A. Importing Libraries

```python
import csv
import math
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
```

### B. Creating the StatisticsAnalyzer Class

```python
class StatisticsAnalyzer:

    def __init__(self, values, name="values"):
        self.values = [float(v) for v in values]
        self.name = name
        self.rejected = []
```

The class stores numerical values and provides methods for statistical analysis.

### C. Calculating Statistics

```python
def mean(self):
    return sum(self.values) / len(self.values)


def median(self):
    num = sorted(self.values)
    no = len(num)
    middle = no // 2

    if no % 2 == 0:
        return (num[middle - 1] +
                num[middle]) / 2

    num[middle]
```

Other methods calculate mode, variance, standard deviation, quartiles and outliers.

### D. NumPy Verification

The comparison function is implemented outside the `StatisticsAnalyzer` class.

```python
def compare_with_numpy(
    analyzer: StatisticsAnalyzer
) -> dict[str, bool]:

    data = analyzer.values

    comparisons = {
        "mean": (analyzer.mean(), np.mean(data)),
        "median": (analyzer.median(), np.median(data)),
        "variance": (
            analyzer.variance(),
            np.var(data, ddof=1)
        ),
        "std_dev": (
            analyzer.std_dev(),
            np.std(data, ddof=1)
        )
    }

    results = {}

    for statistics,(python_result,numpy_result) in comparisons.items()::

        if np.isclose(python_result, numpy_result):
            results[statistics] = True
        else:
            results[statistics] = False

    return results
```

*Note: This example assumes the class uses sample variance and sample standard deviation for the comparison.*

## 5. Running the Project

Install NumPy:

```bash
uv add numpy
```

Run the project:

```bash
uv run python notebooks/stats_toolkit.py
```

Run unit tests:

```bash
uv run python -m unittest discover -s tests -v
```

## 6. Output

The project generates a statistical report containing:
- Number of valid and rejected records.
- Mean, median and mode.
- Variance and standard deviation.
- Quartiles and IQR.
- Identified outliers.
- Statistical summary.

The report is saved in:

```text
output/statistics_report.txt
```

## 7. Technologies Used

- Python
- NumPy
- CSV module
- Object-Oriented Programming (OOP)
- Unit Testing

## 8. Conclusion

This project demonstrates the application of Python in statistical data analysis, data validation, and automated 
reporting. It also strengthens understanding of statistical concepts by implementing calculations manually and 
comparing their results with NumPy
