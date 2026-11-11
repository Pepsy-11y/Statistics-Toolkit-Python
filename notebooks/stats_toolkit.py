"""
Day 3 - Statistical Analysis Toolkit
Squad Python Specialization - Data Science, AI & Analytics

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""
import csv
import math
import argparse
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
    valid_values=[]
    rejected=[]

    path=Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file unavailable: {path}")
    with path.open(mode="r",newline="",encoding="utf-8-sig") as file:
        read_file=csv.reader(file)
        heads=next(read_file,None)

        if heads is None:
            raise ValueError("File has no header or missing.")
        if column not in heads:
            raise ValueError(
                f"Column {column} not found in file"
                f"Available columns {heads}"
            )

        column_index=heads.index(column)

        for row_number,row in enumerate(read_file,start=2):

            try:
                if not row or not any(value.strip() for value in row):
                    raise ValueError("Missing value")
                raw_value=row[column_index].strip()

                if not raw_value:
                    raise ValueError("Missing value")
                number=float(raw_value)
                if not math.isfinite(number):
                    raise ValueError

                valid_values.append(number)
            except (ValueError,TypeError,IndexError):
                raw_value=(row[column_index] if len(row)>column_index else None)
                rejected.append((row_number,raw_value))

    return valid_values,rejected


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate mean, value is invalid")
    return sum(values)/len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate median, value is invalid")
    num=sorted(values)
    no=len(num)
    middle=no//2
    if no%2==1:
        return num[middle]
    else:
        return (num[middle-1]+num[middle])/2


def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""
    if not values:
        return []
    occurrence={}
    for value in values:
        occurrence[value]=occurrence.get(value,0)+1
    high_occurrence=max(occurrence.values())

    if high_occurrence==1:
        return []
    return sorted(
        value for value,occurrence in occurrence.items()
        if occurrence==high_occurrence
    )

def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    n=len(values)
    if n==0:
        raise ValueError ("Cannot calculate variance with no values")
    if sample and n<2:
        raise ValueError("Sample variance requires two values at the least")
    average=mean(values)

    sqr_diff=[
        (value-average)**2 for value in values
    ]
    d=n-1 if sample else n

    return sum(sqr_diff)/d


def std_dev(values: list[float], sample: bool = True) -> float:
    return math.sqrt(variance(values,sample))


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3). Document which quartile method you chose."""
    if not values:
        raise ValueError ("Cannot calculate quartiles with no values")

    num=sorted(values)
    no=len(num)

    def percentile(p:float):
        place=(no-1)*p
        lower_index=math.floor(place)
        upper_index=math.ceil(place)

        if lower_index==upper_index:
            return num[lower_index]
        diff=place-lower_index

        return (
            num[lower_index] + diff*(
            num[upper_index]-num[lower_index]
            )
        )
    q1=percentile(0.25)
    q2=percentile(0.50)
    q3=percentile(0.75)

    return q1,q2,q3


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    if not values:
        raise ValueError ("Unable to identify outliers with no values")
    if k < 0:
       raise ValueError ("Unable to identify outliers with negative values")
    q1, q2, q3=quartiles(values)

    iqr=q3-q1
    lower_fence=q1 - (k*iqr)
    upper_fence=q3 + (k*iqr)

    outliers=[
        value for value in values
        if value < lower_fence or value > upper_fence
    ]

    return {
        "q1":q1,
        "q2":q2,
        "iqr":iqr,
        "lower_fence":lower_fence,
        "upper_fence":upper_fence,
        "outliers":outliers
    }


class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values: list[float], name: str = "values") -> None:
        if not values:
            raise ValueError("Requires data")
        if not all(
            isinstance(value,(int,float))
            and not isinstance(value,bool)
            and math.isfinite(value)
            for value in values
        ):
            raise ValueError("Value must be a finite number")
        self.values=[float(value) for value in values]
        self.name=name
        self.rejected=[]
            

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        values,rejected=load_column(path,column)
        if not values:
            raise ValueError("Valid values were not found")
        statsanalyzer=cls(values,column)
        statsanalyzer.rejected=rejected

        return statsanalyzer

    def mean(self) ->float:
        return mean(self.values)

    def median(self) ->float:
        return median(self.values)

    def mode(self) ->list[float]:
        return mode(self.values)

    def variance(self, sample: bool = True) -> float:
        return variance(self.values,sample)

    def std_dev(self, sample: bool = True) -> float:
        if len(self.values)==1:
            return 0.0
        return math.sqrt(self.variance(sample))

    def quartiles(self) -> tuple[float, float, float]:
        return quartiles(self.values)

    def iqr_outliers(self, k: float = 1.5) -> dict:
        return iqr_outliers(self.values,k)

    def summary(self) -> dict:
        q1,q2,q3=self.quartiles()
        outlier_answers=self.iqr_outliers()

        return {
            "column":self.name,
            "count":len(self.values),
            "mean":self.mean(),
            "median":self.median(),
            "mode":self.mode(),
            "sample_variance":self.variance(),
            "sample_std_dev":self.std_dev(),
            "population_variance":self.variance(sample=False),
            "population_std_dev":self.std_dev(sample=False),
            "q1":q1,
            "q2":q2,
            "q3":q3,
            "iqr":outlier_answers["iqr"],
            "lower_fence":outlier_answers["lower_fence"],
            "upper_fence":outlier_answers["upper_fence"],
            "outliers":outlier_answers["outliers"]
        }

    def without_outliers(self) -> "StatisticsAnalyzer":
        answers=self.iqr_outliers()
        outlier_values=answers["outliers"]

        clean_values=self.values.copy()

        for value in outlier_values:
            clean_values.remove(value)
        if not clean_values:
            raise ValueError("No value present after removing outliers")

        return StatisticsAnalyzer(clean_values,self.name)

    def report(self,report_path: Path,comparison: dict=None) -> None:
        results=self.summary()
        output_dir=Path(__file__).resolve().parent.parent/"output"
        output_dir.mkdir(parents=True,exist_ok=True)
        report_path=output_dir/"stats_report.txt"
        with open(report_path,"w",encoding="utf-8") as file:
            file.write("\nStatistical Analysis Report")
            file.write("="*40+"\n")
            file.write(f"Column: {self.name}\n")
            file.write(f"Valid Count: {len(self.values)}\n")
            file.write(f"Rejected Count: {len(self.rejected)}\n")

            file.write("\nMeasures of Central Tendencies\n")
            file.write(f"Mean: {results['mean']:.2f}\n")
            file.write(f"Median: {results['median']:.2f}\n")
            file.write(f"Mode: {results['mode']}\n")

            file.write("\nMeasures of Variability\n")
            file.write(f"Sample Variance: {results['sample_variance']:.2f}\n")
            file.write(f"Sample standard deviation: {results['sample_std_dev']:.2f}\n")
            file.write(f"Population Variance: {results['population_variance']:.2f}\n")
            file.write(f"Population standard deviation: {results['population_std_dev']:.2f}\n")

            file.write("\nQuartiles and Outliers\n")
            file.write(f"Q1: {results['q1']:.2f}\n")
            file.write(f"Q2: {results['q2']:.2f}\n")
            file.write(f"Q3: {results['q3']:.2f}\n")
            file.write(f"IQR: {results['iqr']:.2f}\n")
            file.write(f"Outliers: {results['outliers']}\n")

            file.write("\nNumpy Comparison\n")

            if comparison is not None:
                for statistic, matched in comparison.items():
                    status="MATCH" if matched else "MISMATCH"
                    file.write(f"{statistic}: {status}\n")
            file.write("\nRejected Values\n")

            for row_number,raw_value in self.rejected:
                file.write("Row {row_number}: {raw_value!r}\n")

            print(f"\nReport saved : {report_path}")



def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    import numpy as np

    data=np.array(analyzer.values,dtype=float)

    if len(data)<2:
        raise ValueError("Two values are required")
    
    comparisons={
        "mean":(analyzer.mean(),np.mean(data)),
        "median":(analyzer.median(),np.median(data)),
        "sample_variance":(analyzer.variance(),np.var(data,ddof=1)),
        "sample_std_dev":(analyzer.std_dev(),np.std(data,ddof=1)),
        "q1":(analyzer.quartiles()[0],np.percentile(data,25)),
        "q2":(analyzer.quartiles()[1],np.percentile(data,50)),
        "q3":(analyzer.quartiles()[2],np.percentile(data,75))
    }

    results={}
    for statistics,(python_result,numpy_result) in comparisons.items():

        if np.isclose(python_result,numpy_result):
            results[statistics]=True
        else:
            results[statistics]=False

    return results


def main():

    try:
        column="daily_deposits_kes"

        analyzer=StatisticsAnalyzer.from_csv(DATA_PATH,column)
        
        print("\nStatistical Summary")
        for statistic,result in analyzer.summary().items():
            print(f"{statistic}: {result}")
            
        comparison=compare_with_numpy(analyzer)
        print("\nNumpy Comparison")
        print("="*40+"\n")
        for statistic,matched in comparison.items():
            print(f"{statistic}: {'MATCH' if matched else 'MISMATCH'}")
                
        analyzer.report(
            Path("output")/"stats_report.txt",
            comparison
        )

    except (FileNotFoundError,ValueError,KeyError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
