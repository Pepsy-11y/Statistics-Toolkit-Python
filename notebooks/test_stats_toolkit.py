"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest)."""
import unittest

from stats_toolkit import StatisticsAnalyzer


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        analyzer=StatisticsAnalyzer([10,20,20,40,50])
        self.assertEqual(analyzer.mean(),28)

    def test_median_even_and_odd_lengths(self):
        odd=StatisticsAnalyzer([10,20,20,30,50])
        even=StatisticsAnalyzer([10,20,30,50])

        self.assertEqual(odd.median(),20)
        self.assertEqual(even.median(),25)

    def test_mode_with_ties(self):
        analyzer=StatisticsAnalyzer([10,10,20,20,50])
        self.assertEqual(analyzer.mode(),[10,20])

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
           StatisticsAnalyzer([]) 

class TestSpread(unittest.TestCase):
    def test_sample_vs_population_variance(self):
        analyzer=StatisticsAnalyzer([1,2,3,4])
        self.assertAlmostEqual(analyzer.variance(sample=True),1.6666667)
        self.assertAlmostEqual(analyzer.variance(sample=False),1.25)

    def test_std_dev_single_value(self):
        analyzer=StatisticsAnalyzer([10])
        self.assertEqual(analyzer.std_dev(),0.0)


class TestOutliers(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        analyzer=StatisticsAnalyzer([10,20,40,30,50])
        result=analyzer.iqr_outliers()
        self.assertEqual(result["outliers"],[])

    def test_detects_obvious_outlier(self):
       analyzer=StatisticsAnalyzer([10,10,20,20,100])
       result=analyzer.iqr_outliers()
       self.assertIn(100,result["outliers"]) 


class TestLoading(unittest.TestCase):
    def test_invalid_and_missing_values_are_rejected(self):
        import tempfile
        import os

        csv_content=("amount\n""100\n""invalid\n""\n""200\n""300\n")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            newline="",
            encoding="utf-8"
        ) as file:
            file.write(csv_content)
            path=file.name

        try:
            analyzer=StatisticsAnalyzer.from_csv(
                path,column="amount"
            )

            self.assertEqual(analyzer.values,[100,200,300])
            self.assertEqual(len(analyzer.rejected),2)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
