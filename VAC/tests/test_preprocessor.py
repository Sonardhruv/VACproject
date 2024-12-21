import unittest
import pandas as pd
import numpy as np
from src.preprocessor import WaterQualityPreprocessor

class TestWaterQualityPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.preprocessor = WaterQualityPreprocessor()

        # Create sample test data
        self.test_data = pd.DataFrame({
            'Temp': [25.5, 26.0, np.nan, 24.5],
            'D.O. (mg/l)': [6.5, 5.5, 7.0, 4.5],
            'PH': [7.2, 6.8, 7.5, np.nan],
            'CONDUCTIVITY (µmhos/cm)': [350, np.nan, 400, 380],
            'B.O.D. (mg/l)': [2.5, 3.0, 2.0, 3.5],
            'NITRATENAN N+ NITRITENANN (mg/l)': [1.5, 1.8, np.nan, 1.6],
            'FECAL COLIFORM (MPN/100ml)': [500, 600, 450, np.nan],
            'TOTAL COLIFORM (MPN/100ml)Mean': [1200, np.nan, 1000, 1100]
        })

        # Save test data to temporary CSV
        self.test_data.to_csv('test_data.csv', index=False)

    def test_initialization(self):
        """Test if preprocessor initializes correctly."""
        self.assertIsNotNone(self.preprocessor.scaler)
        self.assertIsInstance(self.preprocessor.numeric_columns, list)

    def test_preprocess_data(self):
        """Test data preprocessing."""
        X, y = self.preprocessor.preprocess_data('test_data.csv')

        # Check shapes
        self.assertEqual(X.shape[0], len(self.test_data))
        self.assertEqual(X.shape[1], len(self.preprocessor.numeric_columns))

        # Check if missing values are handled
        self.assertFalse(np.isnan(X).any())

        # Check if scaling is working (values between 0 and 1)
        self.assertTrue((X >= 0).all() and (X <= 1).all())

        # Check if water quality labels are created correctly
        self.assertTrue(all(label in ['Good', 'Poor'] for label in y))

    def test_invalid_file(self):
        """Test handling of invalid file."""
        with self.assertRaises(Exception):
            self.preprocessor.preprocess_data('nonexistent_file.csv')

    def tearDown(self):
        """Clean up after tests."""
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
