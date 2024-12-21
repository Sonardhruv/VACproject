import unittest
import numpy as np
from src.model import WaterQualityModel
import os

class TestWaterQualityModel(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.model = WaterQualityModel()

        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 8)  # 100 samples, 8 features
        self.y_train = np.random.choice(['Good', 'Poor'], size=100)
        self.X_test = np.random.rand(20, 8)   # 20 samples for testing
        self.y_test = np.random.choice(['Good', 'Poor'], size=20)

    def test_initialization(self):
        """Test if model initializes correctly."""
        self.assertIsNotNone(self.model.model)

    def test_train(self):
        """Test model training."""
        self.model.train(self.X_train, self.y_train)
        # Check if model is fitted
        self.assertTrue(hasattr(self.model.model, 'classes_'))

    def test_evaluate(self):
        """Test model evaluation."""
        self.model.train(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_test, self.y_test)

        # Check if metrics are returned
        self.assertIn('classification_report', metrics)
        self.assertIn('confusion_matrix', metrics)

    def test_save_load_model(self):
        """Test model saving and loading."""
        # Train model
        self.model.train(self.X_train, self.y_train)

        # Save model
        test_model_path = 'test_model.pkl'
        self.model.save_model(test_model_path)

        # Check if file exists
        self.assertTrue(os.path.exists(test_model_path))

        # Load model
        loaded_model = WaterQualityModel.load_model(test_model_path)

        # Test prediction with loaded model
        pred1 = self.model.model.predict(self.X_test)
        pred2 = loaded_model.model.predict(self.X_test)

        # Compare predictions
        np.testing.assert_array_equal(pred1, pred2)

        # Clean up
        os.remove(test_model_path)

if __name__ == '__main__':
    unittest.main()
