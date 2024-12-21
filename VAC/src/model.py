from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

logger = logging.getLogger(__name__)

class WaterQualityModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")

    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        y_pred = self.model.predict(X_test)
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def save_model(self, path):
        """Save the model."""
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path):
        """Load a trained model."""
        instance = cls()
        instance.model = joblib.load(path)
        return instance
