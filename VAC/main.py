from src.preprocessor import WaterQualityPreprocessor
from src.model import WaterQualityModel
from sklearn.model_selection import train_test_split
import logging

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Paths
    data_path = "data/raw/water_dataX.csv"
    model_path = "models/water_quality_model.pkl"

    try:
        # Initialize
        preprocessor = WaterQualityPreprocessor()
        model = WaterQualityModel()

        # Process data
        X, y = preprocessor.preprocess_data(data_path)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train and evaluate
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        print(f"Classification Report:\n{metrics['classification_report']}")

        # Save model
        model.save_model(model_path)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
