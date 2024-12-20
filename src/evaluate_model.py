import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import preprocess_data

def evaluate_model(data_path, model_path):
    # Preprocess the data
    X, y = preprocess_data(data_path)

    # Split the data (ensure the split matches training conditions)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the trained model
    model = joblib.load(model_path)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluation metrics
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    # Paths
    data_path = "data/water_dataX.csv"
    model_path = "models/water_quality_model.pkl"

    # Evaluate the model
    evaluate_model(data_path, model_path)
