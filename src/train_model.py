import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.preprocess import preprocess_data


def train_and_save_model(data_path, model_path):
    # Preprocess data
    X, y = preprocess_data(data_path)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
