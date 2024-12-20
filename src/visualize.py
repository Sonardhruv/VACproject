import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from preprocess import preprocess_data
import joblib

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def visualize_results(data_path, model_path):
    # Preprocess data
    X, y = preprocess_data(data_path)

    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the model
    model = joblib.load(model_path)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Visualize confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names=['Good', 'Poor'])

if __name__ == "__main__":
    # Paths
    data_path = "data/water_dataX.csv"
    model_path = "models/water_quality_model.pkl"

    # Visualize the results
    visualize_results(data_path, model_path)
