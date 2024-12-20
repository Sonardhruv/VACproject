from src.train_model import train_and_save_model

if __name__ == "__main__":
    # Paths
    data_path = "data/water_dataX.csv"
    model_path = "models/water_quality_model.pkl"

    # Train and save the model
    train_and_save_model(data_path, model_path)
