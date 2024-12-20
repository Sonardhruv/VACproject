import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, encoding='latin1')

    # Separate numeric and non-numeric columns
    numeric_columns = [
        'Temp', 'D.O. (mg/l)', 'PH', 'CONDUCTIVITY (µmhos/cm)', 
        'B.O.D. (mg/l)', 'NITRATENAN N+ NITRITENANN (mg/l)',
        'FECAL COLIFORM (MPN/100ml)', 'TOTAL COLIFORM (MPN/100ml)Mean'
    ]
    
    # Select only numeric columns for scaling
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, invalid values become NaN

    # Handle missing values by filling with the median for numeric columns
    data.loc[:, numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    # Handle missing values for non-numeric columns, if any (you can use fillna with a specific value or drop rows)
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    data[non_numeric_columns] = data[non_numeric_columns].fillna('Unknown')  # You can replace with 'Unknown' or other values
    
    # Add a new target column based on conditions (example based on D.O. levels)
    target = 'Water_Quality'
    data[target] = data['D.O. (mg/l)'].apply(lambda x: 'Good' if x >= 6 else 'Poor')

    # Normalize numeric features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[numeric_columns])
    y = data[target]

    return X, y
