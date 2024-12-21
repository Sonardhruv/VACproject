import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.numeric_columns = [
            'Temp', 'D.O. (mg/l)', 'PH', 'CONDUCTIVITY (µmhos/cm)',
            'B.O.D. (mg/l)', 'NITRATENAN N+ NITRITENANN (mg/l)',
            'FECAL COLIFORM (MPN/100ml)', 'TOTAL COLIFORM (MPN/100ml)Mean'
        ]

    def preprocess_data(self, file_path):
        """Preprocess water quality data."""
        try:
            # Load data
            data = pd.read_csv(file_path, encoding='latin1')

            # Handle numeric columns
            data[self.numeric_columns] = data[self.numeric_columns].apply(
                pd.to_numeric, errors='coerce'
            )

            # Fill missing values
            data[self.numeric_columns] = data[self.numeric_columns].fillna(
                data[self.numeric_columns].median()
            )

            # Create quality labels
            data['Water_Quality'] = data['D.O. (mg/l)'].apply(
                lambda x: 'Good' if x >= 6 else 'Poor'
            )

            # Scale features
            X = self.scaler.fit_transform(data[self.numeric_columns])
            y = data['Water_Quality']

            return X, y

        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
